from typing import Dict, List
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
import torch.utils
from transformers import BertTokenizer
from torch.utils.data import Dataset
import os
import torch
import torch.nn.functional as F
from torchvision.transforms import v2 as T
import numpy as np
from torch.utils.data import Dataset
import lmdb
import io
from operator import itemgetter
import random
from turbojpeg import TurboJPEG, TJPF_GRAY
jpeg = TurboJPEG()

tokenizer = BertTokenizer.from_pretrained("bert-base-cased", use_fast=True)

class LRS2BaseDataset(Dataset):
    def __init__(self,data_dir, max_words_len, max_video_len, mode, split_stride:int, max_data_num:int=None):
        '''
        data_dir: str, path to the data directory
        max_words_len: int, max length of the words
        max_video_len: int, max length of the video
        mode: str, pretrain/preval/train/val/test
        split_stride: int, word stride to split the video
        max_data_num: int, maximum number of data to load
        '''
        self.data_dir = data_dir
        datalist = np.load(os.path.join(data_dir, 'datalist.npz'), allow_pickle=True)

        # choose the datalist according to the mode
        if mode == "pretrain":
            self.datalist = datalist['pretrain_datalist'].tolist()
        elif mode == "preval":
            self.datalist = datalist['preval_datalist'].tolist()
        elif mode == 'train':
            self.datalist = datalist['train_datalist'].tolist()
        elif mode == 'test':
            self.datalist = datalist['test_datalist'].tolist()
        elif mode == 'val':
            self.datalist = datalist['val_datalist'].tolist()
        else:
            raise Exception('Invalid mode type: ', mode)
        
        print("Loading datalist for '{}' mode, {} videos in total".format(mode, len(self.datalist) if max_data_num==-1 else min(max_data_num, len(self.datalist))))
        datalist = sorted(self.datalist, key=itemgetter('video_len'), reverse=True)[:min(max_data_num, len(self.datalist))]
        
        new_datalist = []

        print("Making datalist with max_words_len={}, max_video_len={}, split_stride={}".format(max_words_len, max_video_len, split_stride))
        for video_item in datalist:
            split_videos = self.split_video_sequence(video_item, max_video_len, max_words_len, split_stride)
            new_datalist.extend(split_videos)

        print("Total {} videos after processing".format(len(new_datalist)))
        self.datalist = new_datalist

    @staticmethod
    def split_video_sequence(video, max_video_len, max_words_num, stride)->List[Dict]:
        words = video['words']
        split_videos = []
        num_words = len(words)
        
        # 按照 stride 步长遍历单词
        for start_idx in range(0, num_words, stride):
            # 初步提取最多 max_words_num 个单词
            end_idx = min(start_idx + max_words_num, num_words)
            
            # 获取当前子序列的开始和结束时间
            current_words = words[start_idx:end_idx]
            start_time = current_words[0]['start']
            end_time = current_words[-1]['end']
            
            # 如果时间超过 max_video_len，减少单词数量
            while end_time - start_time > max_video_len and len(current_words) > 1:
                current_words = current_words[:-1]  # 去掉最后一个单词
                end_time = current_words[-1]['end']
            
            # 确保最后的子序列符合时长要求
            if end_time - start_time > max_video_len:
                continue
            
            # 提取子句
            sentence = " ".join([word['word'] for word in current_words])
            
            # 构建子视频序列，保持 id 不变
            split_video = {
                'id': video['id'],
                'sentence': sentence,
                'words': current_words,
                'video_len': video['video_len'],
                'start_frame': start_time,
                'end_frame': end_time,
                'path': video['path']
            }
            split_videos.append(split_video)
        return split_videos

    def __len__(self):
        return len(self.datalist)
    
    def __getitem__(self):
        raise NotImplementedError

    

class LRS2ImageDataset(Dataset):
    def __init__(self, baseset:LRS2BaseDataset, lmdb_name:str, augment:bool=False):
        '''
        baseset: LRS2BaseDataset, the base dataset
        lmdb_name: str, name of the lmdb file
        augment: bool, whether to use data augmentation
        '''
        self.data_dir = baseset.data_dir
        self.datalist = baseset.datalist
        self.augment = augment
        self.env = lmdb.open(os.path.join(self.data_dir,lmdb_name),readonly=True,lock=False,max_spare_txns=50,readahead=False)

        self.WIDTH = 112
        self.DataAug = T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomErasing(p=0.5,
                            scale=(0.33,0.33),
                            ratio=(0.3,3.3),
                            value="random"
                            )
        ])
        self.transform = T.Compose([
            T.ConvertImageDtype(torch.float32),
            T.Normalize([0.4161],[0.1688])
        ])

    def __getitem__(self, index):
        '''
        return: frames, sentence
        '''
        item = self.datalist[index]
        frames = []
        with self.env.begin() as txn:
            for i in range(item['start_frame'],item['end_frame']+1):
                key = f"{item['id']}-{i}".encode()
                value = txn.get(key)
                frames.append(jpeg.decode(value, pixel_format=TJPF_GRAY))
        frames = np.stack(frames) # (T,H,W,C)
        frames = torch.from_numpy(frames).float()
        frames = frames.permute(0,3,1,2).contiguous() # (T,C,H,W)
        if self.augment:
            frames = self.DataAug(frames)
        frames = self.transform(frames)

        sentence = item['sentence']
            
        return frames,sentence

    def __len__(self):
        return len(self.datalist)
    
    def collate_fn(self, batch, tokenizer):
        feats = []
        target = []
        max_feat_len = 0
        
        for feat,sentence in batch:
            target.append(sentence)
            if len(feat) > max_feat_len:
                max_feat_len = len(feat)
                
        assert max_feat_len > 0
        batch_size = len(batch)
        feat_padding_masks = torch.ones((batch_size,max_feat_len), dtype=torch.float)
        
        for index,(feat,_) in enumerate(batch):
            cur_len = len(feat)
            pad_len = max_feat_len - cur_len
            feat = F.pad(feat,(0,0,0,0,0,0,0,pad_len), "constant", 0)
            feats.append(feat)
            feat_padding_masks[index,:cur_len] = 0.

        feats = torch.stack(feats).transpose(1,2).contiguous()
        output=tokenizer(target,padding=True,return_tensors='pt')
        target = output['input_ids']
        target_inp = target[:,:-1]
        target_out = target[:,1:]
        taget_mask = output['attention_mask']
        # teacher forcing, target mask为max_len - 1的下三角方阵
        tgt_attn_mask = nn.Transformer.generate_square_subsequent_mask(target.shape[1]-1).bool()
        target_inp_padding_masks = (1-taget_mask[:,:-1]).bool()
        feat_padding_masks = feat_padding_masks.bool()
        return feats,feat_padding_masks,target_inp,target_out,target_inp_padding_masks,tgt_attn_mask


class LRS2FeatDataset(Dataset):
    def __init__(self,):
        pass

    def __getitem__(self, idx):
        pass





if __name__ == '__main__':
    from functools import partial
    from torch.utils.data import DataLoader
    from tqdm.auto import tqdm
    pretrain_baseset = LRS2BaseDataset(data_dir='../data/LRS2-preprocess', 
                                 max_words_len=30,
                                 max_video_len=250, # 30 250 可覆盖80%的数据
                                 mode='pretrain', 
                                 split_stride=10, 
                                 max_data_num=-1)
    preval_baseset = LRS2BaseDataset(data_dir='../data/LRS2-preprocess',
                                 max_words_len=30,
                                 max_video_len=250,
                                 mode='preval', 
                                 split_stride=10, 
                                 max_data_num=-1)
    pretrain_dataset = LRS2ImageDataset(pretrain_baseset, 'jpeg_lmdb')
    preval_dataset = LRS2ImageDataset(preval_baseset, 'jpeg_lmdb')
    tokenizer = BertTokenizer.from_pretrained("bert-base-cased", use_fast=True)
    pretrain_dataloader = DataLoader(
        dataset = pretrain_dataset,
        batch_size = 16,
        pin_memory = True,
        num_workers = 4,
        shuffle = True,
        collate_fn = partial(pretrain_dataset.collate_fn,tokenizer=tokenizer))
    preval_dataloader = DataLoader(
        dataset = preval_dataset,
        batch_size = 16,
        pin_memory = True,
        num_workers = 4,
        shuffle = True,
        collate_fn = partial(preval_dataset.collate_fn,tokenizer=tokenizer))
    for i in tqdm(pretrain_dataloader):
        continue
    for i in tqdm(preval_dataloader):
        continue
