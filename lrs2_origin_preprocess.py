import cv2 as cv
import os
from tqdm import tqdm
import argparse
from turbojpeg import TurboJPEG, TJPF_GRAY, TJSAMP_GRAY, TJFLAG_PROGRESSIVE, TJFLAG_FASTUPSAMPLE, TJFLAG_FASTDCT
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torchvision import transforms as T
import lmdb
# python .\lrs2_origin_preprocess.py --datalist_path .\lrs2_listed\data.npz --data_prefix .\data\main --origin_lmdb_path .\lmdb_data\

roiSize = 112
jpeg = TurboJPEG()

class LRS2InferenceDataset(Dataset):
    def __init__(self,datalist_path,data_prefix):
        datalist = np.load(datalist_path,allow_pickle=True)
        self.data_prefix = data_prefix        
        new_datalist = []
        # new_datalist.extend(datalist['pretrain_datalist'].tolist())
        # new_datalist.extend(datalist['preval_datalist'].tolist())
        new_datalist.extend(datalist['train_datalist'].tolist())
        new_datalist.extend(datalist['val_datalist'].tolist())
        new_datalist.extend(datalist['test_datalist'].tolist())
        self.datalist = new_datalist
        

        
        
        
    def __getitem__(self, index):
        item = self.datalist[index]
        vidname = os.path.join(self.data_prefix,item['path']+'.mp4')
        video = []
        cap = cv.VideoCapture(vidname)
        while True:
            ret,frame = cap.read()
            if not ret:
                break
            
            frame = cv.resize(frame, (224,224))
            roi = frame[int(112-(roiSize/2)):int(112+(roiSize/2)), int(112-(roiSize/2)):int(112+(roiSize/2))]
            video.append(roi)
        cap.release()
            
        
        # feat = self.feature_extractor(video).squeeze(1).detach().cpu()
        return video,item['id'],item['video_len']
        
    def __len__(self):
        return len(self.datalist)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datalist_path", required=True,help="LRS2数据列表所在路径")
    parser.add_argument("--data_prefix", required=True,help="LRS2数据集所在路径")
    parser.add_argument("--origin_lmdb_path", required=True,help="原始帧lmdb保存路径")
    args = parser.parse_args()
    return args

def collate_fn(x):
    return x[0]

if __name__ == "__main__":
    # COMMIT_CYCLE = 200
    COMMIT_CYCLE = 20
    args = parse_args()
    lrs2_dataset = LRS2InferenceDataset(args.datalist_path,args.data_prefix)
    # loader = DataLoader(lrs2_dataset,batch_size=1,num_workers=8,prefetch_factor=2,collate_fn=lambda x:x[0])
    loader = DataLoader(lrs2_dataset,batch_size=1,num_workers=8,prefetch_factor=2,collate_fn=collate_fn)
    env = lmdb.open(args.origin_lmdb_path,lock=False,map_size=3e11)
    txn = env.begin(write=True)

    for idx,(videos,id,video_len) in enumerate(tqdm(loader)):
        assert len(videos) == video_len
        for f_id,frame in enumerate(videos):
            #success, encoded_frame = cv.imencode('.png', frames)
            frame_name = id + f'-{f_id}'
            encoded_frame = jpeg.encode(frame, jpeg_subsample=TJSAMP_GRAY,quality=100)
            txn.put(frame_name.encode(), encoded_frame)

        if idx % COMMIT_CYCLE == COMMIT_CYCLE-1:
            txn.commit()
            txn = env.begin(write=True)
    txn.commit()
    print('Done')
    
    # if args.feature_type == 'feat':
    #     feat_env = lmdb.open(args.lmdb_path,lock=False,map_size=3e11)
    #     feat_txn = feat_env.begin(write=True)
    #     feature_extractor = VisualFrontend()
    #     ckpt = torch.load(args.visual_frontend_path)
    #     feature_extractor.load_state_dict(ckpt)
    #     feature_extractor = feature_extractor.cuda()
    #     feature_extractor = feature_extractor.eval()
    #     for idx,(i,id,video_len) in enumerate(tqdm(loader)):
    #         i= i[0].cuda()
    #         id = id[0]
    #         CHUNK_SIZE = 500

    #         if i.shape[3] >= CHUNK_SIZE:
    #             feat = []
    #             for j in range(0,i.shape[3],CHUNK_SIZE):
    #                 with torch.no_grad():
    #                     feat_ = feature_extractor(i[:,:,j:j+CHUNK_SIZE]).squeeze(0).cpu()
    #                 feat.append(feat_)
    #             feat = torch.cat(feat,dim=0)

    #         else:
    #             with torch.no_grad():
    #                 feat = feature_extractor(i).squeeze(0).cpu()
    #             assert feat.shape[0] == i.shape[2]
    #             assert feat.shape[0] == video_len
                

    #         buffer = io.BytesIO()
    #         torch.save(feat,buffer)
    #         feat_txn.put(id.encode(), buffer.getvalue())
    #         if idx % COMMIT_CYCLE == 0:
    #             feat_txn.commit()
    #             feat_txn = feat_env.begin(write=True)
    #     feat_txn.commit()


        
    
