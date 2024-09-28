import os
from tqdm import tqdm
import argparse
import numpy as np
from math import floor, ceil
import cv2 as cv

# python mk_lrs2_datalist.py --data_path ./data/ --save_path ./lrs2_listed/data.npz

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True,help="LRS2数据集所在路径")
    parser.add_argument("--save_path", required=True,help="数据列表保存路径")
    args = parser.parse_args()
    return args

def mk_datalist(args):


    print("读取数据列表") #先构建数据列表，方便后续取数据
    # with open(os.path.join(args.data_path,"pretrain.txt")) as f:
    #     pretrain_filelist = [prefix.strip() for prefix in f.readlines()]
    with open(os.path.join(args.data_path,"train.txt")) as f:
        train_filelist = [prefix.strip() for prefix in f.readlines()]
    with open(os.path.join(args.data_path,"val.txt")) as f:
        val_filelist = [prefix.strip() for prefix in f.readlines()]
    with open(os.path.join(args.data_path,"test.txt")) as f:
        test_filelist = [prefix.strip().split(" ")[0] for prefix in f.readlines()]
    # pretrain分为pretrain和preval
    # pretrain_filelist, preval_filelist = np.split(pretrain_filelist, [int(0.99 * len(pretrain_filelist))])
    
    # pretrain_filelist = pretrain_filelist.tolist()
    # preval_filelist = preval_filelist.tolist()

    pretrain_datalist = []
    preval_datalist = []
    train_datalist = []
    test_datalist = []
    val_datalist = []

    #pretrain和preval分别读取每个单词的开始和截止
    # def format_pretrain_datalist(filelist:list, datalist:list, args=args):
    #     for path in tqdm(filelist):
    #         f = os.path.join(args.data_path,'pretrain',path+'.mp4')
    #         cap = cv.VideoCapture(f)
    #         video_len = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    #         cap.release()

    #         with open(os.path.join(args.data_path,'pretrain',path + ".txt")) as f:
    #             txt = [i.strip() for i in f.readlines()]
    #             sentence = txt[0][7:]
                
    #             txt = txt[4:]
    #             id = 'pretrain'+'-'+path.replace('/','-')
    #             wordslist = []
    #             for line in txt:
    #                 word, start, end, _ = line.split(" ")
    #                 start, end = floor(float(start) * 25), ceil(float(end) * 25) # 不能同时用floor或ceil，会出现start < end的情况
    #                 wordslist.append(
    #                     {"start": start, "end": end, "word": word}
    #                 )
                
    #         datalist.append({
    #             "id": id,
    #             'sentence':sentence,
    #             "words": wordslist,
    #             "path":"pretrain/"+path,
    #             "video_len": video_len
    #         })

    # format_pretrain_datalist(pretrain_filelist, pretrain_datalist)
    # format_pretrain_datalist(preval_filelist, preval_datalist)
    
    def format_main_datalist(filelist:list, datalist:list, args=args):
        for path in tqdm(filelist):
            f = os.path.join(args.data_path,'main',path+'.mp4')
            cap = cv.VideoCapture(f)
            video_len = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
            cap.release()

            if len(path)!=0:    
                with open(os.path.join(args.data_path,'main',path + ".txt")) as f:
                    txt = [i.strip() for i in f.readlines()]
                    sentence = txt[0][7:]
                    id = 'main'+'-'+path.replace('/','-')
                
            datalist.append({
                "id": id,
                'sentence':sentence,
                "path":"main/"+path,
                "video_len": video_len
            })
            
    #traintestval只用读每个单词即可
    format_main_datalist(train_filelist, train_datalist)
    format_main_datalist(test_filelist, test_datalist)
    format_main_datalist(val_filelist, val_datalist)

    # print(pretrain_datalist[0])
    # print(preval_datalist[0])
    print(train_datalist[0])
    print(test_datalist[0])
    print(val_datalist[0])

        
    # assert len(pretrain_datalist) == len(pretrain_filelist)
    # assert len(preval_datalist) == len(preval_filelist)
    assert len(train_datalist) == len(train_filelist)
    assert len(test_datalist) == len(test_filelist)
    assert len(val_datalist) == len(val_filelist)

    
    
    #保存
    np.savez(
        args.save_path,
        # pretrain_datalist=pretrain_datalist,
        # preval_datalist=preval_datalist,
        train_datalist=train_datalist,
        test_datalist=test_datalist,
        val_datalist=val_datalist,
    )


if __name__ == '__main__':
    args = parse_args()
    mk_datalist(args)