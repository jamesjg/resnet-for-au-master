import argparse
import os
import parser
import random
from time import perf_counter
import torch
from torch import nn, optim
import torchvision
from torchvision import datasets, transforms
#from Dataset.transform import ToTensor,Normalize,Compose,UnNormalize,RandomCrop,RandomColorjitter,CenterCrop,Resize
import numpy as np
import torch.nn.functional as F
from process.engine import inference
from PIL import Image
import matplotlib.pyplot as plt
#from timm import create_model
#from Dataset.get_landmarks import align_face
from utils.get_logger import create_logger
from utils.utils import get_crop_img_from_bbox
import cv2
import numpy as np

def load_imgs(inference_transform, data_path=None, bbox_txt_path=None, img_path=None, bbox=None, dataset='raf'):
    imgs = []

    if dataset == 'raf':
        assert data_path is not None and bbox_txt_path is not None
        with open(bbox_txt_path, 'r') as f:
            data_infos = f.readlines()[2:]
        for data_info in data_infos:
            img_name = data_info.split(' ')[0]
            img_path = os.path.join(data_path, img_name)
            assert os.path.exists(img_path), 'img not exist'
            x, y, w, h = map(float, data_info.split(' ')[1:])
            bbox = np.array([x, y, w, h])
            img = cv2.cvtColor(cv2.imread(img_path),cv2.COLOR_BGR2RGB)
            print("processing image: ", img_name)
            
            img = get_crop_img_from_bbox(bbox, img)
            img = Image.fromarray(img)
            img = inference_transform(img)
            imgs.append(img)
            # if len(imgs) == 512:
            #     break

    if dataset == 'single_ori':

        assert img_path is not None and bbox is not None
        img = cv2.cvtColor(cv2.imread(img_path),cv2.COLOR_BGR2RGB)
        img = get_crop_img_from_bbox(bbox, img)
        img = Image.fromarray(img)
        img = inference_transform(img)
        imgs.append(img)
        
    if dataset == 'single_crop':

        assert img_path is not None
        img  = cv2.cvtColor(cv2.imread(img_path),cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img = inference_transform(img)
        imgs.append(img)
    return imgs


def main(args):
    torch.manual_seed(args.random_seed)
    random.seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)  # gpu
    np.random.seed(args.random_seed)  # numpy

    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
    logger = create_logger(args.log_dir, model_name="resnet50", phase='test')
    inference_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    
    imgs = load_imgs(inference_transform, data_path=args.data_path, bbox_txt_path=args.bbox_txt_path, dataset=args.dataset)     
    
    logger.info("inference_numbers : {}".format(len(imgs)))
    
    model = torchvision.models.resnet50()
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_features, 24),
        nn.Hardsigmoid()  # 使用 Sigmoid 激活函数将输出限制在0-1之间
        )
    model = model.cuda()
    
    
    model.load_state_dict(torch.load("checkpoint/0813_res50_mse_e100_newdata/best.pth"))
    
    au_preds = inference(imgs,  model, args, logger)
    np.save("raf_au_preds_new.npy", au_preds)
    logger.info("raf_au_preds.npy saved")
    
    # json 保存， path: xxxx, AU:xxxx
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--data_path',type=str,default="/home/dingyan/huguohong/9_10/dataset/Expression/RAF-DB/basic/Image/original")
    parser.add_argument('--bbox_txt_path',type=str,default="raf_bbox.txt")
    parser.add_argument('--dataset',type=str,default="raf")
    
    parser.add_argument('--print_fq',type=int,default=20)
    
    
    parser.add_argument('--num_class',type=int,default=24)
    parser.add_argument('--log_dir',type=str,default="log/log_inference_raf")
    parser.add_argument("--num_workers", type=int, default=32)
    #parser.add_argument("--iscrop", type=str, default="_crop")
    args, _ = parser.parse_known_args()
    main(args)
    
