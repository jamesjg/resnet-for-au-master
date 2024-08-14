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
from torch.utils.data import DataLoader, ConcatDataset
from Dataset.au_dataset import AuDataset, AuRafDataset
import numpy as np
import torch.nn.functional as F
from process.engine import inference
from PIL import Image
import matplotlib.pyplot as plt
#from timm import create_model
#from Dataset.get_landmarks import align_face
from utils.get_logger import create_logger



def main(args):
    torch.manual_seed(args.random_seed)
    random.seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)  # gpu
    np.random.seed(args.random_seed)  # numpy

    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
    logger = create_logger(args.log_dir, model_name="resnet50", phase='test')
    transform_val = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    
    
    Raf_dataset = AuRafDataset(data_path = args.data_path, bbox_txt_path=args.bbox_txt_path, transform=transform_val)
    
    val_num = len(Raf_dataset)
    logger.info("val_num : {}".format(val_num))
    
    val_loader = DataLoader(dataset=Raf_dataset,
                            batch_size=args.batch_size,
                            shuffle=False,
                            num_workers=args.num_workers)
    model = torchvision.models.resnet50()
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_features, 24),
        nn.Hardsigmoid()  # 使用 Sigmoid 激活函数将输出限制在0-1之间
        )
    model = model.cuda()
    
    
    model.load_state_dict(torch.load("checkpoint/0813_res50_mse_e100_newdata/best.pth"))
    
    

    au_preds = inference(val_loader,  model,  0, args, logger)
    np.save("raf_au_preds.npy", au_preds)
    import ipdb;ipdb.set_trace()
    logger.info("raf_au_preds.npy saved")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--data_path',type=str,default="/home/dingyan/huguohong/9_10/dataset/Expression/RAF-DB/basic/Image/original")
    parser.add_argument('--bbox_txt_path',type=str,default="raf_bbox.txt")
    parser.add_argument('--print_fq',type=int,default=20)
    
    parser.add_argument('--num_class',type=int,default=24)
    parser.add_argument('--log_dir',type=str,default="log/log_inference_raf")
    parser.add_argument("--num_workers", type=int, default=32)
    #parser.add_argument("--iscrop", type=str, default="_crop")
    args, _ = parser.parse_known_args()
    main(args)