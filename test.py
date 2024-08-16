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
from Dataset.au_dataset import AuDataset
import numpy as np
import torch.nn.functional as F
from process.engine import train_one_epoch, evalutate
from PIL import Image
import matplotlib.pyplot as plt
#from timm import create_model
#from Dataset.get_landmarks import align_face
from utils.get_logger import create_logger
from utils.cal_less_more import pred_less_and_more
def get_each_au_num(loader):
    matrix = torch.zeros((6,24)).cuda()
    for  imgs, labels in loader:
        bs, _ = labels.shape  #bs, 24
        labels = labels.cuda()
        for i in range(bs):
            matrix[0] += torch.eq(labels[i,:], 0)
            matrix[1] += torch.gt(labels[i,:], 0) & torch.lt(labels[i,:], 0.2) | torch.eq(labels[i,:], 0.2)
            matrix[2] += torch.gt(labels[i,:], 0.2) & torch.lt(labels[i,:], 0.4) | torch.eq(labels[i,:], 0.4)
            matrix[3] += torch.gt(labels[i,:], 0.4) & torch.lt(labels[i,:], 0.6) | torch.eq(labels[i,:], 0.6)
            matrix[4] += torch.gt(labels[i,:], 0.6) & torch.lt(labels[i,:], 0.8) | torch.eq(labels[i,:], 0.8)
            matrix[5] += torch.gt(labels[i,:], 0.8) & torch.lt(labels[i,:], 1.0) | torch.eq(labels[i,:], 1.0)
    return matrix
def get_mean_std_value(loader):
    data_sum,data_squared_sum,num_batches = 0,0,0

    for data,_ in loader:
        # data: [batch_size,channels,height,width]
        # 计算dim=0,2,3维度的均值和，dim=1为通道数量，不用参与计算
        data_sum += torch.mean(data,dim=[0,2,3])    # [batch_size,channels,height,width]
        # 计算dim=0,2,3维度的平方均值和，dim=1为通道数量，不用参与计算
        data_squared_sum += torch.mean(data**2,dim=[0,2,3])  # [batch_size,channels,height,width]
        # 统计batch的数量
        num_batches += 1
    # 计算均值
    mean = data_sum/num_batches
    # 计算标准差
    std = (data_squared_sum/num_batches - mean**2)**0.5
    return mean,std

def main(args):
    torch.manual_seed(args.random_seed)
    random.seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)  # gpu
    np.random.seed(args.random_seed)  # numpy

    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
    logger = create_logger(args.log_dir, model_name=args.model, phase='test')
    transform_val = transforms.Compose([
        #CenterCrop(224),
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        #Normalize(mean =[0.4743, 0.3539, 0.3249],std = [0.2697, 0.2238, 0.2154]),
        ])
    
    # file_names = ['FEAFA','Disfa'] #分别是FEAFA-A,FEAFA-B的存储目录名
    # val_file_path = [os.path.join(args.file_path, i, i+'_test') for i in file_names]
    # logger.info("val_file_path:"+ str(val_file_path))
    # val_dataset_list =[AuDataset(data_path=i, transform=transform_val, iscrop=args.iscrop, mode='Test') for i in val_file_path]
    # val_dataset = ConcatDataset(val_dataset_list)
    val_dataset = AuDataset(data_path = args.data_path, label_json_path=args.val_json_path, transform=transform_val, mode='Test')
    
    val_num = len(val_dataset)
    logger.info("val_num : {}".format(val_num))
    
    val_loader = DataLoader(dataset=val_dataset,
                            batch_size=args.batch_size,
                            shuffle=False,
                            num_workers=args.num_workers)
    model = torchvision.models.resnet18()
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_features, 24),
        nn.Hardsigmoid()  # 使用 Sigmoid 激活函数将输出限制在0-1之间
        )
    model = model.cuda()
    
    
    model.load_state_dict(torch.load(args.model_path))
    
    
    if args.criterion == 'mse':
         criterion = torch.nn.MSELoss()
    elif args.criterion == 'bce':
        criterion = torch.nn.BCELoss()
    elif args.criterion == 'smoothl1':
        criterion = torch.nn.SmoothL1Loss()
    val_loss, val_acc, val_mae, _,_,pred, label, non_zero_mae = evalutate(val_loader,  model, criterion, 0, args, logger)
    logger.info("the val_loss = {:.5f}, val_mae = {:.5f}, acc = {:.5f}, non_zero_mae = {:.5f}".format(val_loss, val_mae, val_acc, non_zero_mae))
    pred_less_and_more(pred, label)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--data_path',type=str,default="/home/dingyan/huguohong/9_10/dataset/FEAFA_ALL/FEAFA_A1_align2")
    parser.add_argument('--val_json_path',type=str,default="Test_json_file.json")
    parser.add_argument('--model',type=str,default='resnet')
    parser.add_argument('--model_path',type=str,default="args.model_path")
    
    parser.add_argument('--num_class',type=int,default=24)
    parser.add_argument('--print_fq', type=int, default=20,
                        )
    parser.add_argument('--log_dir',type=str,default="log/log_val")
    parser.add_argument("--num_workers", type=int, default=32)
    #parser.add_argument("--iscrop", type=str, default="_crop")
    parser.add_argument("--criterion", type=str, default="mse")
    args, _ = parser.parse_known_args()
    main(args)