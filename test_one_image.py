import argparse
import os
import parser
import random
from time import perf_counter
import torch
from torch import nn, optim
import torchvision
from torchvision import datasets, transforms
from Dataset.transform import ToTensor,Normalize,Compose,UnNormalize,RandomCrop,RandomColorjitter,CenterCrop,Resize
from torch.utils.data import DataLoader
from Dataset.au_dataset import AuDataset
import numpy as np
import torch.nn.functional as F
from process.engine import train_one_epoch, evalutate
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import json

save_path = "checkpoint/0813_res18_mse_e100/best.pth"
#img_path = "/home/dingyan/huguohong/9_10/dataset/FEAFA_ALL/FEAFA_A1_align2/PV02400000362.jpg"
label_json_path = "Test_json_file.json"
data_path = "/home/dingyan/huguohong/9_10/dataset/FEAFA_ALL/FEAFA_A1_align2"
if __name__ == '__main__':
    
    torch.manual_seed(42)
    random.seed(42)
    torch.cuda.manual_seed(42)  # gpu
    np.random.seed(42)  # numpy
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
    transform_val = transforms.Compose([
        #CenterCrop(224),
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        #Normalize(mean =[0.4743, 0.3539, 0.3249],std = [0.2697, 0.2238, 0.2154]),
        ])
    model = torchvision.models.resnet18()
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_features, 24),
        nn.Hardsigmoid()  # 使用 Sigmoid 激活函数将输出限制在0-1之间
        )
    model.load_state_dict(torch.load(save_path))
    model.cuda()
    model.eval()
    with torch.no_grad():
        #import ipdb;ipdb.set_trace()
        # assert os.path.exists(img_path), "{} dose not exist".format(img_path)
        # img = Image.open(img_path).convert('RGB')

        # box_info = np.loadtxt(box_path)
        # if len(box_info)<4:
        #     os.remove(box_path)
        #     xmin, ymin, weight, height = (0, 0, img.shape[0], img.shape[1])
        # else:
        #     xmin, ymin, weight, height = int(box_info[0]), int(box_info[1]), int(box_info[2]), int(box_info[3])
        # xmax = xmin + weight
        # ymax = ymin + height
        # img = img[xmin:xmax, ymin:ymax]
        # print(xmin,ymin, xmax, ymax)

        with open(label_json_path, 'r') as f:
            label_infos = json.load(f)
        n = np.random.randint(0, len(label_infos))
        label_info = label_infos[n]
        label = np.array(label_info['au_reg'])
        img_name = label_info['file_name']
        img_path = os.path.join(data_path, img_name)
#        label = np.loadtxt(label_path,dtype=float)
  

        img = Image.open(img_path).convert('RGB')
        img = transform_val(img)
        label = torch.from_numpy(label).float()
        img = img.unsqueeze(0)
        img = img.cuda()

        ori_img=UnNormalize(mean =[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)
        torchvision.utils.save_image(ori_img[0],'images/test_image.jpg',normalize=True)
        label = label.cuda()
        label = label.unsqueeze(0)
        output = model(img)


        mse_loss = torch.nn.MSELoss()
        loss = mse_loss(output, label)
        mae_value = torch.mean(torch.abs(label-output))   
    print("the output is :", output)
    print("the label is :", label)
    print("the loss is  :", loss)
    print("the mae_value is:", mae_value)
