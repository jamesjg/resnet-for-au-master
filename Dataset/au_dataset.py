import random
import torch
from torch.utils.data import Dataset
from torchvision import transforms, utils
from PIL import Image
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import os
import cv2
from Dataset.Landmark_helper import Landmark_helper
from Dataset.face_aligner import FaceAligner
import json
#from utils.detect_face_from_mtcnn_v1 import FaceDetecter
# class AuDataset(Dataset):
#     def __init__(self, data_path, transform=None, iscrop='', detect_face=True, box_transform=True, mode='Train'):  # data_path: train/test
#         super(AuDataset, self).__init__()
#         data_folders = os.listdir(data_path) #["xxxoutput","xxxoutput"...]
#         imgs_path = []
#         labels_path = []
#         loss = 0
#         drop = 0
#         #path_and_name = []
#         for folder in data_folders:
#             item_list = os.listdir(os.path.join(data_path,folder))
#             for item in item_list:
#                 if item.split('.')[-1] == 'auw': #先找label
#                     label_path = os.path.join(data_path,folder,item)  #"../././xxxxx.auw"
#                     #然后读入对应的crop_img
#                     label=np.loadtxt(label_path,dtype=float)
#                     img_path_jpg = os.path.join(data_path,folder,item.split('.')[0]+iscrop+'.jpg')   #".././xxxxx_crop.jpg"
#                     img_path_png = os.path.join(data_path,folder,item.split('.')[0]+iscrop+'.png')
#                     if np.min(label)==0 and mode!="Train":
#                         p=random.uniform(0,1)
#                         if p < 0.5:
#                             drop += 1
#                             continue
                
#                     if os.path.exists(img_path_jpg):
#                         imgs_path.append(img_path_jpg)
#                         labels_path.append(label_path)
#                     elif  os.path.exists(img_path_png):
#                         imgs_path.append(img_path_png)
#                         labels_path.append(label_path)
#                     else:
#                         loss += 1
#         self.transform = transform
#         #self.path_and_name = path_and_name 
#         #import ipdb;ipdb.set_trace()
#         self.face_aligner = FaceAligner()
#         self.labels_path = labels_path
#         self.imgs_path = imgs_path
#         self.loss_num = loss
#         self.iscrop = iscrop 
#         self.drop = drop 
#         self.box_transform = box_transform
#         self.detect_face = detect_face
#         self.mode = mode
#         print(data_path+ " drop :", self.drop)
        
#     def __getitem__(self, idx):

#         #get image
#         label_path = self.labels_path[idx]
#         img_path = self.imgs_path[idx]
#         assert os.path.exists(img_path), "no img found in {}".format(img_path)
#         img = cv2.cvtColor(cv2.imread(img_path),cv2.COLOR_BGR2RGB)
#         #利用特征点检测器处理得到特征点
#         # if not self.iscrop:
#         #     landmark_helper = Landmark_helper(Method_type='dlib')
#         #     landmark, flag = landmark_helper.detect_facelandmark(img)
#         #     #如果检测到人脸
#         #     if flag:
#         #         assert(landmark.shape==(68,2)),'landmark shape is wrong {:}'.format(landmark.shape)
#         #         img, new_landmarks=self.face_aligner.align(img, landmark)
#         if self.detect_face :
#             box_dir = '/media/ljy/ubuntu_disk1/jhy_code/resnet-for-au/face_box_info'
#             box_file_name = img_path.split('/')[-2]+img_path.split('/')[-1].split('.')[0]+'.txt'
#             box_path = os.path.join(box_dir, box_file_name)
#             if not os.path.exists(box_path):
#                 print(box_path)
#                 face_detector = FaceDetecter(net_type="mtcnn", return_type="v1")
#                 box_info = face_detector.detect_img(img_path)  # top-left weight, height
#                 xmin, ymin, weight, height = box_info[0], box_info[1], box_info[2], box_info[3]
#                 xmax = xmin + weight
#                 ymax = ymin + height
#                 with open(box_path, 'w') as f:
#                     f.write("{} {} {} {}".format(xmin, ymin, xmax, ymax))
#             else:
#                 box_info = np.loadtxt(box_path)
#                 if len(box_info)<4:
#                     os.remove(box_path)
#                     xmin, ymin, weight, height = (0, 0, img.shape[0], img.shape[1])
#                 else:
#                     xmin, ymin, weight, height = int(box_info[0]), int(box_info[1]), int(box_info[2]), int(box_info[3])
#                 if self.box_transform and self.mode=="Train":
#                     #对人脸框做数据增强
#                     # 缩小、放大框
#                     weight_scale, height_scale = random.uniform(0.8,1.2),random.uniform(0.8,1.2)
#                     weight, height = int(weight_scale * weight), int(height_scale * height)
#                     #水平移动左右10个像素
#                     p1 = random.randint(-10, 10)
#                     xmin = xmin + p1
#                     #竖直移动上下10个像素
#                     p2 = random.randint(-10, 10)
#                     ymin = ymin + p2
#                 xmax = xmin + weight
#                 ymax = ymin + height
#                 xmin = max(0, xmin)
#                 xmax = min(img.shape[0], xmax)
#                 ymin = max(0, ymin)
#                 ymax = min(img.shape[1], ymax)
#             img = img[xmin:xmax, ymin:ymax]
#         # get label
#         label=np.loadtxt(label_path,dtype=float)
#         #print(label)
#         # transform
#         if self.transform:
#             img, label = self.transform(img, label)
#         assert label.shape[0]==24, 'label shape is {:}'.format(label.shape)
#         # if label[20] > 0.8:
#         #     print(img_path)
#         return img, label.float()
                 
#     def __len__(self):
#         return len(self.labels_path)
    
#     def zero_label_num(self):
#         count = 0
#         for i in self.path_and_name:
#             label_path = i + '.auw'
#             f = open(label_path, 'r')
#             lis = f.readline().split(' ')
#             label = list(map(float, lis))
#             f.close()
#             # transform
#             label = torch.tensor(label)
#             assert label.shape[0]==24, 'label shape is {:}'.format(label.shape)
#             if torch.sum(label) == 0:
#                 count += 1
#                 #print(i)
#         return count, len(self.path_and_name)

class AuDataset(Dataset):
    def __init__(self, data_path, label_json_path, transform=None,  mode='Train'):  # data_path: train/test
        super(AuDataset, self).__init__()
        with open(label_json_path, 'r') as f:
            self.labels_infos = json.load(f)
        # with open(img_names_txt, 'r') as f:
        #     img_names = f.readlines()
        #     img_names = [i.strip() for i in img_names]
        #     print(len(img_names))
        self.img_paths = []
        self.labels= []
        
        #loss_num = 0
        # lis = []
        for label_info in self.labels_infos:
            # dic = {}
            img_name = label_info['file_name']
            # if  not os.path.exists(os.path.join(data_path, img_name)):
            #     'img path {} not exists'.format(os.path.join(data_path, img_name))
            #     loss_num += 1
            #     continue
            
            if os.path.exists(os.path.join(data_path, img_name)):
                self.img_paths.append(os.path.join(data_path, img_name))
                self.labels.append(label_info['au_reg'])
                # dic['file_name'] = img_name
                # dic['au_reg'] = label_info['au_reg']
                # lis.append(dic)
        # with open('{}_json_file.json'.format(mode), 'w') as f:
        #     json.dump(lis, f)
            
        self.transform = transform
        #self.face_aligner = FaceAligner()
        #self.labels_path = labels_path
        #self.imgs_path = imgs_path
        #self.loss_num = loss_num
        #self.iscrop = iscrop 
        #self.drop = drop 
        #self.box_transform = box_transform
        #self.detect_face = detect_face
        self.mode = mode
        #print(data_path+ " drop :", self.drop)
        
    def __getitem__(self, idx):

        #get image and label
        label = self.labels[idx]
        img_path = self.img_paths[idx]
        #img = cv2.cvtColor(cv2.imread(img_path),cv2.COLOR_BGR2RGB)
        img = Image.open(img_path).convert('RGB')
        if  abs(img.size[0] - img.size[1])>=3: 
            print(img.size, img_path)
        label = np.array(label, dtype=np.float32)
        if self.transform:
            img= self.transform(img)
        label = torch.from_numpy(label)
        assert label.shape[0]==24, 'label shape is {:}'.format(label.shape)
        # if label[20] > 0.8:
        #     print(img_path)
        return img, label.float()
                 
    def __len__(self):
        return len(self.labels)
    
    # def zero_label_num(self):
    #     count = 0
    #     for i in self.path_and_name:
    #         label_path = i + '.auw'
    #         f = open(label_path, 'r')
    #         lis = f.readline().split(' ')
    #         label = list(map(float, lis))
    #         f.close()
    #         # transform
    #         label = torch.tensor(label)
    #         assert label.shape[0]==24, 'label shape is {:}'.format(label.shape)
    #         if torch.sum(label) == 0:
    #             count += 1
    #             #print(i)
    #     return count, len(self.path_and_name)