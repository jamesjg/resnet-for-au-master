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


class AuRafDataset(Dataset):
    def __init__(self, data_path, bbox_txt_path, transform=None):  # data_path: train/test
        super(AuRafDataset, self).__init__()
        with open(bbox_txt_path, 'r') as f:
            self.data_infos = f.readlines()[2:]
        self.data_infos = [i.strip() for i in self.data_infos]
        self.img_paths = []
        self.img_names = []
        self.bbox_infos = []

        for data_info in self.data_infos:
            img_name = data_info.split(' ')[0]
            x, y, w, h = map(float, data_info.split(' ')[1:])
            assert os.path.exists(os.path.join(data_path, img_name)), 'img not exist'
        
            self.img_paths.append(os.path.join(data_path, img_name))
            self.bbox_infos.append(np.array([x, y, w, h]))
            self.img_names.append(img_name)
        
        #print(self.img_names)
        self.transform = transform

        
    def __getitem__(self, idx):

        #get image
        img_path = self.img_paths[idx]
        img_name = self.img_names[idx]
        img = cv2.cvtColor(cv2.imread(img_path),cv2.COLOR_BGR2RGB)
        #img = Image.open(img_path).convert('RGB')


        
        # get bbox
        bbox = self.bbox_infos[idx]
        # crop image
        img = self.get_crop_img_from_bbox(bbox, img)
        #img = torch.from_numpy(img).permute(2, 0, 1)
        if self.transform:
            img = Image.fromarray(img)
            img= self.transform(img)

        return img
                 
    def __len__(self):
        return len(self.img_paths)
    
    def get_crop_img_from_bbox(self, bbox, img):
        X1, Y1, w, h = bbox.astype(np.float32)
        #X1, Y1, X2, Y2, score = map(int, bbox)
        x1 = X1     
        y1 = Y1
        x2 = X1 + w
        y2 = Y1 + h
        h_origin, w_origin = img.shape[:2]
        #中心点
        center_x = (x2 + x1) / 2
        center_y = (y2 + y1) / 2
        h= y2-y1
        w= x2-x1
        length = max(h, w)
        square_x1 = int(center_x - length / 2)
        square_y1 = int(center_y - length / 2)
        square_x2 = int(center_x + length / 2)
        square_y2 = int(center_y + length / 2)
        #防止越界
        top_pad = max(0, -square_y1)
        left_pad = max(0, -square_x1)
        bottom_pad = max(0, square_y2 - h_origin)
        right_pad = max(0, square_x2 - w_origin) 
        #更新正方形顶点           
        square_x1 = max(0, square_x1)
        square_y1 = max(0, square_y1)
        square_x2 = min(w_origin, square_x2)
        square_y2 = min(h_origin, square_y2)

        cropped_img = img[int(square_y1):int(square_y2), int(square_x1):int(square_x2)]
        
        # 在必要的地方添加填充
        result_img = cv2.copyMakeBorder(
            cropped_img,
            top_pad, bottom_pad, left_pad, right_pad,
            borderType=cv2.BORDER_CONSTANT,
            value=(0, 0, 0)  # 黑色填充
        )
        
        return result_img