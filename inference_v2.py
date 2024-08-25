import argparse
import os
import parser
import random
from time import perf_counter
import torch
from torch import nn, optim
import torchvision
from torchvision import datasets, transforms
from Dataset.transform import UnNormalize
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
import json
from tqdm import tqdm

def inference_and_save(inference_transform, model, result_json_path, data_path, bbox_txt_path=None, label_json_path=None, dataset='raf'):
    imgs = []
    os.makedirs(os.path.dirname(result_json_path), exist_ok=True)
    model.eval()
    
    if dataset == 'raf':
        assert data_path is not None and bbox_txt_path is not None
        with open(bbox_txt_path, 'r') as f:
            data_infos = f.readlines()[2:]
        f.close()
        with open(result_json_path, 'w') as f:
            f.write('[')
            for i in tqdm(range(len(data_infos))):
                data_info = data_infos[i]
                img_name = data_info.split(' ')[0]
                img_path = os.path.join(data_path, img_name)
                assert os.path.exists(img_path), 'img not exist'
                x, y, w, h = map(float, data_info.split(' ')[1:])
                bbox = np.array([x, y, w, h])
                img = cv2.cvtColor(cv2.imread(img_path),cv2.COLOR_BGR2RGB)      
                img = get_crop_img_from_bbox(bbox, img)
                img = Image.fromarray(img)
                img = inference_transform(img)
                img = img.unsqueeze(0).cuda()
                if i < 10:
                    ori_img = UnNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)
                    torchvision.utils.save_image(ori_img[0],'images/inference_raf_image_{:}.jpg'.format(i),normalize=True)   
                
                # model.eval() ## 放前面了
                with torch.no_grad():
                    output = model(img)
                    au_pred = output.squeeze(0).cpu().numpy().tolist()
                    au_pred = [round(i, 4) for i in au_pred]

                result_dict ={
                    "file_name":img_name,
                    "au": au_pred
                }
                
                if i != len(data_infos) - 1 :
                    f.write(json.dumps(result_dict) + ', \n')
                else:
                    f.write(json.dumps(result_dict) + ']')

    elif dataset == 'affect':
        assert data_path is not None and label_json_path is not None
        with open(label_json_path, 'r') as f:
            data_infos = json.load(f)
        with open(result_json_path, 'w') as f:
            f.write('[')
            for i in tqdm(range(len(data_infos))):
                data_info = data_infos[i]
                img_name = data_info['file_name']
                emotion = data_info['emo']
                img_path = os.path.join(data_path, img_name)
                assert os.path.exists(img_path), 'img not exist'
                
                img = Image.open(img_path).convert('RGB')
                img = inference_transform(img)
                img = img.unsqueeze(0).cuda()
                if i < 10:
                    ori_img = UnNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)
                    torchvision.utils.save_image(ori_img[0],'images/inference_image_{:}.jpg'.format(i),normalize=True)   
                
                # model.eval()  ## 放前面了
                with torch.no_grad():
                    output = model(img)
                    au_pred = output.squeeze(0).cpu().numpy().tolist()
                    au_pred = [round(i, 4) for i in au_pred]

                result_dict ={
                    "file_name":img_name,
                    "emo":emotion,
                    "au": au_pred
                }
                
                if i != len(data_infos) - 1 :
                    f.write(json.dumps(result_dict) + ', \n')
                else:
                    f.write(json.dumps(result_dict) + ']')



def main(args):
    torch.manual_seed(args.random_seed)
    random.seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)  # gpu
    np.random.seed(args.random_seed)  # numpy

    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
    logger = create_logger(args.log_dir, model_name="resnet50", phase='test')
    
    #model 
    model = torchvision.models.resnet50()
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_features, 24),
        nn.Hardsigmoid()  # 使用 Sigmoid 激活函数将输出限制在0-1之间
        )
    model = model.cuda()
    model.load_state_dict(torch.load(args.model_path))
    model.eval()
    
    inference_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    inference_and_save(inference_transform, model, args.result_json_path, args.data_path, bbox_txt_path=args.bbox_txt_path, 
                       label_json_path=args.label_json_path, dataset=args.dataset)
        

    
    # json 保存， path: xxxx, AU:xxxx
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--model_path',type=str,default="checkpoint/0813_res50_mse_e100_newdata/best.pth")
    
    parser.add_argument('--data_path',type=str,default="/home/dingyan/huguohong/9_10/dataset/Expression/AffectNet-kaggle")
    parser.add_argument('--bbox_txt_path',type=str,default="raf_bbox.txt")
    parser.add_argument('--dataset',type=str,default="affect")
    parser.add_argument('--result_json_path',type=str,default="result/Affectnet_kaggle_train-sample-affectnet.json")
    parser.add_argument('--label_json_path',type=str,default="affect_label/Affectnet_kaggle_train-sample-affectnet.json")
    
    
    parser.add_argument('--log_dir',type=str,default="log/log_inference_affect")

    #parser.add_argument("--iscrop", type=str, default="_crop")
    args, _ = parser.parse_known_args()
    main(args)
    

