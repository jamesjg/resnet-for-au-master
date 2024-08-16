from math import cos, pi
import torch
import os
import shutil
import numpy as np
import cv2

def adjust_learning_rate(optimizer, config,epoch, iteration, num_iter):
    lr = optimizer.param_groups[0]['lr']

    warmup_epoch = 5 if config.warmup else 0
    warmup_iter = warmup_epoch * num_iter
    current_iter = iteration + epoch * num_iter
    max_iter = config.epochs * num_iter

    if config.lr_decay == 'step':
        lr = config.lr * (config.gamma ** ((current_iter - warmup_iter) / (max_iter - warmup_iter)))
    elif config.lr_decay == 'cos':
        lr = config.lr * (1 + cos(pi * (current_iter - warmup_iter) / (max_iter - warmup_iter))) / 2
    elif config.lr_decay == 'linear':
        lr = config.lr * (1 - (current_iter - warmup_iter) / (max_iter - warmup_iter))
    elif config.lr_decay == 'schedule':
        count = sum([1 for s in config.schedule if s <= epoch])
        lr = config.lr * pow(config.gamma, count)
    else:
        raise ValueError('Unknown lr mode {}'.format(config.lr_decay))

    if epoch < warmup_epoch:
        lr = config.lr * current_iter / warmup_iter


    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
def load_filtered_state_dict(model, snapshot):
    # By user apaszke from discuss.pytorch.org
    model_dict = model.state_dict()
    snapshot = {k: v for k, v in snapshot.items() if k in model_dict}
    model_dict.update(snapshot)
    model.load_state_dict(model_dict)
def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))
        
        
    
def get_crop_img_from_bbox(bbox, img):
    """
    :param bbox: [x1, y1, w, h], 左上角坐标和宽高
    """
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