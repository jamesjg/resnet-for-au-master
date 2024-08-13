from math import cos, pi
import torch
import os
import shutil

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

def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))