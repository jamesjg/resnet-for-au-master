import os
import logging
import time
from pathlib import Path

import torch
import torch.optim as optim
from collections import OrderedDict


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0


def create_logger(cfg, model_name, phase='train'):
    root_output_dir = Path(cfg.root_dir)
    # set up logger
    if not root_output_dir.exists():
        print('=> creating {}'.format(root_output_dir))
        root_output_dir.mkdir()

    dataset = cfg.dataset + '_'+str(cfg.num_au)
    dataset = dataset.replace(':', '_')

    final_output_dir = root_output_dir / dataset / model_name

    print('=> creating {}'.format(final_output_dir))
    final_output_dir.mkdir(parents=True, exist_ok=True)

    time_str = time.strftime('%Y-%m-%d-%H-%M')
    log_file = '{}_{}_{}.log'.format(model_name, time_str, phase)
    final_log_file = final_output_dir / log_file
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=str(final_log_file),
                        format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    tensorboard_log_dir = Path(cfg.log_dir) / dataset / \
        (model_name + '-' + time_str)
    print('=> creating {}'.format(tensorboard_log_dir))
    tensorboard_log_dir.mkdir(parents=True, exist_ok=True)

    return logger, str(final_output_dir), str(tensorboard_log_dir)


def get_optimizer(cfg, model,finetune=False):
    optimizer = None
    # if model.name=='c':
    #     base=model.resnet.parameters()
    #     new = model.global_net.parameters() and model.refine_net.parameters() and model.posenet.parameters()
    #
    #     if cfg.optimizer == 'sgd':
    #         optimizer = optim.SGD(
    #             [{'params': base, 'lr':0.000001},
    #             {'params': new }],
    #             lr=cfg.lr,
    #             momentum=cfg.momentum,
    #             weight_decay=cfg.weight_decay,
    #             nesterov=cfg.nesterov
    #         )
    #     elif cfg.optimizer == 'adam':
    #         optimizer = optim.Adam(
    #             [{'params': base, 'lr':0.000001},
    #             {'params': new}],
    #             lr=cfg.lr,
    #             weight_decay = cfg.weight_decay,
    #             amsgrad=True
    #         )
    if not finetune:
        if cfg.optimizer == 'sgd':
            optimizer = optim.SGD(
                model.parameters(),
                lr=cfg.lr,
                momentum=cfg.momentum,
                weight_decay=cfg.weight_decay,
                nesterov=cfg.nesterov
            )
        elif cfg.optimizer == 'adam':
            optimizer = optim.Adam(
                model.parameters(),
                lr=cfg.lr,
                weight_decay = cfg.weight_decay,
                amsgrad=cfg.amsgrad
            )
        elif cfg.optimizer == 'adamw':
            optimizer = optim.AdamW(
                model.parameters(),
                lr=cfg.lr,
                weight_decay = cfg.weight_decay,
                amsgrad=cfg.amsgrad
            )
    else:
        base= list(map(id, model.backbone.parameters()))
        new = filter(lambda p: id(p) not in base, model.parameters())
        if cfg.optimizer == 'sgd':
            optimizer = optim.SGD(
                [
                    {'params': model.backbone.parameters(), 'lr': cfg.lr*0.1},
                    {'params': new}
                 ],
                lr=cfg.lr,
                momentum=cfg.momentum,
                weight_decay=cfg.weight_decay,
                nesterov=cfg.nesterov
            )
        elif cfg.optimizer == 'adam':
            optimizer = optim.Adam(
                [{'params': model.backbone.parameters(), 'lr': cfg.lr*0.1},
                 {'params': new}],
                 lr=cfg.lr,
                weight_decay = cfg.weight_decay,
                amsgrad=cfg.amsgrad
            )
        elif cfg.optimizer == 'adamw':
            optimizer = optim.AdamW(
                [{'params': model.backbone.parameters(), 'lr': cfg.lr*0.1},
                 {'params': new}],
                 lr=cfg.lr,
                weight_decay = cfg.weight_decay,
                amsgrad=cfg.amsgrad
            )



    return optimizer


def save_checkpoint(states, is_best, output_dir,
                    filename='checkpoint.pth.tar'):
    torch.save(states, os.path.join(output_dir, filename))
    if is_best and 'state_dict' in states:
        print('save best')
        if( os.path.exists(os.path.join(output_dir, 'model_best.pth'))):
            os.remove(os.path.join(output_dir, 'model_best.pth'))
        torch.save(states['state_dict'],
                   os.path.join(output_dir, 'model_best.pth'))

def remove_module_dict(state_dict, is_print=False):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k[:7] == 'module.':
            name = k[7:] # remove `module.`
        else:
            name = k
        new_state_dict[name] = v
    if is_print: print(new_state_dict.keys())
    return new_state_dict