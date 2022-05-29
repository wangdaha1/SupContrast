from __future__ import print_function

import math
import numpy as np
import torch
import torch.optim as optim
import sys
import os
from datetime import datetime

from torch import nn
from torch.autograd import Variable
from torch.nn.parameter import Parameter


class TwoCropTransform:
    """Create two crops of the same image"""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]


class Logger(object):
    """
    save the output of the console into a log file
    """
    def __init__(self, filename='default.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


def Recording(opt, start=True):
    if start==True:
        # save parameters in argument.txt file
        params = os.path.join(os.path.expanduser(opt.log_folder), 'parameters.txt')
        with open(params, 'w') as f:
            for key, value in vars(opt).items():
                f.write('%s:%s\n' % (key, value))
                print(key, value)
        # save outputs during training
        sys.stdout = Logger(os.path.join(opt.log_folder, 'log.txt'), sys.stdout)
        start_time = datetime.now()
        start_time_str = datetime.strftime(start_time, '%m-%d_%H-%M')
        print("Training starts at " + start_time_str + " !!!!")
    elif start==False:
        end_time = datetime.now()
        end_time_str = datetime.strftime(end_time, '%m-%d_%H-%M')
        print("Training is Finished at " + end_time_str + " !!!!")
        f = open(os.path.join(opt.log_folder, 'log.txt'), 'a')
        sys.stdout = f
        sys.stderr = f


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
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def adjust_learning_rate(args, optimizer, epoch):
    lr = args.learning_rate
    if args.cosine:
        eta_min = lr * (args.lr_decay_rate ** 3)
        lr = eta_min + (lr - eta_min) * (
                1 + math.cos(math.pi * epoch / args.epochs)) / 2
    else:
        steps = np.sum(epoch > np.asarray(args.lr_decay_epochs))
        if steps > 0:
            lr = lr * (args.lr_decay_rate ** steps)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr


def warmup_learning_rate(args, epoch, batch_id, total_batches, optimizer):
    '''这是什么操作啊'''
    if args.warm and epoch <= args.warm_epochs:
        p = (batch_id + (epoch - 1) * total_batches) / (args.warm_epochs * total_batches)
        lr = args.warmup_from + p * (args.warmup_to - args.warmup_from)

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


def set_optimizer(opt, model):
    if opt.optimizer_type=='sgd':
        optimizer = optim.SGD(model.parameters(),
                              lr=opt.learning_rate,
                              momentum=opt.momentum,
                              weight_decay=opt.weight_decay)
    elif opt.optimizer_type=='adam':
        optimizer = optim.Adam(model.parameters(),
                               lr=opt.learning_rate,
                               weight_decay=opt.weight_decay)
    # elif opt.optimizer_type=='lars':
    #     optimizer_0 = optim.SGD(model.parameters(),
    #                           lr=opt.learning_rate,
    #                           momentum=opt.momentum,
    #                           weight_decay=opt.weight_decay)
    #     optimizer = LARS(optimizer_0)
    else:
        raise ValueError("optimizer type not supported: {}".format(opt.optimizer_type))

    return optimizer


def save_model(model, optimizer, opt, epoch, save_file):
    print('==> Saving model...')
    state = {
        'opt': opt,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
    }
    torch.save(state, save_file)
    del state


class LARS(object):
    """
    Slight modification of LARC optimizer from https://github.com/NVIDIA/apex/blob/d74fda260c403f775817470d87f810f816f3d615/apex/parallel/LARC.py
    Matches one from SimCLR implementation https://github.com/google-research/simclr/blob/master/lars_optimizer.py
    Args:
        optimizer: Pytorch optimizer to wrap and modify learning rate for.
        trust_coefficient: Trust coefficient for calculating the adaptive lr. See https://arxiv.org/abs/1708.03888
    """

    def __init__(self,
                 optimizer,
                 trust_coefficient=0.001,
                 ):
        self.param_groups = optimizer.param_groups
        self.optim = optimizer
        self.trust_coefficient = trust_coefficient

    def __getstate__(self):
        return self.optim.__getstate__()

    def __setstate__(self, state):
        self.optim.__setstate__(state)

    def __repr__(self):
        return self.optim.__repr__()

    def state_dict(self):
        return self.optim.state_dict()

    def load_state_dict(self, state_dict):
        self.optim.load_state_dict(state_dict)

    def zero_grad(self):
        self.optim.zero_grad()

    def add_param_group(self, param_group):
        self.optim.add_param_group(param_group)

    def step(self):
        with torch.no_grad():
            weight_decays = []
            for group in self.optim.param_groups:
                # absorb weight decay control from optimizer
                weight_decay = group['weight_decay'] if 'weight_decay' in group else 0
                weight_decays.append(weight_decay)
                group['weight_decay'] = 0
                for p in group['params']:
                    if p.grad is None:
                        continue

                    if weight_decay != 0:
                        p.grad.data += weight_decay * p.data

                    param_norm = torch.norm(p.data)
                    grad_norm = torch.norm(p.grad.data)
                    adaptive_lr = 1.

                    if param_norm != 0 and grad_norm != 0 and group['layer_adaptation']:
                        adaptive_lr = self.trust_coefficient * param_norm / grad_norm

                    p.grad.data *= adaptive_lr

        self.optim.step()
        # return weight decay control to optimizer
        for i, group in enumerate(self.optim.param_groups):
            group['weight_decay'] = weight_decays[i]

