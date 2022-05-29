
# this is used for noisy supervised training
# use 90% noisy data to train and 10% clean data to monitor
# and save the model which achieves best performance on clean val data


from __future__ import print_function

import os
import sys
import shutil
import argparse
import time
import math
import numpy as np

import torch
import torch.backends.cudnn as cudnn
from torchvision import transforms, datasets
from torchvision.datasets import CIFAR10, CIFAR100
from torch.utils.tensorboard import SummaryWriter

import tensorboard_logger as tb_logger
from util import AverageMeter, Recording, save_model
from util import adjust_learning_rate, warmup_learning_rate, accuracy
from util import set_optimizer
from networks.resnet_big import SupCEResNet

from tools import *
try:
    import apex
    from apex import amp, optimizers
except ImportError:
    pass


def parse_option():
    parser = argparse.ArgumentParser('argument for noisy supervised pre-training')

    # noisy dataset
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100'], help='dataset')
    parser.add_argument('--noise_type', type=str, default='symmetric', choices=['pairflip', 'symmetric', 'instance'])
    parser.add_argument('--noise_rate', type=float, default=0.4, help='corruption rate, should be less than 1')
    parser.add_argument('--split_per', default=0.9, type=float, help='percent of data for training')
    parser.add_argument('--seed', type=int, default=123, help='the random seed used in splitting data and creating noise')
    parser.add_argument('--appendix', type=str, default='try1', help='appendix for recording')

    # training
    parser.add_argument('--model', type=str, default='resnet18', choices=['resnet18', 'resnet34', 'resnet50', 'resnet101'])
    parser.add_argument('--epochs', type=int, default=150, help='number of training epochs')
    parser.add_argument('--device', default='8', help="gpu used in training")
    parser.add_argument('--batch_size', type=int, default=256, help='batch_size')
    parser.add_argument('--learning_rate', type=float, default=0.05, help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='100,120', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.2, help='decay rate for learning rate')

    # other setting
    parser.add_argument('--workers', default='4', help='if dont use dataparallel, just stay this unchanged')
    parser.add_argument('--num_workers', type=int, default=16, help='num of workers to use')
    parser.add_argument('--optimizer_type', default='sgd', choices=['sgd', 'adam'])
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--cosine', action='store_true', help='using cosine annealing')
    parser.add_argument('--warm', action='store_true', help='warm-up for large batch training')

    opt = parser.parse_args()

    # set the path according to the environment
    opt.data_folder = './datasets/'
    opt.model_path = './save/noisy_CE_pretrain/{}_models'.format(opt.dataset)  # 储存模型的位置
    opt.tb_path = './save/noisy_CE_pretrain/{}_tensorboard'.format(opt.dataset)  # 储存训练的tb结果的位置
    opt.log_path = './save/noisy_CE_pretrain/{}_logs'.format(opt.dataset)  # log记录结果
    opt.model_name = '{}_{}_{}_{}_{}'.format(opt.noise_type, opt.noise_rate, opt.dataset, opt.model, opt.appendix) # 模型的名字

    gpus = opt.workers.split(',')
    opt.workers = list([])
    for it in gpus:
        opt.workers.append(int(it))
    server = "cuda:{}".format(int(opt.device))
    opt.device = torch.device(server if torch.cuda.is_available() else "cpu")

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    if opt.warm:
        opt.warmup_from = 0.01
        opt.warm_epochs = 10
        if opt.cosine:
            eta_min = opt.learning_rate * (opt.lr_decay_rate ** 3)
            opt.warmup_to = eta_min + (opt.learning_rate - eta_min) * (
                    1 + math.cos(math.pi * opt.warm_epochs / opt.epochs)) / 2
        else:
            opt.warmup_to = opt.learning_rate

    if opt.dataset == 'cifar10':
        opt.n_cls = 10
    elif opt.dataset == 'cifar100':
        opt.n_cls = 100
    else:
        raise ValueError('dataset not supported: {}'.format(opt.dataset))

    # 最后创建文件夹 别管
    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if os.path.isdir(opt.tb_folder):
        shutil.rmtree(opt.tb_folder)
        os.makedirs(opt.tb_folder)
    else:
        os.makedirs(opt.tb_folder)
    # if not os.path.isdir(opt.tb_folder):
    #     os.makedirs(opt.tb_folder)
    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if os.path.isdir(opt.save_folder):
        shutil.rmtree(opt.save_folder)
        os.makedirs(opt.save_folder)
    else:
        os.makedirs(opt.save_folder)
    # if not os.path.isdir(opt.save_folder):
    #     os.makedirs(opt.save_folder)
    opt.log_folder = os.path.join(opt.log_path, opt.model_name)
    if os.path.isdir(opt.log_folder):
        shutil.rmtree(opt.log_folder)
        os.makedirs(opt.log_folder)
    else:
        os.makedirs(opt.log_folder)
    # if not os.path.isdir(opt.log_folder):
    #     os.makedirs(opt.log_folder)

    return opt


def set_noisy_loader(opt):
    '''set dataloader with noisy dataset'''
    if opt.dataset == 'cifar10':
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
    elif opt.dataset == 'cifar100':
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
    else:
        raise ValueError('dataset not supported: {}'.format(opt.dataset))
    normalize = transforms.Normalize(mean=mean, std=std)

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    if opt.dataset == 'cifar10':
        train_val_dataset = CIFAR10(root = opt.data_folder, train=True, download=True)
    elif opt.dataset == 'cifar100':
        train_val_dataset = CIFAR100(root=opt.data_folder, train=True, download=True)

    # 从cifar10/100的trainset里面分出10%作为validation dataset, train和val分别用的自己的transform
    # 这里直接是把整个train_val_dataset拿来做noisify 实际上训练的时候，train用noisy的label，val用clean的label就可以了
    train_data, val_data, train_noisy_labels, val_noisy_labels, train_clean_labels, val_clean_labels = \
        dataset_split(np.array(train_val_dataset.data), np.array(train_val_dataset.targets), opt.noise_rate, opt.noise_type, opt.split_per, opt.seed, opt.n_cls)
    train_dataset = Train_Dataset(train_data, train_noisy_labels, train_clean_labels, train_transform)
    val_dataset = Train_Dataset(val_data, val_noisy_labels, val_clean_labels, val_transform)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers, pin_memory=False)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=256, num_workers=8, shuffle=False, pin_memory=False)

    return train_loader, val_loader


def set_model(opt):
    '''regular resnet'''
    model = SupCEResNet(name=opt.model, num_classes=opt.n_cls)
    criterion = torch.nn.CrossEntropyLoss()

    if torch.cuda.is_available():
        if len(opt.workers) > 1:
            model.encoder = torch.nn.DataParallel(model.encoder, device_ids=opt.workers)
        model = model.to(opt.device)
        criterion = criterion.to(opt.device)
        cudnn.benchmark = True

    return model, criterion


def train(train_loader, model, criterion, optimizer, epoch, opt):
    """one epoch training"""
    model.train()

    losses = AverageMeter()
    top1 = AverageMeter()

    for idx, (images, labels, clean_labels) in enumerate(train_loader):

        images = images.to(opt.device, non_blocking=True)
        labels = labels.to(opt.device, non_blocking=True)
        bsz = labels.shape[0]

        # warm-up learning rate
        warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

        # compute loss
        output = model(images)
        loss = criterion(output, labels)

        # update metric
        losses.update(loss.item(), bsz)
        acc1, acc5 = accuracy(output, labels, topk=(1, 5))
        top1.update(acc1[0], bsz)

        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return losses.avg, top1.avg


def validate(val_loader, model, criterion, opt):
    """validation"""
    model.eval()

    losses = AverageMeter()
    top1 = AverageMeter()

    with torch.no_grad():
        for idx, (images, _, clean_labels) in enumerate(val_loader):  # use clean label for validation
            images = images.float().to(opt.device)
            clean_labels = clean_labels.to(opt.device)
            bsz = clean_labels.shape[0]

            # forward
            output = model(images)
            loss = criterion(output, clean_labels)

            # update metric
            losses.update(loss.item(), bsz)
            acc1, acc5 = accuracy(output, clean_labels, topk=(1, 5))  # acc5没有用到
            top1.update(acc1[0], bsz)

    return losses.avg, top1.avg


def main():
    best_acc = 0; best_epoch = 0
    opt = parse_option()
    save_best_file = os.path.join(opt.save_folder, 'best.pth')

    # start recording
    Recording(opt, start=True)

    # build data loader based on noisy dataset
    train_loader, val_loader = set_noisy_loader(opt)

    # build model, criterion and optimizer
    model, criterion= set_model(opt)

    # set optimizer
    optimizer = set_optimizer(opt, model)

    # tensorboard
    train_writer = SummaryWriter(log_dir=os.path.join(opt.tb_folder, 'train'))
    valid_writer = SummaryWriter(log_dir=os.path.join(opt.tb_folder, 'valid'))

    # training routine
    for epoch in range(1, opt.epochs + 1):
        lr = adjust_learning_rate(opt, optimizer, epoch)

        # train for one epoch
        train_loss, train_acc = train(train_loader, model, criterion, optimizer, epoch, opt)
        print('Train epoch {}, loss:{:.2f}, accuracy:{:.2f}, lr:{:3f}'.format(epoch, train_loss,  train_acc, lr))

        # tensorboard logger
        train_writer.add_scalar('loss', train_loss, epoch)
        train_writer.add_scalar('acc', train_acc, epoch)

        # if epoch % opt.save_freq == 0:
        #     save_file = os.path.join(opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
        #     save_model(model, optimizer, opt, epoch, save_file)

        # eval for one epoch
        val_loss, val_acc = validate(val_loader, model, criterion, opt)
        print('Validation epoch {}, loss:{:.2f}, accuracy:{:.2f}'.format(epoch, val_loss, val_acc))
        if val_acc > best_acc and epoch<50: # 把保存的模型的epoch限制在前50个epoch里面
            best_acc = val_acc; best_epoch=epoch # save the model which achieves best acc on clean val dataset
            save_model(model, optimizer, opt, epoch, save_best_file)

        # tensorboard logger
        valid_writer.add_scalar('loss', val_loss, epoch)
        valid_writer.add_scalar('acc', val_acc, epoch)

    print('best (early training stage) accuracy on noisy validation set: {:.2f} happens at epoch {}'.format(best_acc, best_epoch))

    # end recording
    Recording(opt, start=False)



if __name__ == '__main__':
    main()