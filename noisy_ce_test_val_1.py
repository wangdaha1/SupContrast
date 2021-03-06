
# use noisy data to train from scratch and use clean data to validate

# draw loss/margin distributions of clean and noisy samples along with training epochs

from __future__ import print_function

import os
import sys
import argparse
import shutil
import time
import math
import numpy as np
import matplotlib.pyplot as plt

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
    parser = argparse.ArgumentParser('argument for noisy supervised training')

    # noisy dataset
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100'], help='dataset')
    parser.add_argument('--noise_type', type=str, default='symmetric', choices=['pairflip', 'symmetric', 'instance'])
    parser.add_argument('--noise_rate', type=float, default=0.4, help='corruption rate, should be less than 1')
    parser.add_argument('--split_per', default=1, type=float, help='percent of data for training')
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
    opt.model_path = './save/noisy_CE_1/{}_models'.format(opt.dataset)  # ?????????????????????
    opt.tb_path = './save/noisy_CE_1/{}_tensorboard'.format(opt.dataset)  # ???????????????tb???????????????
    opt.log_path = './save/noisy_CE_1/{}_logs'.format(opt.dataset)  # log????????????
    opt.figure_path = './save/noisy_CE_1/{}_figures'.format(opt.dataset)
    opt.model_name = '{}_{}_{}_{}_{}'.format(opt.noise_type, opt.noise_rate, opt.dataset, opt.model, opt.appendix) # ???????????????

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

    # ????????????????????? ??????
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
    opt.figure_folder = os.path.join(opt.figure_path, opt.model_name)
    if os.path.isdir(opt.figure_folder):
        shutil.rmtree(opt.figure_folder)
        os.makedirs(opt.figure_folder)
    else:
        os.makedirs(opt.figure_folder)

    loss_dir = os.path.join(opt.figure_folder, 'loss')
    if os.path.isdir(loss_dir):
        shutil.rmtree(loss_dir)
        os.makedirs(loss_dir)
    else:
        os.makedirs(loss_dir)
    margin_dir = os.path.join(opt.figure_folder, 'margin')
    if os.path.isdir(margin_dir):
        shutil.rmtree(margin_dir)
        os.makedirs(margin_dir)
    else:
        os.makedirs(margin_dir)

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

    if opt.dataset == 'cifar10':  # use 100% noisy data to train and clean test data for validation
        train_dataset = CIFAR10(root=opt.data_folder, train=True, download=True)
        val_dataset = CIFAR10(root=opt.data_folder, train=False, download=True, transform=val_transform)
    elif opt.dataset == 'cifar100':
        train_dataset = CIFAR100(root=opt.data_folder, train=True, download=True)
        val_dataset = CIFAR100(root=opt.data_folder, train=False, download=True, transform=val_transform)

    train_data, _, train_noisy_labels, _, train_clean_labels, _ = \
        dataset_split(np.array(train_dataset.data), np.array(train_dataset.targets), opt.noise_rate,
                      opt.noise_type, 1, opt.seed, opt.n_cls)
    train_dataset = Train_Dataset(train_data, train_noisy_labels, train_clean_labels, train_transform)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=opt.batch_size, shuffle=True,
                                               num_workers=opt.num_workers, pin_memory=False)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=256, num_workers=8, shuffle=False,
                                             pin_memory=False)

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


def del_tensor_ele(arr, index):
    arr1 = arr[0:index]
    arr2 = arr[index+1:]
    return torch.cat((arr1, arr2),dim=0)


def train(train_loader, model, criterion, optimizer, epoch, opt):
    """one epoch training"""
    model.train()

    losses = AverageMeter()
    top1 = AverageMeter()
    clean_losses, noisy_losses, clean_margins, noisy_margins = [], [], [], []

    for idx, (images, labels, clean_labels) in enumerate(train_loader):

        images = images.to(opt.device, non_blocking=True)
        labels = labels.to(opt.device, non_blocking=True)
        clean_labels = clean_labels.to(opt.device, non_blocking=True)
        bsz = labels.shape[0]

        # warm-up learning rate
        warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

        # compute loss
        output = model(images)
        loss = criterion(output, labels)

        # losses and margins of clean and noisy samples
        loss_per_sample = torch.nn.CrossEntropyLoss(reduction='none')(output, labels) # ????????????sample???loss
        output_softmax = torch.nn.Softmax(dim=1)(output)

        margins =  [] # ??????????????? ??????batch????????????margins list
        for sample in range(bsz): # ??????margin
            candidate = del_tensor_ele(output_softmax[sample], labels[sample].item())
            max_except_label = torch.max(candidate).item()
            margins.append(output_softmax[sample][labels[sample].item()].item() - max_except_label)

        for sample in range(bsz):
            if labels[sample] == clean_labels[sample]:
                clean_losses.append(loss_per_sample[sample].item())
                clean_margins.append(margins[sample])
            else:
                noisy_losses.append(loss_per_sample[sample].item())
                noisy_margins.append(margins[sample])

        # update metric
        losses.update(loss.item(), bsz)
        acc1, acc5 = accuracy(output, labels, topk=(1, 5))
        top1.update(acc1[0], bsz)

        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return losses.avg, top1.avg, clean_losses, noisy_losses, clean_margins, noisy_margins


def validate(val_loader, model, criterion, opt):
    """validation"""
    model.eval()

    losses = AverageMeter()
    top1 = AverageMeter()

    with torch.no_grad():
        for idx, (images, clean_labels) in enumerate(val_loader):  # ???????????????clean label???????????????
            images = images.float().to(opt.device)
            clean_labels = clean_labels.to(opt.device)
            bsz = clean_labels.shape[0]

            # forward
            output = model(images)
            loss = criterion(output, clean_labels)

            # update metric
            losses.update(loss.item(), bsz)
            acc1, acc5 = accuracy(output, clean_labels, topk=(1, 5))  # acc5????????????
            top1.update(acc1[0], bsz)

    return losses.avg, top1.avg


def draw_graph(type, clean_samples, noisy_samples, epoch, opt):
    # draw distributions and save as .png
    save_dir = os.path.join(opt.figure_folder, '{}'.format(type))
    plt.figure()
    if type=='margin':
        bins = np.linspace(-1, 1, 100)
        plt.ylim((0, 2000))
    else:
        bins = np.linspace(0, 5, 100)
        plt.ylim((0, 2000))
    plt.hist(clean_samples, bins, alpha=0.5, label='clean')
    plt.hist(noisy_samples, bins, alpha=0.5, label='noisy')
    plt.legend(loc='upper left')
    plt.show()
    plt.savefig(os.path.join(save_dir, 'epoch_{}'.format(epoch)+'.png'))


def main():
    opt = parse_option()

    # start recording
    Recording(opt, start=True)

    # build data loader based on noisy dataset
    train_loader, val_loader = set_noisy_loader(opt)

    # build model, criterion and optimizer
    model, criterion = set_model(opt)

    # set optimizer
    optimizer = set_optimizer(opt, model)

    # tensorboard
    train_writer = SummaryWriter(log_dir=os.path.join(opt.tb_folder, 'train'))
    valid_writer = SummaryWriter(log_dir=os.path.join(opt.tb_folder, 'valid'))

    # training routine
    for epoch in range(1, opt.epochs + 1):
        lr = adjust_learning_rate(opt, optimizer, epoch)

        # train for one epoch
        train_loss, train_acc, clean_losses, noisy_losses, clean_margins, noisy_margins = \
            train(train_loader, model, criterion, optimizer, epoch, opt)
        print('Train epoch {}, loss:{:.2f}, accuracy:{:.2f}, lr:{:3f}'.format(epoch, train_loss,  train_acc, lr))

        draw_graph('loss', clean_losses, noisy_losses, epoch, opt)
        draw_graph('margin', clean_margins, noisy_margins, epoch, opt)

        # tensorboard logger
        train_writer.add_scalar('loss', train_loss, epoch)
        train_writer.add_scalar('acc', train_acc, epoch)

        # eval for one epoch on test data
        val_loss, val_acc = validate(val_loader, model, criterion, opt)
        print('Val epoch {}, loss:{:.2f}, accuracy:{:.2f}'.format(epoch, val_loss, val_acc))

        # tensorboard logger
        valid_writer.add_scalar('loss', val_loss, epoch)
        valid_writer.add_scalar('acc', val_acc, epoch)

    # end recording
    Recording(opt, start=False)



if __name__ == '__main__':
    main()