
# use Linear Probe to evaluate the quality of representations learned by SSL and noisy SL on 'cifar10' and 'cifar100'
# for SSL, we use all the data to train; for SL, we use 90% noisy data to train and 10% clean data to monitor
# this is good for SL
# we use the 100% clean data for re-training. We can change re-training epochs.

from __future__ import print_function

import os
import sys
import argparse
import time
import math

import torch
import torch.backends.cudnn as cudnn
from torchvision import transforms, datasets

import tensorboard_logger as tb_logger
from main_ce import set_loader
from util import AverageMeter, Recording, save_model
from util import adjust_learning_rate, warmup_learning_rate, accuracy
from util import set_optimizer
from networks.resnet_big import SupConResNet, LinearClassifier

try:
    import apex
    from apex import amp, optimizers
except ImportError:
    pass

from tools import *


def parse_option():
    parser = argparse.ArgumentParser('argument for evaluate quality of representations (linear probe)')

    # dataset and saved model
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100'], help='dataset')
    parser.add_argument('--ckpt', type=str, default='./pretrained/ckpt_cifar10_resnet18.pth', help='path to ssl or sl pre-trained model')
    parser.add_argument('--sslorsl', type=str, default='ssl', choices=['ssl', 'sl'], help='which pretrained model to evaluate')

    # training
    parser.add_argument('--model', type=str, default='resnet18', choices=['resnet18', 'resnet34', 'resnet50', 'resnet101'])
    parser.add_argument('--epochs', type=int, default=100, help='number of training epochs')
    parser.add_argument('--device', default='4', help="gpu used in training")
    parser.add_argument('--batch_size', type=int, default=256, help='batch_size')
    parser.add_argument('--learning_rate', type=float, default=0.05, help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='60,75,90', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.2, help='decay rate for learning rate')

    # other setting
    parser.add_argument('--workers', default='4', help='if dont use dataparallel, stay this unchanged is ok')
    parser.add_argument('--optimizer_type', default='sgd', choices=['sgd', 'adam'])
    parser.add_argument('--num_workers', type=int, default=16, help='num of workers to use')
    parser.add_argument('--weight_decay', type=float, default=0, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--cosine', action='store_true', help='using cosine annealing')
    parser.add_argument('--warm', action='store_true', help='warm-up for large batch training')

    # for recording
    parser.add_argument('--noise_type', type=str, default='symmetric', choices=['pairflip', 'symmetric', 'instance'])
    parser.add_argument('--noise_rate', type=float, help='corruption rate, should be less than 1', default=0.4)

    opt = parser.parse_args()

    # set the path according to the environment
    opt.data_folder = './datasets/'
    opt.model_path = './save/LinearProbe/{}_models'.format(opt.dataset)  # 储存模型的位置
    opt.tb_path = './save/LinearProbe/{}_tensorboard'.format(opt.dataset)  # 储存训练的tb结果的位置
    opt.log_path = './save/LinearProbe/{}_logs'.format(opt.dataset)  # log记录结果
    if opt.sslorsl == 'ssl':
        opt.model_name = 'SimCLR_{}_{}'.format(opt.dataset, opt.model)
    else:
        opt.model_name = 'NoisyCE_{}_{}_{}_{}'.format(opt.noise_type, opt.noise_rate, opt.dataset, opt.model)

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

    # warm-up for large-batch training,
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
    names = [opt.model_name_ssl, opt.model_name_sl]
    for i in range(2):
        opt.tb_folder = os.path.join(opt.tb_path, names[i])
        if not os.path.isdir(opt.tb_folder):
            os.makedirs(opt.tb_folder)
        opt.save_folder = os.path.join(opt.model_path, names[i])
        if not os.path.isdir(opt.save_folder):
            os.makedirs(opt.save_folder)
        opt.log_folder = os.path.join(opt.log_path, names[i])
        if not os.path.isdir(opt.log_folder):
            os.makedirs(opt.log_folder)

    return opt


def set_loader(opt):
    # construct data loader
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
        train_dataset = datasets.CIFAR10(root=opt.data_folder, transform=train_transform, download=True)
        val_dataset = datasets.CIFAR10(root=opt.data_folder, train=False, transform=val_transform)
    elif opt.dataset == 'cifar100':
        train_dataset = datasets.CIFAR100(root=opt.data_folder, transform=train_transform, download=True)
        val_dataset = datasets.CIFAR100(root=opt.data_folder, train=False, transform=val_transform)
    else:
        raise ValueError("not supported dataset:{}".format(opt.dataset))

    train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size, shuffle=(train_sampler is None),
        num_workers=opt.num_workers, pin_memory=True, sampler=train_sampler)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=8, pin_memory=True)

    return train_loader, val_loader


def set_model(opt):
    '''linear probe for ssl pretrained and sl pretrained models'''
    model = SupConResNet(name=opt.model)
    criterion = torch.nn.CrossEntropyLoss()

    classifier = LinearClassifier(name=opt.model, num_classes=opt.n_cls)
    ckpt = torch.load(opt.ckpt, map_location='cpu')
    state_dict = ckpt['model']

    if torch.cuda.is_available():
        if len(opt.workers) > 1: # 意思是继续使用dataparallel 用多张卡来训练
            model.encoder = torch.nn.DataParallel(model.encoder, device_ids=opt.workers)
        else:  # 因为之前是在多张卡上面训练的，这里读取的时候需要做一点点改动
            new_state_dict = {}
            for k, v in state_dict.items():
                k = k.replace("module.", "")
                new_state_dict[k] = v
            state_dict = new_state_dict
        model = model.to(opt.device)
        classifier = classifier.to(opt.device)
        criterion = criterion.to(opt.device)
        cudnn.benchmark = True

        model.load_state_dict(state_dict)

    return model, classifier, criterion


def train(train_loader, model, classifier, criterion, optimizer, epoch, opt):
    """one epoch training"""
    model.eval()
    classifier.train()

    losses = AverageMeter()
    top1 = AverageMeter()

    for idx, (images, labels) in enumerate(train_loader):

        images = images.to(opt.device, non_blocking=True)
        labels = labels.to(opt.device, non_blocking=True)
        bsz = labels.shape[0]

        # warm-up learning rate
        warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

        # compute loss
        with torch.no_grad():
            features = model.encoder(images)
        output = classifier(features.detach())
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


def validate(val_loader, model, classifier, criterion, opt):
    """validation"""
    model.eval()
    classifier.eval()

    losses = AverageMeter()
    top1 = AverageMeter()

    with torch.no_grad():
        for idx, (images, labels) in enumerate(val_loader):
            images = images.float().to(opt.device)
            labels = labels.to(opt.device)
            bsz = labels.shape[0]

            # forward
            output = classifier(model.encoder(images))
            loss = criterion(output, labels)

            # update metric
            losses.update(loss.item(), bsz)
            acc1, acc5 = accuracy(output, labels, topk=(1, 5))  # acc5没有用到
            top1.update(acc1[0], bsz)

    return losses.avg, top1.avg



def main():
    best_acc= 0
    opt = parse_option()

    # start recording
    Recording(opt, start=True)

    # build data loader based on noisy dataset
    train_loader, val_loader = set_loader(opt)

    # build model and criterion
    model, classifier, criterion = set_model(opt)

    # build optimizer   linear probe, only the params for classifier need to be updated
    optimizer= set_optimizer(opt, classifier)

    # tensorboard
    logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)

    # training
    for epoch in range(1, opt.epochs + 1):
        adjust_learning_rate(opt, optimizer, epoch)

        # train for one epoch
        time1 = time.time()
        train_loss, train_acc = train(train_loader, model, classifier, criterion, optimizer, epoch, opt)
        time2 = time.time()
        print('({}) Train epoch {}, total time {:.2f}s, loss:{:.2f}, accuracy:{:.2f}, lr:{:.3f}'. \
              format(opt.sslorsl, epoch, time2 - time1, train_loss, train_acc, optimizer.param_groups['lr']))

        # tensorboard logger
        logger.log_value('train_loss', train_loss, epoch)
        logger.log_value('train_acc', train_acc, epoch)

        ####################### 这里的val_loader就是测试集
        val_loss, val_acc = validate(val_loader, model, classifier, criterion, opt)
        print('({}) Validation epoch {}, loss:{:.2f}, accuracy:{:.2f}'.format(opt.sslorsl, epoch, val_loss, val_acc))
        if val_acc > best_acc:
            best_acc = val_acc

        # tensorboard logger
        logger.log_value('val_loss', val_loss, epoch)
        logger.log_value('val_acc', val_acc, epoch)

    print('({}) best accuracy on noisy validation set: {:.2f}'.format(opt.sslorsl, best_acc))

    # end recording
    Recording(opt, start=False)


if __name__ == '__main__':
    main()