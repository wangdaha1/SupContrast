
# this is used for fine tune by noisy data and validate on clean test data

from __future__ import print_function

import os
import shutil
import sys
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
from networks.resnet_big import SupConResNet, LinearClassifier, FineTune, SupCEResNet

from tools import *

try:
    import apex
    from apex import amp, optimizers
except ImportError:
    pass

def parse_option():
    parser = argparse.ArgumentParser('argument for fine tune on noisy dataset')

    # noisy dataset and saved model for fine tune
    # please note that if you choose NoisySL pretrained model, you should keep other params same as the model folder
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100'], help='dataset')
    parser.add_argument('--noise_type', type=str, choices=['pairflip', 'symmetric', 'instance'], default='symmetric')
    parser.add_argument('--noise_rate', type=float, help='corruption rate, should be less than 1', default=0.4)
    parser.add_argument('--split_per', default=1, type=float, help='percent of data for training')
    parser.add_argument('--seed', type=int, default=123, help='the random seed used in torch related')
    parser.add_argument('--ckpt', type=str, default='./pretrained/ckpt_cifar10_resnet18.pth', help='path to pre-trained model')
    parser.add_argument('--model', type=str, default='resnet18', choices=['resnet18', 'resnet34', 'resnet50', 'resnet101'])
    parser.add_argument('--sslorsl', type=str, default='ssl', choices=['ssl', 'sl'], help='separate recordings of SSL and SL')
    parser.add_argument('--appendix', type=str, default='try1', help='appendix for recording')

    # training
    parser.add_argument('--epochs', type=int, default=150, help='number of training epochs')
    parser.add_argument('--device', default='8', help="gpu used in training")
    parser.add_argument('--batch_size', type=int, default=256, help='batch_size')
    parser.add_argument('--learning_rate', type=float, default=0.05, help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='100,120', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.2, help='decay rate for learning rate')

    # other setting
    parser.add_argument('--num_workers', type=int, default=8, help='num of workers to use')
    parser.add_argument('--optimizer_type', default='sgd', choices=['sgd', 'adam'])
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--cosine', action='store_true', help='using cosine annealing')
    parser.add_argument('--warm', action='store_true', help='warm-up for large batch training')

    opt = parser.parse_args()

    # set the path according to the environment
    opt.data_folder = './datasets/'
    opt.model_path = './save/noisy_FT_1/{}/{}_models'.format(opt.sslorsl, opt.dataset)  # ?????????????????????
    opt.tb_path = './save/noisy_FT_1/{}/{}_tensorboard'.format(opt.sslorsl, opt.dataset)  # ???????????????tb???????????????
    opt.log_path = './save/noisy_FT_1/{}/{}_logs'.format(opt.sslorsl, opt.dataset)  # log????????????
    opt.model_name = '{}_{}_{}_{}_{}'.format(opt.noise_type, opt.noise_rate, opt.dataset, opt.model, opt.appendix)  # ???????????????

    # gpus = opt.workers.split(',')
    # opt.workers = list([])
    # for it in gpus:
    #     opt.workers.append(int(it))
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
    if opt.sslorsl == 'ssl':
        model = SupConResNet(name=opt.model)
    else:
        model = SupCEResNet(name=opt.model, num_classes=opt.n_cls)
    criterion = torch.nn.CrossEntropyLoss()
    classifier = LinearClassifier(name=opt.model, num_classes=opt.n_cls)

    ckpt = torch.load(opt.ckpt, map_location='cpu')
    state_dict = ckpt['model']

    if torch.cuda.is_available():
        if opt.sslorsl=='ssl':  # ??????????????????????????????????????????????????????????????????????????????????????????
            new_state_dict = {}
            for k, v in state_dict.items():
                k = k.replace("module.", "")
                new_state_dict[k] = v
            state_dict = new_state_dict
        model =  model.to(opt.device)
        classifier = classifier.to(opt.device)
        criterion = criterion.to(opt.device)
        cudnn.benchmark = True

        model.load_state_dict(state_dict)

    return model, classifier, criterion


def train(train_loader, model, criterion, optimizer, epoch, opt):
    """one epoch training for fine tuning"""
    model.train() # ?????????model???????????????encoder???classifier?????????model

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

        # optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return losses.avg, top1.avg  # ??????????????????????????????


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


# ????????????label precision???noise detect???validation
# def validate_cal(val_loader, model, criterion, opt):
#     """validation"""
#     model.eval()
#
#     # losses = AverageMeter()
#     top1 = AverageMeter()
#     # ABCD: A:target=pred, target=clean; B:target=pred, target=noisy;
#     # C: target!=pred, target=clean; D: target!=pred, target=noisy
#     A = AverageMeter()
#     B = AverageMeter()
#     C = AverageMeter()
#     D = AverageMeter()
#
#     with torch.no_grad():
#         for idx, (images, labels, clean_labels) in enumerate(val_loader):
#             images = images.float().to(opt.device)
#             labels = labels.to(opt.device)
#             clean_labels = clean_labels.to(opt.device)
#             bsz = clean_labels.shape[0]
#
#             # forward
#             output = model(images)
#             # loss = criterion(output, clean_labels)
#             _, pred = torch.max(output.data, 1) # ?????????label
#
#             # update metric
#             # losses.update(loss.item(), bsz)
#             acc1, acc5 = accuracy(output, clean_labels, topk=(1, 5))
#             top1.update(acc1[0], bsz)
#             A.update(int(((pred == labels) & (labels == clean_labels)).sum()))
#             B.update(int(((pred == labels) & (labels != clean_labels)).sum()))
#             C.update(int(((pred != labels) & (labels == clean_labels)).sum()))
#             D.update(int(((pred != labels) & (labels != clean_labels)).sum()))
#
#     label_precision_rate = A.sum/(A.sum+B.sum)
#     clean_selection_num = A.sum
#     noise_detection_rate = D.sum/(C.sum+D.sum)
#     noise_detect_num = D.sum
#
#     return top1.avg, label_precision_rate, clean_selection_num, noise_detection_rate, noise_detect_num


def main():
    opt = parse_option()

    # start recording
    Recording(opt, start=True)

    # build data loader based on noisy dataset
    train_loader, val_loader = set_noisy_loader(opt)

    # build model and criterion
    model_temp, classifier, criterion = set_model(opt)
    model = FineTune(opt.model, model_temp, opt.n_cls)  # the model that need to optimize
    model.to(opt.device)

    # build optimizer
    optimizer = set_optimizer(opt, model)  # ????????????model???params

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

        # eval for one epoch
        val_loss, val_acc= validate(val_loader, model, criterion, opt) # clean test data to validate
        print('Val epoch {}, loss:{:.2f}, accuracy:{:.2f}'.format(epoch, val_loss, val_acc))

        # tensorboard logger
        valid_writer.add_scalar('loss', val_loss, epoch)
        valid_writer.add_scalar('acc', val_acc, epoch)

    # end recording
    Recording(opt, start=False)


if __name__ == '__main__':
    main()