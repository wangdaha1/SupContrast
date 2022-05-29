# supervised CL / CL用这个

from __future__ import print_function

import os
import sys
import argparse
import time
import math

import tensorboard_logger as tb_logger  # 就直接pip install tensorboard_logger
import torch
import torch.backends.cudnn as cudnn
from torchvision import transforms, datasets

from util import TwoCropTransform, AverageMeter, Recording
from util import adjust_learning_rate, warmup_learning_rate
from util import set_optimizer, save_model
from networks.resnet_big import SupConResNet
from losses import SupConLoss

try:
    import apex # 这个可以在requirements之前手动装一下 cd apex ; python3 setup.py install
    from apex import amp, optimizers
except ImportError:
    pass


def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    # CL related method
    parser.add_argument('--method', type=str, default='SimCLR',
                        choices=['SupCon', 'SimCLR'], help='choose method')
    parser.add_argument('--temp', type=float, default=0.5,
                        help='temperature for loss function in SimCLR')
    parser.add_argument('--trial', type=str, default='0',
                        help='id for recording multiple runs')

    # training devices allocation
    parser.add_argument('--device', default='4', help="server used in DataParallel")
    parser.add_argument('--workers', default='4,5,6,7', help='workers including server used in DataParallel')

    # model dataset
    parser.add_argument('--model', type=str, default='resnet18')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100', 'path'], help='dataset')
    parser.add_argument('--data_folder', type=str, default=None, help='path to custom dataset') # 如果数据不是cifar10/100
    parser.add_argument('--mean', type=str, help='mean of dataset in path in form of str tuple')
    parser.add_argument('--std', type=str, help='std of dataset in path in form of str tuple')
    parser.add_argument('--size', type=int, default=32, help='parameter for RandomResizedCrop') # 这里如果data image的size变了也要改动哦

    # model training
    parser.add_argument('--batch_size', type=int, default=1024,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.5,
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='700,800,900',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')
    parser.add_argument('--optimizer_type', type=str, default='sgd', choices=['sgd', 'adam', 'lars'],
                        help='Optimizer to use')
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--syncBN', action='store_true',
                        help='using synchronized batch normalization')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')

    # other setting
    parser.add_argument('--print_freq', type=int, default=10,
                        help='print frequency')
    parser.add_argument('--save_freq', type=int, default=100,
                        help='save model frequency')

    opt = parser.parse_args()


    # check if dataset is path that passed required arguments
    if opt.dataset == 'path':
        assert opt.data_folder is not None \
            and opt.mean is not None \
            and opt.std is not None

    # learning rate decay
    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    # set training devices for using DataParallel
    gpus = opt.workers.split(',')
    opt.workers = list([])
    for it in gpus:
        opt.workers.append(int(it))
    server = "cuda:{}".format(int(opt.device))
    opt.device = torch.device(server if torch.cuda.is_available() else "cpu")

    # save model/results的名字和位置
    if opt.data_folder is None:
        opt.data_folder = './datasets/'
    opt.model_path = './save/{}/{}_models'.format(opt.method, opt.dataset)  # 储存模型的位置
    opt.tb_path = './save/{}/{}_tensorboard'.format(opt.method, opt.dataset)  # 储存训练的tb结果的位置
    opt.log_path='./save/{}/{}_logs'.format(opt.method, opt.dataset)  # log记录结果
    opt.model_name = '{}_{}_{}_lr_{}_decay_{}_bsz_{}_temp_{}_trial_{}'.\
        format(opt.method, opt.dataset, opt.model, opt.learning_rate,
               opt.weight_decay, opt.batch_size, opt.temp, opt.trial)

    # 如果使用了余弦退火/warm up learning rate 在model_name上再记录一下子
    if opt.cosine:
        opt.model_name = '{}_cosine'.format(opt.model_name)
    # warm-up for large-batch training
    if opt.batch_size > 256:
        opt.warm = True
    if opt.warm:
        opt.model_name = '{}_warm'.format(opt.model_name)
        opt.warmup_from = 0.01
        opt.warm_epochs = 10
        if opt.cosine:
            eta_min = opt.learning_rate * (opt.lr_decay_rate ** 3)
            opt.warmup_to = eta_min + (opt.learning_rate - eta_min) * (
                    1 + math.cos(math.pi * opt.warm_epochs / opt.epochs)) / 2
        else:
            opt.warmup_to = opt.learning_rate

    # 最后创建文件夹 别管
    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)
    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)
    opt.log_folder = os.path.join(opt.log_path, opt.model_name)
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
    elif opt.dataset == 'path':
        mean = eval(opt.mean)
        std = eval(opt.std)
    else:
        raise ValueError('dataset not supported: {}'.format(opt.dataset))
    normalize = transforms.Normalize(mean=mean, std=std)

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=opt.size, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        normalize,
    ])

    if opt.dataset == 'cifar10':
        train_dataset = datasets.CIFAR10(root=opt.data_folder,
                                         transform=TwoCropTransform(train_transform),
                                         download=True)
    elif opt.dataset == 'cifar100':
        train_dataset = datasets.CIFAR100(root=opt.data_folder,
                                          transform=TwoCropTransform(train_transform),
                                          download=True)
    elif opt.dataset == 'path':
        train_dataset = datasets.ImageFolder(root=opt.data_folder,
                                            transform=TwoCropTransform(train_transform))
    else:
        raise ValueError(opt.dataset)

    train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size, shuffle=(train_sampler is None),
        num_workers=opt.num_workers, pin_memory=True, sampler=train_sampler)

    return train_loader


def set_model(opt):
    device = opt.device
    model = SupConResNet(name=opt.model)
    criterion = SupConLoss(temperature=opt.temp)

    # enable synchronized Batch Normalization
    if opt.syncBN: # 这个目前是没有用的
        model = apex.parallel.convert_syncbn_model(model)

    if torch.cuda.is_available():
        # 为什么是把encoder放到多张卡上但是projector就没有呢
        if len(opt.workers)>1:
            model.encoder = torch.nn.DataParallel(model.encoder, device_ids=opt.workers)
        model = model.to(device)
        criterion = criterion.to(device)
        cudnn.benchmark = True # 让内置的 cudnn的 auto-tuner 自动寻找最适合当前配置的高效算法，优化运行效率

    return model, criterion


def train(train_loader, model, criterion, optimizer, epoch, opt):
    """one epoch training"""
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    end = time.time()
    for idx, (images, labels) in enumerate(train_loader):
        data_time.update(time.time() - end)

        images = torch.cat([images[0], images[1]], dim=0)
        if torch.cuda.is_available():
            # 如果pin_memory=True的话，将数据放入GPU的时候，也应该把non_blocking打开，这样就只把数据放入GPU而不取出，访问时间会大大减少
            images = images.to(opt.device, non_blocking=True)
            labels = labels.to(opt.device, non_blocking=True)
        bsz = labels.shape[0] # batchsize

        # warm-up learning rate
        warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

        # compute loss
        features = model(images)
        f1, f2 = torch.split(features, [bsz, bsz], dim=0)
        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
        if opt.method == 'SupCon':
            loss = criterion(features, labels, device=opt.device)
        elif opt.method == 'SimCLR':
            loss = criterion(features, device=opt.device)  # 看看对不对
        else:
            raise ValueError('contrastive method not supported: {}'.format(opt.method))

        # update metric
        losses.update(loss.item(), bsz)

        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % opt.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})'.format(
                   epoch, idx + 1, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses))
            sys.stdout.flush()

    return losses.avg


def main():
    opt = parse_option()

    # start recording
    Recording(opt, start=True)

    # build data loader
    train_loader = set_loader(opt)

    # build model and criterion
    model, criterion = set_model(opt)

    # build optimizer
    optimizer = set_optimizer(opt, model)

    # tensorboard
    logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)

    # training routine
    for epoch in range(1, opt.epochs + 1):
        adjust_learning_rate(opt, optimizer, epoch)

        # train for one epoch
        time1 = time.time()
        loss = train(train_loader, model, criterion, optimizer, epoch, opt)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        # tensorboard logger
        logger.log_value('loss', loss, epoch)
        logger.log_value('learning_rate', optimizer.param_groups[0]['lr'], epoch)

        if epoch % opt.save_freq == 0:
            save_file = os.path.join(opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            save_model(model, optimizer, opt, epoch, save_file)

    # save the last model
    save_file = os.path.join(opt.save_folder, 'last.pth')
    save_model(model, optimizer, opt, opt.epochs, save_file)

    # end recording
    Recording(opt, start=False)


if __name__ == '__main__':
    main()
