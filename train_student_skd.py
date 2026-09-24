"""Single-teacher DKD/DIST student training entry point."""

import argparse
import datetime
import os
import random
import shutil
import warnings

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim

from train_loops import test
from train_loops_skd import train_dkd, train_dist
from models import model_dict
from setting import teacher_model_path_dict
from dataset.cifar100 import get_cifar100_dataloaders
from utils import DistillKL, set_logger


parser = argparse.ArgumentParser(description='Single-teacher training')
parser.add_argument('--data', metavar='DIR', default='imagenet')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18_imagenet')
parser.add_argument('-j', '--workers', default=8, type=int)
parser.add_argument('--epochs', default=240, type=int)
parser.add_argument('--start-epoch', default=0, type=int)
parser.add_argument('-b', '--batch-size', default=64, type=int)
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float, dest='lr')
parser.add_argument('--momentum', default=0.9, type=float)
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float, dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int)
parser.add_argument('--resume', default='', type=str)
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true')
parser.add_argument('--pretrained', dest='pretrained', action='store_true')
parser.add_argument('--world-size', default=-1, type=int)
parser.add_argument('--rank', default=-1, type=int)
parser.add_argument('--dist-url', default='tcp://127.0.0.1:23456', type=str)
parser.add_argument('--dist-backend', default='nccl', type=str)
parser.add_argument('--seed', default=None, type=int)
parser.add_argument('--gpu', default=None, type=int)
parser.add_argument('--multiprocessing-distributed', action='store_true')
parser.add_argument('--dummy', action='store_true')
parser.add_argument('--milestones', default=[150, 180, 210], type=int, nargs='+')
parser.add_argument('--init-lr', default=0.05, type=float)
parser.add_argument('--lr-type', default='multistep', type=str)
parser.add_argument('--feat-kd', default='mse', type=str)
parser.add_argument('--kd-T', type=int, default=4)
parser.add_argument('--checkpoint-dir', default='/data/myh/checkpoints/mkd_checkpoints/student_skd', type=str)
parser.add_argument('--teacher-name-list', default=['resnet32x4'], type=str, nargs='+')
parser.add_argument('--dataset', type=str, default='cifar100', choices=['cifar100', 'imagenet', 'tinyimagenet', 'dogs', 'cub_200_2011', 'mit67'])
parser.add_argument('--trial', type=str, default='1')
parser.add_argument('--method', choices=['dkd', 'dist'], default='dkd')
parser.add_argument('--dkd-alpha', type=float, default=1.0)
parser.add_argument('--dkd-beta', type=float, default=8.0)
parser.add_argument('--dist-beta', type=float, default=1.0)
parser.add_argument('--dist-gamma', type=float, default=1.0)
parser.add_argument('--kd-weight', type=float, default=1.0)


def main():
    args = parser.parse_args()
    if len(args.teacher_name_list) != 1:
        parser.error('DKD/DIST require exactly one teacher; pass one --teacher-name-list value')
    args.teacher_name_str = args.teacher_name_list[0]
    args.teacher_num = 1
    args.model_name = '{}_{}_{}_{}_{}'.format(args.arch, args.dataset,
                                               args.method, args.trial,
                                               args.teacher_name_str)
    info = args.model_name + datetime.datetime.now().strftime('%d-%m-%Y_%H-%M-%S')
    args.checkpoint_dir = os.path.join(args.checkpoint_dir, info)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    if args.rank == 0:
        args.log_txt = os.path.join(args.checkpoint_dir, info + '.txt')
        args.logger = set_logger(args.log_txt)
        args.logger.info('==========\nArgs:{}\n=========='.format(args))
    if args.seed is not None:
        random.seed(args.seed); torch.manual_seed(args.seed)
        cudnn.deterministic, cudnn.benchmark = True, False
        warnings.warn('Deterministic training is enabled.')
    if args.dist_url == 'env://' and args.world_size == -1:
        args.world_size = int(os.environ['WORLD_SIZE'])
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    # The rest of this project uses rank 0 as the single-process rank.
    if not args.distributed:
        args.rank = 0
    ngpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
    if args.multiprocessing_distributed:
        args.world_size = ngpus * args.world_size
        mp.spawn(main_worker, nprocs=ngpus, args=(ngpus, args))
    else:
        main_worker(args.gpu, ngpus, args)


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu
    if args.distributed:
        if args.dist_url == 'env://' and args.rank == -1:
            args.rank = int(os.environ['RANK'])
        if args.multiprocessing_distributed:
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    def load_teacher(model_name):
        teacher = model_dict[model_name](num_classes=args.n_cls).cuda()
        teacher.load_state_dict(torch.load(teacher_model_path_dict[model_name],
                                           map_location='cuda:0')['model'])
        teacher.eval()
        for parameter in teacher.parameters():
            parameter.requires_grad = False
        return teacher

    if args.dataset.startswith('cifar100'):
        args.n_cls, args.res = 100, (1, 3, 32, 32)
    elif args.dataset.startswith('imagenet'):
        args.n_cls, args.res = 1000, (1, 3, 224, 224)
    else:
        raise NotImplementedError('Only CIFAR100 and ImageNet are wired in this entry point')

    teacher = load_teacher(args.teacher_name_list[0])
    model = model_dict[args.arch](num_classes=args.n_cls).cuda()
    if args.resume:
        state = torch.load(args.resume, map_location='cuda:0')
        model.load_state_dict(state['model']); args.start_epoch = state.get('epoch', 0)

    if args.distributed:
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model = model.cuda(args.gpu)
            teacher = teacher.cuda(args.gpu)
            args.batch_size //= ngpus_per_node
            args.workers = (args.workers + ngpus_per_node - 1) // ngpus_per_node
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model = torch.nn.parallel.DistributedDataParallel(model.cuda())
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu); model = model.cuda(args.gpu); teacher = teacher.cuda(args.gpu)

    device = torch.device('cuda:%d' % args.gpu if args.gpu is not None and torch.cuda.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
    if args.rank == 0 and not hasattr(args, 'logger'):
        args.log_txt = os.path.join(args.checkpoint_dir, 'train.log')
        args.logger = set_logger(args.log_txt)
    criterion_list = nn.ModuleList([nn.CrossEntropyLoss().to(device), DistillKL(args.kd_T).to(device)])
    optimizer = optim.SGD(model.parameters(), lr=args.init_lr, momentum=args.momentum,
                          weight_decay=args.weight_decay, nesterov=True)
    train_loader, val_loader = get_cifar100_dataloaders(data_folder=args.data,
                                                        batch_size=args.batch_size,
                                                        num_workers=args.workers)
    best_acc = 0.
    for epoch in range(args.start_epoch, args.epochs):
        loop = train_dkd if args.method == 'dkd' else train_dist
        loop(train_loader, model, criterion_list, optimizer, epoch, device, args, teacher)
        acc = test(epoch, model, device, val_loader, criterion_list[0], args)
        if args.rank == 0:
            state = {'epoch': epoch + 1, 'arch': args.arch,
                     'model': model.module.state_dict() if args.distributed else model.state_dict(),
                     'acc': acc, 'optimizer': optimizer.state_dict()}
            path = os.path.join(args.checkpoint_dir, args.arch + '.pth.tar')
            torch.save(state, path)
            if acc > best_acc:
                best_acc = acc; shutil.copyfile(path, os.path.join(args.checkpoint_dir, args.arch + '_best.pth.tar'))
    
    if args.rank == 0 :
        args.logger.info('Evaluate the best model:')
        args.evaluate = True
        checkpoint = torch.load(os.path.join(args.checkpoint_dir, args.arch + '_best.pth.tar'),
                                    map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model'])
        top1_acc = test(epoch, model, device, val_loader, criterion_list[0], args)
        args.logger.info('Test top-1 best_accuracy: {}'.format(top1_acc))
        args.logger.info('load pre-trained weights from: {}'.format(os.path.join(args.checkpoint_dir,  args.arch + '_best.pth.tar')))


if __name__ == '__main__':
    main()
