# New: 将RL中的REINFORCE更新成SAC(soft actor-critic),即在train_student_rl.py上进行一定的修改.
# (第2版)
import argparse
import os
import random
import shutil
import time
import warnings
from enum import Enum
import datetime
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms


from train_loops import train, test
from train_loops_sac2 import train_sac, update_actor
from torch.utils.tensorboard import SummaryWriter
from models import model_dict
from setting import  teacher_model_path_dict
from dataset.cifar100 import get_cifar100_dataloaders; from dataset.imagenet import get_imagenet_dataloaders
from utils import set_logger
from models.util import Regress, TransFeat

from models.SAC import ReplayBuffer
from models.TeacherWrapper import TeacherWrapper


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data', metavar='DIR', nargs='?', default='imagenet',
                    help='path to dataset (default: imagenet)')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18_imagenet')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=160, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')  # 32*2
                         
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('--dummy', action='store_true', help="use fake data to benchmark")
parser.add_argument('--dynamic', action='store_true', help="use dynamic weight aggregation strategy")
parser.add_argument('--ce-weight', type=float, default=1, help='ce loss coefficient')
parser.add_argument('--kd-weight', type=float, default=1, help='kd loss coefficient')
parser.add_argument('--feat-weight', type=float, default=5, help='kd loss coefficient')


import models
from utils import cal_param_size, cal_multi_adds, AverageMeter, adjust_lr, DistillKL, correct_num
parser.add_argument('--milestones', default=[100,120,140], type=int, nargs='+', help='milestones for lr-multistep')
parser.add_argument('--init-lr', default=0.05, type=float, help='learning rate')
parser.add_argument('--lr-type', default='multistep', type=str, help='learning rate strategy')
parser.add_argument('--feat-kd', default='mse', type=str, help='feature kd loss')
parser.add_argument('--kd-T', type=int, default=4, help='temperature')
parser.add_argument('--agent-step', type=int, default=1000, help='agent optimization step')
#parser.add_argument('--checkpoint-dir', default='./checkpoint', type=str, help='checkpoint directory')
parser.add_argument('--checkpoint-dir', default='/data/myh/checkpoints/mkd_checkpoints/student_sac2', type=str, help='checkpoint directory')
parser.add_argument('--teacher-name-list', default=['resnet32x4', 'wrn_28_4'], type=str, nargs='+', help='teacher models')
parser.add_argument('--dataset', type=str, default='cifar100', choices=['cifar100', 'imagenet', 'tinyimagenet', 'dogs', 'cub_200_2011', 'mit67'], help='dataset')
parser.add_argument('--trial', type=str, default='1', help='trial id')

def main():
    args = parser.parse_args()
    args.teacher_name_str = "_".join(args.teacher_name_list)
    print('args.teacher_name_str', args.teacher_name_str)
    args.teacher_num = len(args.teacher_name_list)

    args.model_name = args.arch + '_'+ args.dataset+ '_'+ 'sac2'+'_'+ args.trial+'_'+str(args.teacher_num)+'_'+args.teacher_name_str

    info_time = datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    info = args.model_name + info_time
    print(f'===> info is : {info}')
    args.checkpoint_dir = os.path.join(args.checkpoint_dir, info)
    if not os.path.isdir(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)
    print(f'===>args.checkpoint_dir is : {args.checkpoint_dir}')

    if args.rank == 0 :
        args.log_txt = os.path.join(args.checkpoint_dir, info + '.txt')
        args.logger = set_logger(args.log_txt)
        args.logger.info("==========\nArgs:{}\n==========".format(args))

    if args.seed is not None :
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        cudnn.benchmark = False
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')
    
    if args.gpu is not None :
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')
    
    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed # True
    print(f'======> args.distributed is {args.distributed}')

    if torch.cuda.is_available():
        ngpus_per_node = torch.cuda.device_count()
    else:
        ngpus_per_node = 1
    print(f'======> ngpus_per_node is {ngpus_per_node}') 
    

    
    if args.multiprocessing_distributed:
        args.world_size = ngpus_per_node * args.world_size 
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        main_worker(args.gpu, ngpus_per_node, args)

def get_sac_actor_critic(teacher_models, args):
    teacher_num = len(teacher_models)
    x = torch.rand(args.res).cuda() #由224~229行对args.res的定义可知,x是(1,3,32,32)或(1,3,224,224)形状的vector
    logits_dim = 0 #累加所有教师模型分类输出维度的总和。
    feature_dim = 0 #累加所有教师模型最后一层特征维度的总和。
    feature_dims = [] #记录每个教师模型倒数第二层特征的形状。
    policy_input_size = []
    for t in teacher_models :
        feature, logits = t(x, is_feat=True) #feature是一个列表,包含教师模型中不同层的特征.每个元素的形状取决于教师模型的架构。
        logits_dim += logits.size(1) #logits是教师模型的分类输出,形状是(1,num_classes)
        feature_dim += feature[-1].size(1) #若教师模型是ResNet-like网络,则倒数第一层feature[-1]形状为(1,D).
        feature_dims.append(feature[-2].size()) #教师为ResNet-like网络,则倒数第二层feature[-2]形状为(1,C,H,W).
        policy_input_size.append(feature[-1].size(1) + logits.size(1) + 3)
        #policy_input_size其实对应的是state中的teacher_infos,而不是完整的state.
    actor = model_dict['SACActor'](policy_input_size, teacher_num, args.dynamic).cuda()
    critic_1 = model_dict['SACCritic'](policy_input_size, teacher_num).cuda() #双Q网络要有两个critic实例
    critic_2 = model_dict['SACCritic'](policy_input_size, teacher_num).cuda()
    # 再加上对应的target_critic
    target_critic_1 = model_dict['SACCritic'](policy_input_size, teacher_num).cuda()
    target_critic_2 = model_dict['SACCritic'](policy_input_size, teacher_num).cuda()
    # 将target_critic的参数设置为与critic网络相同
    target_critic_1.load_state_dict(critic_1.state_dict())
    target_critic_2.load_state_dict(critic_2.state_dict())

    # 构造ReplayBuffer实例!
    state_dim = (logits_dim + feature_dim + 3 * teacher_num) + 3 * teacher_num
        #与train_loops中的get_agent_state相对照,括号内的是teacher_infos部分,后面的3 * teacher_num对应t_ces, t_s_logit_div, t_s_feat_div这三个.
        #teacher_infos本应是对每个teacher计算(3 + embedding_dim + num_classes)再求和,
        #其中的embedding_dim对应代码169行feature[-1].size(1),num_classes对应代码168行的logits.size(1).
        #而logits_dim和feature_dim恰好已经是对所有教师的求和了,因此这两个不需要再乘teacher_num,只需要将3去乘teacher_num即可.
    action_dim = teacher_num * 2  # 动作维度 (logits_actions + feature_actions)
    buffer_size = 1280 # 通过限制buffer的大小其实就可以限制sample中从最近的几个batch中获取数据, 如640即从最近的10个batch中取样.
    device = 'cuda' if torch.cuda.is_available() else 'cpu' #之后还会重新设置过device!
    replay_buffer = ReplayBuffer(buffer_size, state_dim, action_dim, device)

    return actor, critic_1, critic_2, target_critic_1, target_critic_2, replay_buffer, feature_dims


def get_feat_trans(model, args):
    model.eval()
    with torch.no_grad():
        s_feat, s_logits = model(torch.rand(args.res).cuda(), is_feat=True)
    args.s_feat_dim = s_feat[-2].size()
    return TransFeat(args.s_feat_dim, args.t_feat_dims).cuda()

def main_worker(gpu, ngpus_per_node, args):
    
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])

        if args.multiprocessing_distributed:
            args.rank = args.rank * ngpus_per_node + gpu 
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
        
        
    def load_teacher(model_path, n_cls, model_t, opt=None):
        model = model_dict[model_t](num_classes=n_cls).cuda()
        map_location = None if opt.gpu is None else {'cuda:0': 'cuda:%d' % (opt.gpu if opt.multiprocessing_distributed else 0)}
        model.load_state_dict(torch.load(model_path, map_location=map_location)['model'])
        model.eval()
        for t_n, t_p in model.named_parameters():
            t_p.requires_grad = False
        return model
    
    # # 用于ImageNet的load_teacher函数
    # def load_teacher(model_name, opt):
    #     print(f"==> Loading teacher model: {model_name}")
    #     model = TeacherWrapper(model_name=model_name, num_classes=opt.n_cls).cuda()
    #     model.eval()
    #     for t_n, t_p in model.named_parameters():
    #         t_p.requires_grad = False
    #     return model


    def load_teacher_list(opt):
        print('==> loading teacher model list')
        teacher_model_list = [load_teacher(teacher_model_path_dict[model_name], args.n_cls, model_name, opt)
                            for model_name in opt.teacher_name_list]
        print('==> done')
        return teacher_model_list
    
    # # 用于ImageNet的load_teacher_list函数
    # def load_teacher_list(opt):
    #     # model_names = ['resnet50', 'resnet101', 'wide_resnet50_2', 'resnext50_32x4d']
    #     model_names = ['resnet50', 'resnet101', 'resnext50_32x4d']
    #         # 上面这四个分别是timm中在ImageNet上训练的ResNet-50，ResNet-101，Wide ResNet-50-2，ResNeXt-50(32×4d)的名字
    #     print('==> Loading teacher model list')
    #     teacher_model_list = [load_teacher(model_name, opt) for model_name in model_names]
    #     print('==> Done loading teacher models')
    #     return teacher_model_list
    
    if args.dataset.startswith('cifar100'):
        args.n_cls = 100
        args.res = (1, 3, 32, 32)
    elif args.dataset.startswith('imagenet'):
        args.n_cls = 1000
        args.res = (1, 3, 224, 224)
            
    teacher_models = load_teacher_list(args)

    ##### load student model #####
    model = model_dict[args.arch](num_classes=args.n_cls).cuda()

    args.start_epoch = 0
    if len(args.resume) != 0:
        map_location = None if args.gpu is None else 'cuda:%d' % (args.gpu if args.multiprocessing_distributed else 0)
        model_info_dict = torch.load(args.resume, map_location=map_location)
        model.load_state_dict(model_info_dict['model'])
        args.start_epoch = model_info_dict['epoch']
        
    print('======> load student model finish')
    actor, critic_1, critic_2, target_critic_1, target_critic_2, replay_buffer, args.t_feat_dims = get_sac_actor_critic(teacher_models, args)
    print("===> get agent finish......")

    feat_trans = get_feat_trans(model, args)
    print("===> get feat_trans finish......")

    #下面将agent改成了actor以及两个critic,还新增了replay_buffer的device设置.
    if args.distributed:
        if torch.cuda.is_available():
            if args.gpu is not None:
                torch.cuda.set_device(args.gpu)
                for teacher in teacher_models :
                    teacher.cuda(args.gpu)
                actor.cuda(args.gpu)
                critic_1.cuda(args.gpu)
                critic_2.cuda(args.gpu)
                replay_buffer.device = torch.device('cuda:{}'.format(args.gpu))
                model.cuda(args.gpu)
                args.batch_size = int(args.batch_size / ngpus_per_node)
                args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
                model = torch.nn.parallel.DistributedDataParallel(model,device_ids=[args.gpu]) 
                for teacher in teacher_models : 
                    teacher = torch.nn.parallel.DistributedDataParallel(teacher,device_ids=[args.gpu])
                actor = torch.nn.parallel.DistributedDataParallel(actor,device_ids=[args.gpu])
                critic_1 = torch.nn.parallel.DistributedDataParallel(critic_1,device_ids=[args.gpu])
                critic_2 = torch.nn.parallel.DistributedDataParallel(critic_2,device_ids=[args.gpu])
                
            else:
                model.cuda()
                model = torch.nn.parallel.DistributedDataParallel(model)
                for teacher in teacher_models :
                    teacher = teacher.cuda()
                actor = actor.cuda()
                actor = torch.nn.parallel.DistributedDataParallel(actor)  
                critic_1.cuda()
                critic_1 = torch.nn.parallel.DistributedDataParallel(critic_1)
                critic_2.cuda()
                critic_2 = torch.nn.parallel.DistributedDataParallel(critic_2)
                replay_buffer.device = torch.device("cuda")
                
    elif args.gpu is not None and torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        actor = actor.cuda(args.gpu)
        critic_1.cuda(args.gpu)
        critic_2.cuda(args.gpu)
        replay_buffer.device = torch.device('cuda:{}'.format(args.gpu))
        for teacher in teacher_models :
            teacher = teacher.cuda(args.gpu)
    else:
        model = torch.nn.DataParallel(model).cuda()
        replay_buffer.device = torch.device("cuda")
    
    if torch.cuda.is_available():
        if args.gpu:
            device = torch.device('cuda:{}'.format(args.gpu))
        else:
            device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    ################### loss function and optimizer ###################
   
          
    criterion_list = nn.ModuleList([])
    criterion_ce = nn.CrossEntropyLoss().to(device)
    criterion_div = DistillKL(args.kd_T).to(device) #这里的args.kd_T仅为初始蒸馏温度,后面将被动态替换
    criterion_list.append(criterion_ce)
    criterion_list.append(criterion_div)

    trainable_list = nn.ModuleList([])
    trainable_list.append(model)
    trainable_list.append(feat_trans)

    optimizer = optim.SGD(trainable_list.parameters(),
                        lr=0.1, momentum=0.9, weight_decay=args.weight_decay, nesterov=True)
    # args.init_lr = 0.001
    # args.warmup_epochs = 10
    # args.lr_type = 'cosine'
    # optimizer = optim.AdamW(trainable_list.parameters(), lr=0.001, weight_decay=0.05)

    # actor_optimizer = optim.SGD(actor.parameters(), lr=0.1, )
    actor_optimizer = optim.Adam(actor.parameters(), lr=0.0003)
    #下面两个的optimizer种类以及learning rate可以尝试修改(当前是随便选了一个)
    critic_1_optimizer = optim.Adam(critic_1.parameters(), lr=0.001)
    critic_2_optimizer = optim.Adam(critic_2.parameters(), lr=0.001)

    # 又新增了SAC中熵正则项系数alpha: 一个可训练参数以及一个optimizer
    H_alpha = torch.tensor(0.2, requires_grad=True, device=device)  # 初始值为 0.2
    alpha_optimizer = optim.Adam([H_alpha], lr=0.001)

    ################### load data ###################

    train_loader, val_loader = get_cifar100_dataloaders(data_folder=args.data,
                                                        batch_size=args.batch_size,
                                                        num_workers=args.workers)
    '''
    train_loader, val_loader = get_imagenet_dataloaders(data_folder='/data/shared/datasets/IMNET/ILSVRC2012',
                                                        batch_size=args.batch_size,
                                                        num_workers=args.workers)
    '''
    ################### train model ###################
    best_acc = 0.  # best test accuracy
    
    t_results = []
    for t_model in teacher_models:
        acc = test(0, t_model, device, val_loader, criterion_ce, args, verbose=False)
        t_results.append(round(acc, 2))
        torch.cuda.empty_cache()
    args.logger.info('Teacher accruacy: '+ str(t_results))

    for epoch in range(args.start_epoch, args.epochs) :
        # 在critic相对稳定之后每隔几个epoch就尝试多次更新actor
        if epoch >= 80 and epoch % 10 == 0: 
            for _ in range(5):
                update_actor(args, epoch, replay_buffer, actor, actor_optimizer, critic_1, critic_2, H_alpha)
        
        train_sac(train_loader, model, criterion_list, optimizer, epoch, device, args, actor, feat_trans, teacher_models, actor_optimizer, 
                  #下面这些是新加的,上面一行将agent改成actor
                  critic_1, critic_1_optimizer, critic_2, critic_2_optimizer, target_critic_1, target_critic_2, 
                  replay_buffer, H_alpha, alpha_optimizer)
        acc = test(epoch, model, device, val_loader, criterion_ce, args)

        if args.rank == 0 :
            state = {
                    'epoch' : epoch + 1,
                    'arch' : args.arch, 
                    'model': model.module.state_dict() if args.distributed else model.state_dict() ,
                    'acc': acc,
                    'epoch': epoch,
                    'optimizer': optimizer.state_dict()
            }

            torch.save(state, os.path.join(args.checkpoint_dir, args.arch+'.pth.tar'))

            is_best = False
            if best_acc < acc:
                best_acc = acc
                is_best = True
            if is_best:
                shutil.copyfile(os.path.join(args.checkpoint_dir, args.arch + '.pth.tar'),
                                    os.path.join(args.checkpoint_dir, args.arch + '_best.pth.tar'))

    
    if args.rank == 0 :
        args.logger.info('Evaluate the best model:')
        args.evaluate = True
        checkpoint = torch.load(os.path.join(args.checkpoint_dir, args.arch + '_best.pth.tar'),
                                    map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model'])
        top1_acc = test(epoch, model, device, val_loader, criterion_ce, args)
        args.logger.info('Test top-1 best_accuracy: {}'.format(top1_acc))
        args.logger.info('load pre-trained weights from: {}'.format(os.path.join(args.checkpoint_dir,  args.arch + '_best.pth.tar')))

if __name__ == '__main__' :
     os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
     main()