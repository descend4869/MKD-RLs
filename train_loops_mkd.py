from utils import cal_param_size, cal_multi_adds, AverageMeter, adjust_lr, adjust_lr_vit, DistillKL, correct_num
import random
import time
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

import os
import shutil
import argparse
import numpy as np
from distiller_zoo import FeatureKLLoss, FeatureMSELoss

def train_camkd(train_loader, model, criterion_list, optimizer, epoch, device, 
          args, feat_trans, teacher_models,
          # 下面是新加的，分别对应CA-MKD论文中的超参数alpha和beta，默认值也参考了论文
          alpha=1, beta=50):
    
    train_loss = AverageMeter('train_loss', ':.4e')
    train_loss_cls = AverageMeter('train_loss_cls', ':.4e')
    train_loss_kd = AverageMeter('train_loss_kd', ':.4e')
    train_loss_feat = AverageMeter('train_loss_feat', ':.4e')

    top1_num = 0
    top5_num = 0
    total = 0

    lr = adjust_lr(optimizer, epoch, args)

    start_time = time.time()
    criterion_ce = criterion_list[0]
    criterion_div = criterion_list[1]

    model.train()
    
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        batch_start_time = time.time()
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        
        features, logits = model(inputs, is_feat=True) 
        trans_student_features = feat_trans(features[-2])
        
        teacher_logits = []
        teacher_features = []
        teacher_embeddings = []
        with torch.no_grad():
            for t_model in teacher_models:
                t_features, t_logits = t_model(inputs, is_feat=True)
                t_logits = t_logits.detach() 
                
                teacher_features.append(t_features[-2].detach())
                teacher_logits.append(t_logits)
                teacher_embeddings.append(t_features[-1])

        # 对应论文中的L_CE
        loss_cls = criterion_ce(logits, targets)
        
        # 对应论文中的L_KD
        # 先计算教师的logit置信度权重，即论文中的 w_KD^k，不过之后我让 w_inter^k = w_KD^k，就不重新计算feature置信度权重了
        criterion_cls_lc = nn.CrossEntropyLoss(reduction='none') 
            # reduction='none': 不对batch内的损失做平均或求和，返回每个样本的损失.
            # 因此返回的形状为[batch_size]
        loss_t_list = [criterion_cls_lc(logit_t, targets) for logit_t in teacher_logits]
            # logit_t: 第i个教师模型的输出，形状为[batch_size, num_classes]
            # targets: 真实标签，形状为[batch_size]，每个值是类别的索引，范围[0, num_classes - 1]
            # 输出一个列表，长度为teacher_num，每个元素是形状为[batch_size]的loss
        loss_t = torch.stack(loss_t_list, dim=0)
            # 将所有教师的损失堆叠成一个tensor，形状为[teacher_num, batch_size]，每一行是一个教师在所有样本上的损失
        attention = (1.0 - F.softmax(loss_t, dim=0)) / (args.teacher_num - 1)
            # 计算每个教师对每个样本的置信度权重，形状依然为[teacher_num, batch_size]
        # 再计算L_KD
        # loss_kd = torch.tensor(0.).cuda(args.gpu)
        loss_div_list = [criterion_div(logits, logit_t, unreduce=True) for logit_t in teacher_logits]
            # 我发现 MTKD-RL 和 CA-MKD 的 criterion_div 其实都是自定义了一个叫 DistillKL 的类，
            # 且 DistillKL 的实现基本完全一致，只不过参数名有所区别，因此可以直接用.
            # 输出一个列表，长度为teacher_num，每个元素是形状为[batch_size]的loss
        loss_div = torch.stack(loss_div_list, dim=0) # 与loss_t类似的堆叠，形状也是[teacher_num, batch_size]
        loss_kd = (torch.mul(attention, loss_div).sum()) / (1.0 * args.batch_size * args.teacher_num)
            # 将置信度权重与蒸馏损失逐元素相乘，然后求和并归一化，得到最终的加权蒸馏损失.
        
        # 对应论文中的L_inter
        # loss_feat = torch.tensor(0.).cuda(args.gpu)
        if args.feat_kd == 'mse':
            feat_kd_func = FeatureMSELoss()
        elif args.feat_kd == 'kl':
            feat_kd_func = FeatureKLLoss(args.kd_T)
        loss_st = []
        for mid_feat_s, mid_feat_t in zip(trans_student_features, teacher_features):
            tmp_loss_st = feat_kd_func(mid_feat_s, mid_feat_t) # 形状为[batch_size]
            loss_st.append(tmp_loss_st)
        loss_st = torch.stack(loss_st, dim=0) 
            # 原先是长为teacher_num，元素均为[batch_size]的列表. 堆叠后形状为[teacher_num, batch_size]
        # loss = torch.mul(weight, loss_st).sum()
        loss_feat = torch.mul(attention, loss_st).sum()
        loss_feat /= (1.0 * args.batch_size * args.teacher_num)
        
        
        loss = loss_cls + alpha * loss_kd + beta * loss_feat
        loss.backward()
        optimizer.step()
        
        train_loss.update(loss.item(), inputs.size(0))
        train_loss_cls.update(loss_cls.item(), inputs.size(0))
        train_loss_kd.update(loss_kd.item(), inputs.size(0))
        train_loss_feat.update(loss_feat.item(), inputs.size(0))
        
        top1, top5 = correct_num(logits, targets, topk=(1, 5))
        top1_num += top1
        top5_num += top5
        total += targets.size(0)

        if args.rank == 0:
            print('Epoch:{}, batch_idx:{}/{}, lr:{:.5f}, Duration:{:.2f}, CLS Loss:{:.2f},' 
                'KD Loss:{:.2f}, Feature Loss:{:.2f}, Top-1 Acc:{:.2f}'.format(
                epoch, batch_idx, len(train_loader), lr, time.time()-batch_start_time, 
                train_loss_cls.avg, train_loss_kd.avg, train_loss_feat.avg, 
                (top1_num/total*100.).item()))
    
    acc1 = top1_num / total
    acc5 = top5_num / total

    if args.rank == 0:
        args.logger.info('Epoch:{}\t lr:{:.4f}\t Duration:{:.3f}'
                    '\n Train_loss:{:.5f}'
                    '\t Train_loss_cls:{:.5f}'
                    '\t Train_loss_kd:{:.5f}'
                    '\t Train_loss_feat:{:.5f}'
                    '\nTrain top-1 accuracy:{:.2f}'
                    .format(epoch, lr, time.time() - start_time,
                            train_loss.avg,
                            train_loss_cls.avg,
                            train_loss_kd.avg,
                            train_loss_feat.avg,
                            acc1*100.))