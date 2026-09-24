from utils import cal_param_size, cal_multi_adds, AverageMeter, adjust_lr, adjust_lr_vit, DistillKL, correct_num
import random
import time
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from collections import OrderedDict

try:
    from torch.func import functional_call
except ImportError:  # PyTorch < 2.0
    from torch.nn.utils.stateless import functional_call

import os
import shutil
import argparse
import numpy as np
from distiller_zoo import FeatureKLLoss, FeatureMSELoss


class MMKDLogitsWeight(nn.Module):
    """Meta-weight network for the output/logit distillation losses."""
    def __init__(self, n_cls, teacher_num, factor=2):
        super().__init__()
        n_feature = n_cls * (teacher_num + 1)
        self.layer = nn.Linear(n_feature, max(n_feature // factor, 1))
        self.relu = nn.ReLU(inplace=True)
        self.all_act = nn.Linear(max(n_feature // factor, 1), teacher_num)

    def forward(self, teacher_logits, student_logits):
        x = torch.cat(teacher_logits + [student_logits], dim=1)
        x = self.relu(self.layer(x))
        return F.softmax(self.all_act(x), dim=-1)


class MMKDFeatureWeight(nn.Module):
    """Meta-weight network for intermediate features (MMKD, Eq. for w_f)."""
    def __init__(self, batch_size, teacher_num, factor=2):
        super().__init__()
        n_feature = batch_size * (teacher_num + 1)
        self.batch_size = batch_size
        self.layer = nn.Linear(n_feature, max(n_feature // factor, 1))
        self.relu = nn.ReLU(inplace=True)
        self.all_act = nn.Linear(max(n_feature // factor, 1), teacher_num)

    def forward(self, teacher_features, student_feature):
        # MMKD represents each feature map by its B x B sample-similarity matrix.
        # Padding the columns keeps the last, possibly short, batch usable.
        batch_size = student_feature.size(0)
        states = []
        for feature in teacher_features + [student_feature]:
            flat = feature.reshape(batch_size, -1)
            similarity = torch.matmul(flat, flat.t())
            if batch_size < self.batch_size:
                padded = similarity.new_zeros(batch_size, self.batch_size)
                padded[:, :batch_size] = similarity
                similarity = padded
            elif batch_size > self.batch_size:
                raise ValueError('MMKD feature-weight batch is larger than configured batch_size')
            states.append(similarity)
        x = torch.cat(states, dim=1)
        x = self.relu(self.layer(x))
        return F.softmax(self.all_act(x), dim=-1)


def _mmkd_losses(logits, trans_student_features, teacher_logits, teacher_features,
                 criterion_ce, criterion_div, feat_kd_func, targets,
                 logits_weight, feature_weight, alpha, beta):
    """Build the MMKD inner objective and keep it sample-wise for weighting."""
    loss_cls = criterion_ce(logits, targets)
    loss_div = torch.stack([
        criterion_div(logits, logit_t, unreduce=True)
        for logit_t in teacher_logits
    ], dim=1)
    loss_kd = (logits_weight * loss_div).sum(dim=1).mean()

    loss_feat = torch.stack([
        feat_kd_func(feat_s, feat_t)
        for feat_s, feat_t in zip(trans_student_features, teacher_features)
    ], dim=1)
    loss_feat = (feature_weight * loss_feat).sum(dim=1).mean()
    return loss_cls + alpha * loss_kd + beta * loss_feat, loss_cls, loss_kd, loss_feat


def train_mmkd(train_loader, model, criterion_list, optimizer, epoch, device,
               args, feat_trans, teacher_models, weight_logits, weight_feature,
               weight_optimizer, alpha=1, beta=50):
    """Train with MMKD's bilevel meta-learning of teacher weights.

    The current batch is used as the meta batch, matching the public MMKD
    implementation when its optional hard buffer is disabled.  A differentiable
    one-step student update is evaluated on the classification loss, and that
    loss updates the two meta-weight networks.
    """
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

    if args.feat_kd == 'mse':
        feat_kd_func = FeatureMSELoss()
    elif args.feat_kd == 'kl':
        feat_kd_func = FeatureKLLoss(args.kd_T)
    else:
        raise ValueError('Unsupported feature KD loss: {}'.format(args.feat_kd))

    model.train()
    feat_trans.train()
    meta_freq = max(getattr(args, 'meta_freq', 5), 1)
    meta_enabled = epoch >= getattr(args, 'meta_warmup', 0)
    model_base = model.module if hasattr(model, 'module') else model
    feat_trans_base = feat_trans.module if hasattr(feat_trans, 'module') else feat_trans

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        # The feature-weight network is defined for the configured batch size.
        if inputs.size(0) > args.batch_size:
            continue
        batch_start_time = time.time()
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad()
        if weight_optimizer is not None:
            weight_optimizer.zero_grad()

        features, logits = model(inputs, is_feat=True)
        trans_student_features = feat_trans(features[-2])
        teacher_logits = []
        teacher_features = []
        with torch.no_grad():
            for t_model in teacher_models:
                t_features, t_logits = t_model(inputs, is_feat=True)
                teacher_features.append(t_features[-2].detach())
                teacher_logits.append(t_logits.detach())

        # Student-side weights are detached for ordinary student updates; at a
        # meta step the non-detached version is retained to form the meta graph.
        logits_weight = weight_logits(teacher_logits, logits.detach())
        feature_weight = weight_feature(teacher_features, features[-2].detach())
        do_meta = meta_enabled and (batch_idx % meta_freq == 0)
        if not do_meta:
            logits_weight = logits_weight.detach()
            feature_weight = feature_weight.detach()

        loss, loss_cls, loss_kd, loss_feat = _mmkd_losses(
            logits, trans_student_features, teacher_logits, teacher_features,
            criterion_ce, criterion_div, feat_kd_func, targets,
            logits_weight, feature_weight, alpha, beta)

        meta_grads = None
        if do_meta:
            # MMKD's virtual inner update optimizes only the matching losses;
            # the supervised CE loss is reserved for the outer/meta objective.
            meta_inner_loss = alpha * loss_kd + beta * loss_feat
            student_params = list(model_base.parameters()) + list(feat_trans_base.parameters())
            inner_grads = torch.autograd.grad(
                meta_inner_loss, student_params, create_graph=True, retain_graph=True,
                allow_unused=True)
            inner_lr = optimizer.param_groups[0]['lr']
            model_param_count = len(list(model_base.parameters()))
            model_params = OrderedDict(model_base.named_parameters())
            virtual_model_params = OrderedDict(
                (name, param - inner_lr * (grad if grad is not None else torch.zeros_like(param)))
                for (name, param), grad in zip(model_params.items(), inner_grads[:model_param_count])
            )

            # The outer/meta objective is the student's supervised loss after
            # the virtual inner update, as in MMKD's meta-learning loop.
            was_training = model_base.training
            model_base.eval()
            _, virtual_logits = functional_call(
                model_base, virtual_model_params, (inputs,), {'is_feat': True})
            meta_loss = criterion_ce(virtual_logits, targets)
            if was_training:
                model_base.train()
            meta_params = list(weight_logits.parameters()) + list(weight_feature.parameters())
            meta_grads = torch.autograd.grad(
                meta_loss, meta_params, retain_graph=True, allow_unused=True)

        # Update the student using the current weighted KD objective.
        loss.backward()
        optimizer.step()

        if do_meta and meta_grads is not None:
            weight_optimizer.zero_grad()
            for parameter, grad in zip(
                    list(weight_logits.parameters()) + list(weight_feature.parameters()),
                    meta_grads):
                if grad is not None:
                    parameter.grad = grad.detach()
            weight_optimizer.step()

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
                  ' KD Loss:{:.2f}, Feature Loss:{:.2f}, Top-1 Acc:{:.2f}'.format(
                      epoch, batch_idx, len(train_loader), lr,
                      time.time() - batch_start_time, train_loss_cls.avg,
                      train_loss_kd.avg, train_loss_feat.avg,
                      (top1_num / total * 100.).item()))

    acc1 = top1_num / total
    if args.rank == 0:
        args.logger.info('Epoch:{}\t lr:{:.4f}\t Duration:{:.3f}'
                         '\n Train_loss:{:.5f}\t Train_loss_cls:{:.5f}'
                         '\t Train_loss_kd:{:.5f}\t Train_loss_feat:{:.5f}'
                         '\nTrain top-1 accuracy:{:.2f}'
                         .format(epoch, lr, time.time() - start_time,
                                 train_loss.avg, train_loss_cls.avg,
                                 train_loss_kd.avg, train_loss_feat.avg,
                                 acc1 * 100.))

def train_camkd(train_loader, model, criterion_list, optimizer, epoch, device, 
          args, feat_trans, teacher_models,
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

        loss_cls = criterion_ce(logits, targets)
        
        criterion_cls_lc = nn.CrossEntropyLoss(reduction='none') 
        loss_t_list = [criterion_cls_lc(logit_t, targets) for logit_t in teacher_logits]
        loss_t = torch.stack(loss_t_list, dim=0)
        attention = (1.0 - F.softmax(loss_t, dim=0)) / (args.teacher_num - 1)
        # loss_kd = torch.tensor(0.).cuda(args.gpu)
        loss_div_list = [criterion_div(logits, logit_t, unreduce=True) for logit_t in teacher_logits]
        loss_div = torch.stack(loss_div_list, dim=0)
        loss_kd = (torch.mul(attention, loss_div).sum()) / (1.0 * args.batch_size * args.teacher_num)
        
        # loss_feat = torch.tensor(0.).cuda(args.gpu)
        if args.feat_kd == 'mse':
            feat_kd_func = FeatureMSELoss()
        elif args.feat_kd == 'kl':
            feat_kd_func = FeatureKLLoss(args.kd_T)
        loss_st = []
        for mid_feat_s, mid_feat_t in zip(trans_student_features, teacher_features):
            tmp_loss_st = feat_kd_func(mid_feat_s, mid_feat_t)
            loss_st.append(tmp_loss_st)
        loss_st = torch.stack(loss_st, dim=0) 
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
