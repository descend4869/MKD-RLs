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

def get_agent_state(trans_student_features, teacher_embeddings, logits, teacher_logits, targets, criterion_div):
    trans_student_embeddings = []
    for idx in range(len(trans_student_features)): 
        trans_student_embedding = F.adaptive_avg_pool2d(trans_student_features[idx], (1,1)) 
        trans_student_embedding = trans_student_embedding.view(trans_student_embedding.size(0), -1)
        trans_student_embeddings.append(trans_student_embedding)

    teacher_infos = []
    t_ces = []
    t_s_feat_div = []
    t_s_logit_div = []
    for idx in range(len(teacher_embeddings)):
        feat_cos_sim = F.cosine_similarity(trans_student_embeddings[idx], teacher_embeddings[idx]).unsqueeze(-1)
        t_s_feat_div.append(feat_cos_sim)
        logit_kl = criterion_div(logits, teacher_logits[idx], unreduce=True).unsqueeze(-1)
        t_s_logit_div.append(logit_kl)
        teachers_ce = F.cross_entropy(teacher_logits[idx], targets, reduction='none').unsqueeze(-1)
        t_ces.append(teachers_ce)
        teacher_info = torch.cat([feat_cos_sim, logit_kl, teachers_ce, teacher_embeddings[idx], teacher_logits[idx]], dim=1).detach()
        teacher_infos.append(teacher_info)
    t_ces = torch.cat(t_ces, dim=1).detach()
    t_s_logit_div = torch.cat(t_s_logit_div, dim=1).detach()
    t_s_feat_div = torch.cat(t_s_feat_div, dim=1).detach()
    return teacher_infos, t_ces, t_s_logit_div, t_s_feat_div

def train_high(args, epoch, agent_state, agent_rewards, agent, agent_optimizer):
    agent.train()
    agent_loss = AverageMeter('agent_loss', ':.4e')
    for state, rewards in zip(agent_state, agent_rewards):
        
        kd_temp, teacher_importance = agent(state)
        agent_optimizer.zero_grad() 

        kd_temp_normalized = (kd_temp - 1) / 4
        
        action_label = torch.ones_like(kd_temp).detach() * 0.99
        loss_kd_temp = F.binary_cross_entropy(kd_temp_normalized, action_label, weight=rewards.unsqueeze(-1))
        loss_kd_temp += 0.01 * torch.mean(kd_temp ** 2)

        action_label2 = torch.ones_like(teacher_importance).detach()
        loss_teacher_importance = F.binary_cross_entropy(teacher_importance, action_label2, weight=rewards.unsqueeze(-1))

        loss = loss_kd_temp + loss_teacher_importance
        loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.parameters(), max_norm=2.0)
        agent_optimizer.step()

        agent_loss.update(loss.item(), rewards.size(0))

    if args.rank == 0:
        args.logger.info('Epoch:{}, high agent Loss:{:.6f}'.format(epoch, agent_loss.avg))


#def train_agent(args, epoch, agent_state, agent_rewards, logits_agent_actions, agent, agent_optimizer):
def train_agent(args, epoch, agent_state, agent_rewards, agent, agent_optimizer):
    agent.train()
    agent_loss = AverageMeter('agent_loss', ':.4e')
    # for state, rewards, actions in zip(agent_state, agent_rewards, logits_agent_actions):
    for state, rewards in zip(agent_state, agent_rewards):
        
        agent_pred = agent(state)
        agent_optimizer.zero_grad() 
        
        action_label = torch.ones_like(agent_pred[0]).detach() 
        loss_logits = F.binary_cross_entropy(agent_pred[0], action_label, weight=rewards.unsqueeze(-1))
        loss_feature = F.binary_cross_entropy(agent_pred[1], action_label, weight=rewards.unsqueeze(-1))
        loss = loss_feature + loss_logits
        loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.parameters(), max_norm=2.0)
        agent_optimizer.step()

        # agent_loss.update(loss.item(), actions.size(0))
        agent_loss.update(loss.item(), rewards.size(0))

    if args.rank == 0:
        args.logger.info('Epoch:{}, agent Loss:{:.6f}'.format(epoch, agent_loss.avg))

def adjust_teacher_weights(weights, mask_tensor):
    batch_size, teacher_num = weights.shape
    
    _, min_indices = torch.min(mask_tensor, dim=1, keepdim=True)  # (batch_size, 1)
    
    mask = torch.ones_like(weights, dtype=torch.bool)
    mask.scatter_(1, min_indices, False)
    
    adjusted_weights = weights * mask.float()
    
    row_sums = adjusted_weights.sum(dim=1, keepdim=True)
    
    adjusted_weights = adjusted_weights / row_sums
    
    return adjusted_weights 

def train_hp(train_loader, model, criterion_list, optimizer, epoch, device, 
          args, agent, feat_trans, teacher_models, agent_optimizer,
          high_agent, high_agent_optimizer):
    
    train_loss = AverageMeter('train_loss', ':.4e')
    train_loss_cls = AverageMeter('train_loss_cls', ':.4e')
    train_loss_kd = AverageMeter('train_loss_kd', ':.4e')
    train_loss_feat = AverageMeter('train_loss_feat', ':.4e')

    top1_num = 0
    top5_num = 0
    total = 0

    lr = adjust_lr(optimizer, epoch, args)
    # lr = adjust_lr_vit(optimizer, epoch, args)

    start_time = time.time()
    criterion_ce = criterion_list[0]
    criterion_div = criterion_list[1]

    model.train()
    agent.eval()
    agent_states = []
    agent_rewards = []

    # high_states = []
    # # high_actions = []
    # high_rewards = []

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
                t_feature = t_features[-1]
                t_feature = t_feature.detach() 
                t_logits = t_logits.detach() 
                
                teacher_features.append(t_features[-2])
                teacher_logits.append(t_logits)
                teacher_embeddings.append(t_features[-1])

        agent_state = get_agent_state(trans_student_features, teacher_embeddings, logits, teacher_logits, targets, criterion_div)

        agent_states.append(agent_state)

        high_state = agent_state
        kd_temp_n, teacher_importance = high_agent(high_state)
        kd_temp = kd_temp_n.float().detach().mean(dim=0)
        teacher_importance = teacher_importance.detach() # [batch_size, teacher_num]

        # high_states.append(high_state)
        # high_actions.append(kd_temp_n)

        with torch.no_grad():
            logits_actions, feature_actions, alpha, _ = agent(agent_state)
            # logits_actions, feature_actions, alpha, kd_temp_n = agent(agent_state)
        if epoch == 0:
            logits_actions = torch.ones_like(logits_actions).cuda(args.gpu)
            feature_actions = torch.ones_like(feature_actions).cuda(args.gpu)
            alpha = torch.ones(1).cuda(args.gpu)
            # kd_temp = torch.tensor(args.kd_T).cuda(args.gpu)
        logits_actions = logits_actions.detach() # batch_size x teacher_number
        feature_actions = feature_actions.detach()
        alpha = alpha.detach().mean()

        logits_actions = adjust_teacher_weights(logits_actions, teacher_importance)
        feature_actions = adjust_teacher_weights(feature_actions, teacher_importance)

        # logits_agent_actions.append(logits_actions)
        # feature_agent_actions.append(feature_actions)

        if args.rank == 0 and batch_idx % 10 == 0:
            args.logger.info('actions:{}'.format(str(logits_actions[0])))

        loss_cls = criterion_ce(logits, targets)
        
        loss_kd = torch.tensor(0.).cuda(args.gpu)
        for idx in range(len(teacher_models)):
            criterion_div_new = DistillKL(kd_temp[idx]).to(device) # criterion_div_new = DistillKL(args.kd_T).to(device)
            loss_kd = loss_kd + (logits_actions[:, idx] * criterion_div_new(logits, teacher_logits[idx].detach(), unreduce=True)).mean()
        loss_feat = torch.tensor(0.).cuda(args.gpu)
        
        if args.feat_kd == 'mse':
            feat_kd_func = FeatureMSELoss()
        elif args.feat_kd == 'kl':
            feat_kd_func = FeatureKLLoss(args.kd_T)

        for idx in range(len(teacher_models)):
            loss_feat = loss_feat +  (feature_actions[:, idx] * feat_kd_func(trans_student_features[idx], teacher_features[idx])).mean()
        loss_feat = args.feat_weight * loss_feat
        
        loss = loss_cls + alpha * loss_kd + loss_feat
        loss.backward()
        optimizer.step()

        sample_ce_loss = F.cross_entropy(logits, targets, reduction='none')
        sample_kd_loss = torch.tensor(0.).cuda(args.gpu)
        sample_feat_loss = torch.tensor(0.).cuda(args.gpu)
        for idx in range(len(teacher_models)):
            criterion_div_new = DistillKL(kd_temp[idx]).to(device) # criterion_div_new = DistillKL(args.kd_T).to(device)
            sample_kd_loss = sample_kd_loss + logits_actions[:, idx] * criterion_div_new(logits, teacher_logits[idx].detach(), unreduce=True)
            sample_feat_loss = sample_feat_loss + (feature_actions[:, idx] * feat_kd_func(trans_student_features[idx], teacher_features[idx]))
        reward = -(sample_ce_loss + sample_kd_loss+ args.feat_weight * sample_feat_loss)
        rewards_mean = reward.mean() 
        rewards_std = reward.std()
        normalized_reward = (reward - rewards_mean) / rewards_std
        normalized_reward = normalized_reward.detach()
        normalized_reward = torch.clamp(normalized_reward, min=0, max=1) 
        agent_rewards.append(normalized_reward)

        # high_rewards.append(normalized_reward)

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
        # if batch_idx % 100 == 0 and batch_idx != 0: 
        #     train_high(args, epoch, high_states, high_rewards, high_actions, high_agent, high_agent_optimizer)
        #     high_states = []
        #     high_actions = []
        #     high_rewards = []
        if batch_idx % args.agent_step == 0 and batch_idx != 0: 
            train_high(args, epoch, agent_states, agent_rewards, high_agent, high_agent_optimizer)
            # train_high(args, epoch, agent_states, agent_rewards, high_actions, high_agent, high_agent_optimizer)
            train_agent(args, epoch, agent_states, agent_rewards, agent, agent_optimizer)
            # train_agent(args, epoch, agent_states, agent_rewards, logits_agent_actions, agent, agent_optimizer)
            agent_states = []
            # logits_agent_actions = []
            # feature_agent_actions = []
            agent_rewards = []
            # high_actions = []
            torch.cuda.empty_cache()

        
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

    if len(agent_states) != 0:
        train_high(args, epoch, agent_states, agent_rewards, high_agent, high_agent_optimizer)
        train_agent(args, epoch, agent_states, agent_rewards, agent, agent_optimizer)
        
