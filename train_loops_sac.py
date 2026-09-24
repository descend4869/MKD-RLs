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
import csv
import shutil
import argparse
import numpy as np
from distiller_zoo import FeatureKLLoss, FeatureMSELoss
from models.SAC import ReplayBuffer, SACActor, SACCritic


def get_actor_state(trans_student_features, teacher_embeddings, logits, teacher_logits, targets, criterion_div):
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



def train_rl(args, epoch, replay_buffer, actor, actor_optimizer,
            critic_1, critic_1_optimizer, critic_2, critic_2_optimizer, target_critic_1, target_critic_2, 
            H_alpha, alpha_optimizer, H0=None ,gamma=0.99, tau=0.005, target_update_freq=5):
    teacher_num = args.teacher_num
    states, actions, rewards, next_states = replay_buffer.sample(args.batch_size)

    def unpack_state(state):
        t_s_feat_div = state[:, -teacher_num:]
        t_s_logit_div = state[:, -2 * teacher_num:-teacher_num]
        t_ces = state[:, -3 * teacher_num:-2 * teacher_num]
        teacher_infos_flat = state[:, :-3 * teacher_num]
        teacher_info_dim = teacher_infos_flat.shape[1] // teacher_num
        teacher_infos = torch.split(teacher_infos_flat, teacher_info_dim, dim=1)
        return teacher_infos, t_ces, t_s_logit_div, t_s_feat_div
    
    states = unpack_state(states)
    next_states = unpack_state(next_states)

    with torch.no_grad():
        _, _, _, _, next_action_mean, next_action_log_std = actor(next_states)
        #_, _, next_action_mean, next_action_log_std = actor(next_states)
        next_action_std = next_action_log_std.exp()
        next_action_normal = torch.distributions.Normal(next_action_mean, next_action_std)
        next_eps = next_action_normal.rsample()
        next_action = torch.tanh(next_eps)

        softmax = nn.Softmax(dim=1)
        next_logits_actions = softmax(next_action[:, :teacher_num])
        next_feature_actions = softmax(next_action[:, teacher_num:])

        target_q1 = target_critic_1(next_states, next_logits_actions, next_feature_actions)
        target_q2 = target_critic_2(next_states, next_logits_actions, next_feature_actions) # target_q2 = target_q1 
        
        target_q = torch.min(target_q1, target_q2)
        # next_log_prob = next_action_normal.log_prob(next_eps)
        # next_log_prob -= torch.log(1 - next_action.pow(2) + 1e-6)
        # next_log_prob = next_log_prob.sum(dim=1, keepdim=True)
        # target_q = torch.min(target_q1, target_q2) - H_alpha * next_log_prob

        target_q = rewards + gamma * target_q # Bellman backup  
    current_q1 = critic_1(states, actions[:, :teacher_num], actions[:, teacher_num:])
    current_q2 = critic_2(states, actions[:, :teacher_num], actions[:, teacher_num:])
    critic_loss_1 = F.mse_loss(current_q1, target_q)
    critic_loss_2 = F.mse_loss(current_q2, target_q)
    critic_1_optimizer.zero_grad(); critic_loss_1.backward(); critic_1_optimizer.step()
    critic_2_optimizer.zero_grad(); critic_loss_2.backward(); critic_2_optimizer.step()


    _, _, _, _, action_mean, action_log_std = actor(states)
    #_, _, action_mean, action_log_std = actor(states)
    action_std = action_log_std.exp()
    action_normal = torch.distributions.Normal(action_mean, action_std)
    eps = action_normal.rsample()
    action = torch.tanh(eps) # [batch_size, teacher_num * 2]
    softmax = nn.Softmax(dim=1)
    logits_actions = softmax(action[:, :teacher_num]) 
    feature_actions = softmax(action[:, teacher_num:])
    q1 = critic_1(states, logits_actions, feature_actions)
    q2 = critic_2(states, logits_actions, feature_actions) # q2 = q1
    log_prob = action_normal.log_prob(eps)
    log_prob -= torch.log(1 - action.pow(2) + 1e-6)
    log_prob = log_prob.sum(dim=1, keepdim=True)
    actor_loss = (H_alpha * log_prob - torch.min(q1, q2)).mean()
    actor_optimizer.zero_grad(); actor_loss.backward(); 
    torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=1.0)
    actor_optimizer.step()


    if not hasattr(args, 'H0'):
        args.H0 = -2 * teacher_num 
    # args.H0 = (1 - 0.01) * args.H0 + 0.01 * (-log_prob.mean().item())
    H0 = args.H0
    alpha_loss = (-H_alpha * (log_prob + H0).detach()).mean()
    alpha_optimizer.zero_grad(); alpha_loss.backward(); alpha_optimizer.step(); 
    H_alpha.data.clamp_(min=1e-6)
    if args.rank == 0:
        args.logger.info(f"Epoch: {epoch}, H0: {H0:.4f}, Log_Prob: {log_prob.mean().item():.4f}, H_alpha: {H_alpha.item():.4f}")


    if not hasattr(args, "target_updated_epoch"):
        args.target_updated_epoch = 0
    if epoch % target_update_freq == 0 and epoch != args.target_updated_epoch:
        args.target_updated_epoch = epoch
        with torch.no_grad():
            for param, target_param in zip(critic_1.parameters(), target_critic_1.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
            for param, target_param in zip(critic_2.parameters(), target_critic_2.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
    
    if args.rank == 0:
        args.logger.info(f"Epoch: {epoch}, Critic Loss: {critic_loss_1.item():.4f}, Actor Loss: {actor_loss.item():.4f}")
        
def train_sac(train_loader, model, criterion_list, optimizer, epoch, device, 
          args, actor, feat_trans, teacher_models, actor_optimizer,
          critic_1, critic_1_optimizer, critic_2, critic_2_optimizer, target_critic_1, target_critic_2, 
          replay_buffer: ReplayBuffer, H_alpha, alpha_optimizer):
    
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
    actor.eval()
    # critic_1.eval()
    # critic_2.eval()
    # actor_rewards = []

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        torch.cuda.empty_cache()
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
                torch.cuda.empty_cache()
                t_features, t_logits = t_model(inputs, is_feat=True)
                t_feature = t_features[-1]
                t_feature = t_feature.detach() 
                t_logits = t_logits.detach() 
                
                teacher_features.append(t_features[-2])
                teacher_logits.append(t_logits)
                teacher_embeddings.append(t_features[-1])

        actor_state = get_actor_state(trans_student_features, teacher_embeddings, logits, teacher_logits, targets, criterion_div)


        with torch.no_grad():
            logits_actions, feature_actions, alpha, kd_temp, action_mean, action_log_std = actor(actor_state)
        if epoch == 0:
            logits_actions = torch.ones_like(logits_actions).cuda(args.gpu)
            feature_actions = torch.ones_like(feature_actions).cuda(args.gpu)
            alpha = torch.ones(1).cuda(args.gpu)
            kd_temp = torch.tensor(args.kd_T).cuda(args.gpu)
        logits_actions = logits_actions.detach() # batch_size x teacher_number
        feature_actions = feature_actions.detach()

        alpha = alpha.detach().mean()
        kd_temp = kd_temp.float().detach().mean()

        criterion_list[1].T = kd_temp
        criterion_div = criterion_list[1]


        if args.rank == 0 and batch_idx % 10 == 0:
            args.logger.info('actions:{}'.format(str(logits_actions[0])))

        loss_cls = criterion_ce(logits, targets)
        
        loss_kd = torch.tensor(0.).cuda(args.gpu)
        for idx in range(len(teacher_models)):
            loss_kd = loss_kd + (logits_actions[:, idx] * criterion_div(logits, teacher_logits[idx].detach(), unreduce=True)).mean()
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
            sample_kd_loss = sample_kd_loss + logits_actions[:, idx] * criterion_div(logits, teacher_logits[idx].detach(), unreduce=True)
            sample_feat_loss = sample_feat_loss + (feature_actions[:, idx] * feat_kd_func(trans_student_features[idx], teacher_features[idx]))
        reward = -(sample_ce_loss + sample_kd_loss+ args.feat_weight * sample_feat_loss)
        rewards_mean = reward.mean() 
        rewards_std = reward.std()
        normalized_reward = (reward - rewards_mean) / rewards_std
        normalized_reward = normalized_reward.detach()
        normalized_reward = torch.clamp(normalized_reward, min=0, max=1) 
        # actor_rewards.append(normalized_reward)


        state = torch.cat([torch.cat(actor_state[0], dim=1),
                   actor_state[1],
                   actor_state[2],  # t_s_logit_div
                   actor_state[3]], dim=1).detach()  # t_s_feat_div
        action = torch.cat([logits_actions, feature_actions], dim=1).detach()
        new_features, new_logits = model(inputs, is_feat=True)
        new_trans_student_features = feat_trans(new_features[-2])
        next_actor_state = get_actor_state(new_trans_student_features, teacher_embeddings, new_logits, teacher_logits, targets, criterion_div)
        next_state = torch.cat([torch.cat(next_actor_state[0], dim=1),  
                   next_actor_state[1],
                   next_actor_state[2],
                   next_actor_state[3]], dim=1).detach()
        replay_buffer.add(state, action, normalized_reward, next_state)
        del state, action, next_state
        del new_features, new_logits, new_trans_student_features, next_actor_state
        torch.cuda.empty_cache() 


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
        if batch_idx % args.agent_step == 0 and batch_idx != 0:
            train_rl(args, epoch, replay_buffer, actor, actor_optimizer,
                     critic_1, critic_1_optimizer, critic_2, critic_2_optimizer, target_critic_1, target_critic_2,
                     H_alpha, alpha_optimizer)
            # actor_rewards = []
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

    train_rl(args, epoch, replay_buffer, actor, actor_optimizer,
                critic_1, critic_1_optimizer, critic_2, critic_2_optimizer, target_critic_1, target_critic_2,
                H_alpha, alpha_optimizer)

    # replay_buffer.clear()
        
