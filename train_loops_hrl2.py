# HRL的train函数相关
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
    trans_student_embeddings = [] #学生的特征向量经处理后得到的嵌入向量.
    for idx in range(len(trans_student_features)): 
        #trans_student_features是已经经过FeatTran(这个网络将student_feature的维度转化为teacher_feature的维度,笔记中有介绍)后的student_feature.
        #它是一个列表(每一项对应一个teacher),每一项的形状为(batch_size, channels, t_H, t_H)。
        trans_student_embedding = F.adaptive_avg_pool2d(trans_student_features[idx], (1,1)) 
            #全局平均池化,用于将特征图的空间信息压缩为单个值.从而提取全局特征.
            #将每个特征图的空间维度(t_H, t_H)压缩为(1, 1).结果的形状变为(batch_size, channels, 1, 1).
        trans_student_embedding = trans_student_embedding.view(trans_student_embedding.size(0), -1)
            #将池化后的特征张量展平,形状从(batch_size, channels, 1, 1)转换为(batch_size, channels).将每个样本的特征表示为一个长为channels的一维向量.
        trans_student_embeddings.append(trans_student_embedding)

    teacher_infos = []
    t_ces = []
    t_s_feat_div = []
    t_s_logit_div = []
    for idx in range(len(teacher_embeddings)): #这里的idx代表不同的teacher        
        feat_cos_sim = F.cosine_similarity(trans_student_embeddings[idx], teacher_embeddings[idx]).unsqueeze(-1) #论文中的(4)Teacher-student feature similarity.
            #cosine_similarity的两个参数形状为(batch_size,channels),输出形状为(batch_size)，unsquezze后为(batch_size,1)
        t_s_feat_div.append(feat_cos_sim)
        logit_kl = criterion_div(logits, teacher_logits[idx], unreduce=True).unsqueeze(-1) #论文中的(5)Teacher-student probability KL-divergence.
            #criterion_div实际上使用了DistillKL函数."unreduce=True"表示不对损失进行归约操作.
            #logits形状为(batch_size, num_classes),函数输出形状为(batch_size),unsqueeze后为(batch_size,1)
        t_s_logit_div.append(logit_kl)
        teachers_ce = F.cross_entropy(teacher_logits[idx], targets, reduction='none').unsqueeze(-1) #论文中的(3)Teacher cross-entropy loss.
            #teacher_logits[idx]形状为(batch_size, num_classes)，targets形状为(batch_size).
            #cross_entropy输出为(batch_size)，unsqueeze后为(batch_size, 1)
        t_ces.append(teachers_ce)
        teacher_info = torch.cat([feat_cos_sim, logit_kl, teachers_ce, teacher_embeddings[idx], teacher_logits[idx]], dim=1).detach()
            #除了上面提到的三个,teacher_embeddings[idx]的形状是(batch_size, embedding_dim),teacher_logits[idx]的形状是(batch_size, num_classes).
            #因此拼接后形状是(batch_size, 3 + embedding_dim + num_classes)
        teacher_infos.append(teacher_info)
    t_ces = torch.cat(t_ces, dim=1).detach() #形状为(batch_size, num_teachers).后面几个一样.
    t_s_logit_div = torch.cat(t_s_logit_div, dim=1).detach()
    t_s_feat_div = torch.cat(t_s_feat_div, dim=1).detach()
    return teacher_infos, t_ces, t_s_logit_div, t_s_feat_div

# 高层策略的训练
# 仿照了train_loops.py中的train_agent().    
# agent_state,agent_reward分别使用了train_hrl()中的actor_state和actor_rewards.(暂时先复用,有需要再改)
# agent_actions就是agent的输出kd_temp和teacher_importance.  
# agent和agent_optimizer实际就是high_agent以及它的optimizer.
def train_high(args, epoch, agent_state, agent_rewards, agent, agent_optimizer):
    agent.train()
    agent_loss = AverageMeter('agent_loss', ':.4e')
    for state, rewards in zip(agent_state, agent_rewards):
        
        kd_temp, teacher_importance = agent(state) # 两个输出形状均为[batch_size, teacher_num]
        agent_optimizer.zero_grad() 

        kd_temp_normalized = (kd_temp - 1) / 4  # kd_temp原先是[1, 5]的值范围(见HighAgent类), 现在映射到[0, 1]以符合binary_cross_entropy的要求.
        
        action_label = torch.ones_like(kd_temp).detach() * 0.99 # 平滑目标值, 保持一定梯度, 防止过拟合
        loss_kd_temp = F.binary_cross_entropy(kd_temp_normalized, action_label, weight=rewards.unsqueeze(-1))
        loss_kd_temp += 0.01 * torch.mean(kd_temp ** 2) # 在loss中新增L2正则化项(平方取平均), 防止过拟合/极端化

        # 新增 teacher_importance 相关
            # 由于 kd_temp 和 teacher_importance 形状一致，因此可以复用上面的 action_label.
        action_label2 = torch.ones_like(teacher_importance).detach()
        loss_teacher_importance = F.binary_cross_entropy(teacher_importance, action_label2, weight=rewards.unsqueeze(-1))
        # loss_teacher_importance += 0.01 * torch.mean(teacher_importance ** 2)  # L2 正则化

        loss = loss_kd_temp + loss_teacher_importance
        loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.parameters(), max_norm=2.0) # 新增梯度裁剪,防止梯度爆炸
        agent_optimizer.step()

        agent_loss.update(loss.item(), rewards.size(0))

    if args.rank == 0:
        args.logger.info('Epoch:{}, high agent Loss:{:.6f}'.format(epoch, agent_loss.avg))


# 低层策略的训练
# 复用了train_loops.py中的train_agent().    
#def train_agent(args, epoch, agent_state, agent_rewards, logits_agent_actions, agent, agent_optimizer):
def train_agent(args, epoch, agent_state, agent_rewards, agent, agent_optimizer):
    agent.train()
    agent_loss = AverageMeter('agent_loss', ':.4e')
    #下面这行的zip操作具体看代码笔记
    # for state, rewards, actions in zip(agent_state, agent_rewards, logits_agent_actions):
    for state, rewards in zip(agent_state, agent_rewards):
        
        agent_pred = agent(state)
        agent_optimizer.zero_grad() 
        
        action_label = torch.ones_like(agent_pred[0]).detach() 
        loss_logits = F.binary_cross_entropy(agent_pred[0], action_label, weight=rewards.unsqueeze(-1))
        loss_feature = F.binary_cross_entropy(agent_pred[1], action_label, weight=rewards.unsqueeze(-1))
        loss = loss_feature + loss_logits
        loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.parameters(), max_norm=2.0) # 新增梯度裁剪,防止梯度爆炸
        agent_optimizer.step()

        # agent_loss.update(loss.item(), actions.size(0))
        agent_loss.update(loss.item(), rewards.size(0))

    if args.rank == 0:
        args.logger.info('Epoch:{}, agent Loss:{:.6f}'.format(epoch, agent_loss.avg))

# New: 用于对teacher_importance最小的teacher的权重置零的函数
def adjust_teacher_weights(weights, mask_tensor):
    """
    weights: (batch_size, teacher_num) - 原始教师权重, 每行和为1
    mask_tensor: (batch_size, teacher_num) - 用于确定要置零位置的 teacher_importance 张量
    return: 调整后的权重张量, 形状仍为(batch_size, teacher_num), 满足每行最小位置置零, 其他位置按比例放大, 每行和仍为1
    """
    batch_size, teacher_num = weights.shape
    
    # 1. 找到每行最小值的索引
    # mask_tensor每行最小的值对应的位置就是要置零的位置
    _, min_indices = torch.min(mask_tensor, dim=1, keepdim=True)  # (batch_size, 1)
    
    # 2. 创建掩码，将最小位置置为0，其他位置为1
    mask = torch.ones_like(weights, dtype=torch.bool)
    # 使用scatter_将最小位置置为False
    mask.scatter_(1, min_indices, False)
    
    # 3. 将原始权重中对应位置置零
    adjusted_weights = weights * mask.float()
    
    # 4. 重新归一化，使得每行和为1
    # 计算每行非零位置的和
    row_sums = adjusted_weights.sum(dim=1, keepdim=True)
    
    # 归一化
    adjusted_weights = adjusted_weights / row_sums
    
    return adjusted_weights 

def train_hrl(train_loader, model, criterion_list, optimizer, epoch, device, 
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

    model.train() #开始时先让model(学生模型)处于训练状态(参数可变)，而agent则属于评估状态(参数冻结)
    agent.eval()
    agent_states = [] #对每个batch的agent网络的输入(agent_state)作记录
    # logits_agent_actions = [] #对每个batch的agent网络输出的各教师在logit kd loss中的权重作记录
    # feature_agent_actions = [] #对每个batch的agent网络输出的各教师在feature kd loss中的权重作记录
    agent_rewards = []

    # # 新增高层的states,actions,rewards记录(由于高层策略与低层策略异步更新)
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
        #agent_state是一个tuple: (teacher_infos, t_ces, t_s_logit_div, t_s_feat_div)
        #agent_state其实是一个batch_size的整体state! 这个tuple的每一项都是(batch_size, ...)的形状.

        agent_states.append(agent_state)

        # 先用高层策略输出蒸馏温度kd_temp(因此原先低层输出的kd_temp就没用了)
        high_state = agent_state # 高层的输入状态复用了actor_state
        kd_temp_n, teacher_importance = high_agent(high_state) # 二者形状均为[batch_size, teacher_num]
        kd_temp = kd_temp_n.float().detach().mean(dim=0) #对batch_size取平均，变成[teacher_num]形状
        teacher_importance = teacher_importance.detach() # [batch_size, teacher_num]

        # high_states.append(high_state)
        # high_actions.append(kd_temp_n)

        with torch.no_grad():
            logits_actions, feature_actions, alpha, _ = agent(agent_state)
            # logits_actions, feature_actions, alpha, kd_temp_n = agent(agent_state)
        if epoch == 0:
            logits_actions = torch.ones_like(logits_actions).cuda(args.gpu) #由论文algorithm 1,第一个循环采用全1作为weight.
            feature_actions = torch.ones_like(feature_actions).cuda(args.gpu)
            alpha = torch.ones(1).cuda(args.gpu)
            # kd_temp = torch.tensor(args.kd_T).cuda(args.gpu)
        logits_actions = logits_actions.detach() # batch_size x teacher_number
        feature_actions = feature_actions.detach()
        alpha = alpha.detach().mean() #先detach再取平均,让alpha变为标量scalar
        # kd_temp = kd_temp.float().detach().mean()  #将kd_temperature转为标量
        # kd_temp = kd_temp_n.float().detach().mean(dim=0) #对batch_size取平均，变成[teacher_num]形状

        # 对一个batch中的每个input，将teacher_importance最小的teacher的权重置为0
        logits_actions = adjust_teacher_weights(logits_actions, teacher_importance)
        feature_actions = adjust_teacher_weights(feature_actions, teacher_importance)

        # logits_agent_actions.append(logits_actions)
        # feature_agent_actions.append(feature_actions)

        if args.rank == 0 and batch_idx % 10 == 0:
            #print('actions:{}'.format(str(actions)))
            args.logger.info('actions:{}'.format(str(logits_actions[0]))) #注: txt文件中的大部分输出其实就是这个！
            #args.logger.info('actions:{}'.format(str(logits_actions.max().item())+str(logits_actions.argmax(dim=1))))

        loss_cls = criterion_ce(logits, targets)
        
        loss_kd = torch.tensor(0.).cuda(args.gpu)
        for idx in range(len(teacher_models)):
            criterion_div_new = DistillKL(kd_temp[idx]).to(device) # criterion_div_new = DistillKL(args.kd_T).to(device)
            loss_kd = loss_kd + (logits_actions[:, idx] * criterion_div_new(logits, teacher_logits[idx].detach(), unreduce=True)).mean()
            #logits_actions是logit kd loss的各教师权重,形状为(batch_size, teacher_number),
            #因此logits_actions[:, idx]就是把此idx对应教师的所有权重都取出来,形状是(batch_size).
            #而后面的criterion_div()中的logits形状是(batch_size, num_classes),因此输出形状也是(batch_size).
            #因此可以相乘,之后取了平均就是一个数了.
        loss_feat = torch.tensor(0.).cuda(args.gpu)
        
        if args.feat_kd == 'mse':
            feat_kd_func = FeatureMSELoss() #均方误差
        elif args.feat_kd == 'kl':
            feat_kd_func = FeatureKLLoss(args.kd_T) #KL散度

        for idx in range(len(teacher_models)):
            loss_feat = loss_feat +  (feature_actions[:, idx] * feat_kd_func(trans_student_features[idx], teacher_features[idx])).mean()
            #类似于loss_kd的计算.
        loss_feat = args.feat_weight * loss_feat
        
        #loss = loss_cls + loss_kd + loss_feat #即MTKD的总loss,用它来更新student网络.
        loss = loss_cls + alpha * loss_kd + loss_feat #新增了参数α
        loss.backward()
        optimizer.step()

        #下面这部分是在计算Reward,采用论文中的Eq(6)
        sample_ce_loss = F.cross_entropy(logits, targets, reduction='none') #Eq(6)的第一项
        sample_kd_loss = torch.tensor(0.).cuda(args.gpu) #第二项
        sample_feat_loss = torch.tensor(0.).cuda(args.gpu) #第三项
        for idx in range(len(teacher_models)):
            criterion_div_new = DistillKL(kd_temp[idx]).to(device) # criterion_div_new = DistillKL(args.kd_T).to(device)
            sample_kd_loss = sample_kd_loss + logits_actions[:, idx] * criterion_div_new(logits, teacher_logits[idx].detach(), unreduce=True)
            sample_feat_loss = sample_feat_loss + (feature_actions[:, idx] * feat_kd_func(trans_student_features[idx], teacher_features[idx]))
        reward = -(sample_ce_loss + sample_kd_loss+ args.feat_weight * sample_feat_loss)
        rewards_mean = reward.mean() 
        rewards_std = reward.std() #标准差
        normalized_reward = (reward - rewards_mean) / rewards_std #标准化:将奖励值调整为均值为0,标准差为1的分布.
        normalized_reward = normalized_reward.detach()
        normalized_reward = torch.clamp(normalized_reward, min=0, max=1) 
            #将normalized_reward中的值限制在[0,1]的范围内.原来小于min则设置为min,原来大于max则设为max.
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
        # # 高层策略更频繁地更新
        # if batch_idx % 100 == 0 and batch_idx != 0: 
        #     train_high(args, epoch, high_states, high_rewards, high_actions, high_agent, high_agent_optimizer)
        #     high_states = []
        #     high_actions = []
        #     high_rewards = []
        if batch_idx % args.agent_step == 0 and batch_idx != 0: 
            # 由于agent_step默认是1000,但batch最多就782个,因此这部分代码不会被执行,训练函数只会在代码末尾结束此epoch时被调用(针对CIFAR)
            # 高层策略的训练
            train_high(args, epoch, agent_states, agent_rewards, high_agent, high_agent_optimizer)
            # train_high(args, epoch, agent_states, agent_rewards, high_actions, high_agent, high_agent_optimizer)
            # 低层策略的训练
            train_agent(args, epoch, agent_states, agent_rewards, agent, agent_optimizer)
            # train_agent(args, epoch, agent_states, agent_rewards, logits_agent_actions, agent, agent_optimizer)
            agent_states = []
            # logits_agent_actions = []
            # feature_agent_actions = []
            agent_rewards = []
            # high_actions = []
            torch.cuda.empty_cache()  # 清理显存(释放已被释放但仍占用显存的缓存内存)

        
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
        # 高层策略的训练
        train_high(args, epoch, agent_states, agent_rewards, high_agent, high_agent_optimizer)
        # 低层策略的训练
        train_agent(args, epoch, agent_states, agent_rewards, agent, agent_optimizer)
        

