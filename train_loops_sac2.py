# SAC的train函数相关(第2版)
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
from models.SAC import ReplayBuffer, SACActor, SACCritic


# 这个函数与train_loops中的get_agent_state一致
def get_actor_state(trans_student_features, teacher_embeddings, logits, teacher_logits, targets, criterion_div):
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


def train_rl(args, epoch, replay_buffer, actor, actor_optimizer,
            critic_1, critic_1_optimizer, critic_2, critic_2_optimizer, target_critic_1, target_critic_2, 
            # H_alpha是熵正则化系数, gamma是折扣因子, tau是soft update的参数, target_update_freq是软更新目标网络的频率
            H_alpha, alpha_optimizer, H0=None ,gamma=0.99, tau=0.005, target_update_freq=10):
    teacher_num = args.teacher_num
    # 从replay_buffer中取样
    states, actions, rewards, next_states = replay_buffer.sample(args.batch_size)

    # 将states和next_states还原为元组的函数(在replaybuffer的add部分,见train_sac函数,它们被转化为张量存储)
    def unpack_state(state):
        t_s_feat_div = state[:, -teacher_num:]  # 提取 t_s_feat_div
        t_s_logit_div = state[:, -2 * teacher_num:-teacher_num]  # 提取 t_s_logit_div
        t_ces = state[:, -3 * teacher_num:-2 * teacher_num]  # 提取 t_ces
        teacher_infos_flat = state[:, :-3 * teacher_num]  # 剩余部分是 teacher_infos，形状为(batch_size, teacher_num * (3 + embedding_dim + num_classes) )
        assert teacher_infos_flat.shape[1] % teacher_num == 0 , ("Not Divisible!") #作个断言
        teacher_info_dim = teacher_infos_flat.shape[1] // teacher_num  # 每个 teacher_info 的维度
        teacher_infos = torch.split(teacher_infos_flat, teacher_info_dim, dim=1)  # 平均分成 teacher_num 份
        return teacher_infos, t_ces, t_s_logit_div, t_s_feat_div
    
    # 还原 states 和 next_states 为四元tuple
    states = unpack_state(states)
    next_states = unpack_state(next_states)

    # 更新critic网络
    # 目标是学习Q函数,满足Bellman方程 Q(s, a) = reward + gamma * Q(s', a'). 其中s'和a'是下一个state和action.
    # 又由于critic参数实时更新并不稳定,因此改成 Q(s, a) = reward + gamma * Q_target(s', a').换用延迟更新的目标网络提高训练稳定性.
    ## 上面的公式存在一点问题，详见《SAC优化》第3条
    with torch.no_grad():
        # 计算target Q值
        _, _, _, _, next_action_mean, next_action_log_std = actor(next_states)
        #_, _, next_action_mean, next_action_log_std = actor(next_states)
        next_action_std = next_action_log_std.exp() # 和上面的两个输出形状一样 [batch_size, teacher_num * 2]
        next_action_normal = torch.distributions.Normal(next_action_mean, next_action_std) # 形状不变
        next_eps = next_action_normal.rsample() # rsample()是一种采样方法,它支持反向传播(通过重参数化技巧); 形状不变
        next_action = torch.tanh(next_eps) # tanh()变换将最终的action限制在[-1, 1]范围内. 形状不变

        # 将next_action分为 logits_actions 和 feature_actions, 并用softmax归一化
        softmax = nn.Softmax(dim=1)
        next_logits_actions = softmax(next_action[:, :teacher_num])  # 前 teacher_num 个维度
        next_feature_actions = softmax(next_action[:, teacher_num:])  # 后 teacher_num 个维度

        target_q1 = target_critic_1(next_states, next_logits_actions, next_feature_actions)
        target_q2 = target_critic_2(next_states, next_logits_actions, next_feature_actions)
        
        target_q = torch.min(target_q1, target_q2)  # 二者取小,更稳定
        # # 或许将上面这一行换成下面这些注释才是正确版本(这部分完全模仿本函数的"更新Actor网络"部分，解释也在那里)
        # next_log_prob = next_action_normal.log_prob(next_eps)
        # next_log_prob -= torch.log(1 - next_action.pow(2) + 1e-6)
        # next_log_prob = next_log_prob.sum(dim=1, keepdim=True)
        # target_q = torch.min(target_q1, target_q2) - H_alpha * next_log_prob

        target_q = rewards + gamma * target_q # Bellman backup  
    # 计算当前Q值
    current_q1 = critic_1(states, actions[:, :teacher_num], actions[:, teacher_num:])
    current_q2 = critic_2(states, actions[:, :teacher_num], actions[:, teacher_num:])
    # 计算loss并优化
    critic_loss_1 = F.mse_loss(current_q1, target_q) #差作平方再取平均
    critic_loss_2 = F.mse_loss(current_q2, target_q)
    critic_1_optimizer.zero_grad(); critic_loss_1.backward(); critic_1_optimizer.step() #分号实现一行多语句
    critic_2_optimizer.zero_grad(); critic_loss_2.backward(); critic_2_optimizer.step()


    # 更新Actor网络
    # loss = alpha * log π(a|s) - Q(s, a). 化简过程见笔记中的链接.
    _, _, _, _, action_mean, action_log_std = actor(states)
    #_, _, action_mean, action_log_std = actor(states)
    action_std = action_log_std.exp()
    action_normal = torch.distributions.Normal(action_mean, action_std)
    eps = action_normal.rsample() # 重参数化采样原始动作eps，还未经tanh()限制在[-1, 1]内
    action = torch.tanh(eps) # [batch_size, teacher_num * 2]
    # action拆成两部分
    softmax = nn.Softmax(dim=1)
    logits_actions = softmax(action[:, :teacher_num]) 
    feature_actions = softmax(action[:, teacher_num:])
    # 计算Q值
    q1 = critic_1(states, logits_actions, feature_actions)
    q2 = critic_2(states, logits_actions, feature_actions)
    # 计算 log π(a|s)
    # 有关tanh变换带来的修正可以参考笔记中的链接
    log_prob = action_normal.log_prob(eps) # [batch_size, action_dim], 其中action_dim = 2 * teacher_num.
        # 我们目标是要计算action(已经过tanh()变换)在正态分布下的对数概率，但我们不能直接用action_normal.log_prob(action)
        # 因为action在经过tanh()缩放后已经不服从高斯分布了，只有eps还服从高斯分布，
        # 因此只能采用action_normal.log_prob(eps)再补充修正项来间接获得action的对数概率.
    log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        # 修正 tanh 的变换.这个公式怎么来的可以看笔记中的链接1的推导.
        # 其中 + 1e-6 是为了避免数值稳定性问题(防止分母为零).
    log_prob = log_prob.sum(dim=1, keepdim=True)  # 形状变为[batch_size, 1], 即每个样本的总对数概率.
    # 计算loss并优化
    actor_loss = (H_alpha * log_prob - torch.min(q1, q2)).mean() # [batch_size, 1]对整个批次的样本取平均, .mean()在无参数时得到一个标量.
    actor_optimizer.zero_grad(); actor_loss.backward(); 
    torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=1.0) # 梯度裁剪,用在backward()之后step()之前.限制梯度的最大范数防止梯度爆炸
    actor_optimizer.step()
    if args.rank == 0:
        args.logger.info(f"Epoch: {epoch}, Q value: {torch.min(q1, q2).mean().item():.4f}")

    # 更新熵正则化系数H_alpha
    # loss = -α log π(a|s) - α H0. 其中H0是一个常数目标熵值. 可见SAC算法论文.
    # 先动态调整目标熵H0，使其不断接近真实策略熵
    if not hasattr(args, 'H0'):
        # args.H0 = -2 * teacher_num 
        args.H0 = 2 * teacher_num 
        # 在《SAC算法与应用》论文的附录D中提到，H0一般默认为 -action_dim.
        # 然而log_prob是概率取log，一定是负的，因此 当前策略熵 = -log_prob一定是正的.
        # H_alpha的更新逻辑是比较H0与当前策略熵的大小，如果H0为负，那alpha_loss就一定为正（无论log_prob怎么变化），alpha只会永远减小
        # 这显然是不合理的，因此我尝试将H0设置成正值的 action_dim.
    args.H0 = (1 - 0.01) * args.H0 + 0.01 * (-log_prob.mean().item())
    H0 = args.H0
    # 再更新H_alpha
    alpha_loss = (-H_alpha * (log_prob + H0).detach()).mean()
    alpha_optimizer.zero_grad(); alpha_loss.backward(); alpha_optimizer.step(); 
    H_alpha.data.clamp_(min=1e-6) # 限制 H_alpha 的最小值，使其不小于0
    if args.rank == 0:
        args.logger.info(f"Epoch: {epoch}, H0: {H0:.4f}, Log_Prob: {log_prob.mean().item():.4f}, H_alpha: {H_alpha.item():.4f}")


    # Target网络的软更新(soft update): 并非每个周期都更新, 保持一定的稳定性
    # if not hasattr(args, "target_updated_epoch"):
    #     args.target_updated_epoch = 0 # 如果一个epoch内会有多次调用train_rl()时，确保单个epoch最多更新一回目标网络
    if epoch % target_update_freq == 0: # and epoch != args.target_updated_epoch:
        # args.target_updated_epoch = epoch
        with torch.no_grad():
            for param, target_param in zip(critic_1.parameters(), target_critic_1.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
            for param, target_param in zip(critic_2.parameters(), target_critic_2.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
    
    if args.rank == 0:
        args.logger.info(f"Epoch: {epoch}, Critic Loss: {critic_loss_1.item():.4f}, Actor Loss: {actor_loss.item():.4f}")


# 将上面train_rl()函数中的actor更新部分单独抽取出来，用于actor的更频繁更新
def update_actor(args, epoch, replay_buffer, actor, actor_optimizer, critic_1, critic_2, H_alpha):
    teacher_num = args.teacher_num
    # 从replay_buffer中取样
    states, _, _, _ = replay_buffer.sample(args.batch_size)

    # 将states和next_states还原为元组的函数(在replaybuffer的add部分,见train_sac函数,它们被转化为张量存储)
    def unpack_state(state):
        t_s_feat_div = state[:, -teacher_num:]  # 提取 t_s_feat_div
        t_s_logit_div = state[:, -2 * teacher_num:-teacher_num]  # 提取 t_s_logit_div
        t_ces = state[:, -3 * teacher_num:-2 * teacher_num]  # 提取 t_ces
        teacher_infos_flat = state[:, :-3 * teacher_num]  # 剩余部分是 teacher_infos，形状为(batch_size, teacher_num * (3 + embedding_dim + num_classes) )
        assert teacher_infos_flat.shape[1] % teacher_num == 0 , ("Not Divisible!") #作个断言
        teacher_info_dim = teacher_infos_flat.shape[1] // teacher_num  # 每个 teacher_info 的维度
        teacher_infos = torch.split(teacher_infos_flat, teacher_info_dim, dim=1)  # 平均分成 teacher_num 份
        return teacher_infos, t_ces, t_s_logit_div, t_s_feat_div
    
    # 还原 states 为四元tuple
    states = unpack_state(states)

    # 更新Actor网络
    # loss = alpha * log π(a|s) - Q(s, a). 化简过程见笔记中的链接.
    _, _, _, _, action_mean, action_log_std = actor(states)
    action_std = action_log_std.exp()
    action_normal = torch.distributions.Normal(action_mean, action_std)
    eps = action_normal.rsample() # 重参数化采样原始动作eps，还未经tanh()限制在[-1, 1]内
    action = torch.tanh(eps) # [batch_size, teacher_num * 2]
    # action拆成两部分
    softmax = nn.Softmax(dim=1)
    logits_actions = softmax(action[:, :teacher_num]) 
    feature_actions = softmax(action[:, teacher_num:])
    # 计算Q值
    q1 = critic_1(states, logits_actions, feature_actions)
    q2 = critic_2(states, logits_actions, feature_actions)
    # 计算 log π(a|s)
    # 有关tanh变换带来的修正可以参考笔记中的链接
    log_prob = action_normal.log_prob(eps) # [batch_size, action_dim], 其中action_dim = 2 * teacher_num.
        # 我们目标是要计算action(已经过tanh()变换)在正态分布下的对数概率，但我们不能直接用action_normal.log_prob(action)
        # 因为action在经过tanh()缩放后已经不服从高斯分布了，只有eps还服从高斯分布，
        # 因此只能采用action_normal.log_prob(eps)再补充修正项来间接获得action的对数概率.
    log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        # 修正 tanh 的变换.这个公式怎么来的可以看笔记中的链接1的推导.
        # 其中 + 1e-6 是为了避免数值稳定性问题(防止分母为零).
    log_prob = log_prob.sum(dim=1, keepdim=True)  # 形状变为[batch_size, 1], 即每个样本的总对数概率.
    # 计算loss并优化
    actor_loss = (H_alpha * log_prob - torch.min(q1, q2)).mean() # [batch_size, 1]对整个批次的样本取平均, .mean()在无参数时得到一个标量.
    actor_optimizer.zero_grad(); actor_loss.backward(); 
    torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=1.0) # 梯度裁剪,用在backward()之后step()之前.限制梯度的最大范数防止梯度爆炸
    actor_optimizer.step()

    if args.rank == 0:
        args.logger.info(f"Update Actor in Epoch: {epoch}, Actor Loss: {actor_loss.item():.4f}")


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

    model.train() #开始时先让model(学生模型)处于训练状态(参数可变)，而agent则属于评估状态(参数冻结)
    actor.eval()
    #critic_1.eval()
    #critic_2.eval()
    # actor_states = [] #对每个batch的actor网络的输入(actor_state)作记录
    # logits_actor_actions = [] #对每个batch的actor网络输出的各教师在logit kd loss中的权重作记录
    # feature_actor_actions = [] #对每个batch的actor网络输出的各教师在feature kd loss中的权重作记录
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
        #actor_state是一个tuple: (teacher_infos, t_ces, t_s_logit_div, t_s_feat_div)
        #actor_state其实是一个batch_size的整体state! 这个tuple的每一项都是(batch_size, ...)的形状.

        # actor_states.append(actor_state)

        with torch.no_grad():
            logits_actions, feature_actions, alpha, kd_temp, action_mean, action_log_std = actor(actor_state)
            #logits_actions, feature_actions, action_mean, action_log_std = actor(actor_state)
            #actor的最后两个输出是在更新RL框架(actor,critic等)时使用的,现在是在更新model,因此不会用到. 
        if epoch == 0:
            logits_actions = torch.ones_like(logits_actions).cuda(args.gpu) #由论文algorithm 1,第一个循环采用全1作为weight.
            feature_actions = torch.ones_like(feature_actions).cuda(args.gpu)
            alpha = torch.ones(1).cuda(args.gpu)
            kd_temp = torch.tensor(args.kd_T).cuda(args.gpu)
        logits_actions = logits_actions.detach() # batch_size x teacher_number
        feature_actions = feature_actions.detach()
        alpha = alpha.detach().mean() #先detach再取平均,让alpha变为标量scalar
        kd_temp = kd_temp.float().detach().mean()  #将kd_temperature转为标量

        #新加的,直接对criterion_div的蒸馏温度进行更新
        criterion_list[1].T = kd_temp #下次进入train()时,criterion_list[1]的T(即蒸馏温度)会保留上一个循环中更新的kd_temp值
        criterion_div = criterion_list[1]

        # logits_actor_actions.append(logits_actions)
        # feature_actor_actions.append(feature_actions)

        if args.rank == 0 and batch_idx % 10 == 0:
            #print('actions:{}'.format(str(actions)))
            args.logger.info('actions:{}'.format(str(logits_actions[0]))) #注: txt文件中的大部分输出其实就是这个！
            #args.logger.info('actions:{}'.format(str(logits_actions.max().item())+str(logits_actions.argmax(dim=1))))

        loss_cls = criterion_ce(logits, targets)
        
        loss_kd = torch.tensor(0.).cuda(args.gpu)
        for idx in range(len(teacher_models)):
            loss_kd = loss_kd + (logits_actions[:, idx] * criterion_div(logits, teacher_logits[idx].detach(), unreduce=True)).mean()
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
            sample_kd_loss = sample_kd_loss + logits_actions[:, idx] * criterion_div(logits, teacher_logits[idx].detach(), unreduce=True)
            sample_feat_loss = sample_feat_loss + (feature_actions[:, idx] * feat_kd_func(trans_student_features[idx], teacher_features[idx]))
        reward = -(sample_ce_loss + sample_kd_loss+ args.feat_weight * sample_feat_loss)
        rewards_mean = reward.mean() 
        rewards_std = reward.std() #标准差
        normalized_reward = (reward - rewards_mean) / rewards_std #标准化:将奖励值调整为均值为0,标准差为1的分布.
        normalized_reward = normalized_reward.detach()
        normalized_reward = torch.clamp(normalized_reward, min=0, max=1) 
            #将normalized_reward中的值限制在[0,1]的范围内.原来小于min则设置为min,原来大于max则设为max.
        # actor_rewards.append(normalized_reward)


        # ReplayBuffer的Add
        state = torch.cat([torch.cat(actor_state[0], dim=1),  # teacher_infos,一个长为num_teachers的列表,每一项形状为(batch_size, 3 + embedding_dim + num_classes),因此自己要先拼接
                   actor_state[1],  # t_ces,形状为(batch_size, num_teachers),后几个一样.
                   actor_state[2],  # t_s_logit_div
                   actor_state[3]], dim=1).detach()  # t_s_feat_div
        action = torch.cat([logits_actions, feature_actions], dim=1).detach() #动作是logits_actions和feature_actions的拼接
        #由于上面已经对model(学生模型)进行了更新,因此就能计算新state(即next_state)了
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
            #根据replay_buffer更新rl系统(包括actor,critic等)，然后进入下一个循环.
            train_rl(args, epoch, replay_buffer, actor, actor_optimizer,
                     critic_1, critic_1_optimizer, critic_2, critic_2_optimizer, target_critic_1, target_critic_2,
                     H_alpha, alpha_optimizer)
            # actor_states = []
            # logits_actor_actions = []
            # feature_actor_actions = []
            # actor_rewards = []
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
    # if len(actor_states) != 0:
    #     train_rl(args, epoch, replay_buffer, actor, actor_optimizer,
    #              critic_1, critic_1_optimizer, critic_2, critic_2_optimizer, target_critic_1, target_critic_2,
    #              H_alpha, alpha_optimizer)

    # 上面的actor_states因为没用被我删掉了，同时鉴于args.agent_step的设置，
    # " if len(actor_states) != 0: "这个条件基本总能满足，
    # 因此干脆去掉这个条件，每个epoch结尾必定做一次SAC网络更新就好了
    train_rl(args, epoch, replay_buffer, actor, actor_optimizer,
                critic_1, critic_1_optimizer, critic_2, critic_2_optimizer, target_critic_1, target_critic_2,
                H_alpha, alpha_optimizer)
        

