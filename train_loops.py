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


def get_actions(agent_pred):
    batch_size = agent_pred.size(0)
    teacher_num = agent_pred.size(1)
    index = torch.from_numpy(np.random.randint(0, teacher_num, batch_size).astype(np.int64)).cuda(args.gpu)
    random_select =  F.one_hot(index, num_classes=teacher_num).float().cuda(args.gpu)
    actions = torch.where(agent_pred>=0.5, torch.ones_like(agent_pred), torch.zeros_like(agent_pred)).cuda(args.gpu)
    is_random = (actions.sum(1) ==  0)[:, None].float().cuda(args.gpu)
    actions = actions + random_select * is_random
    return actions

'''
def train_agent(args, epoch, agent_state, agent_rewards, logits_agent_actions, agent, agent_optimizer):
    agent.train()
    agent_loss = AverageMeter('agent_loss', ':.4e')
    #下面这行的zip操作具体看代码笔记
    for state, rewards, actions in zip(agent_state, agent_rewards, logits_agent_actions):
        
        agent_pred = agent(state) #agent网络的输出值见policy.py中的PolicyTrans类的forward,即(all_logit_weights, all_feature_weights),但事实上应该是batch_size个这样的东西.
        agent_optimizer.zero_grad() #将agent_optimizer中的所有参数的梯度设置为0(reset),避免backward()时还有上次循环的梯度,一般在train循环开始时用
        
        action_label = torch.ones_like(agent_pred[0]).detach() 
        #与all_logit_weights(即agent_pred[0])形状一致的全为1的vector(形状为[batch_size, num_teachers],用1填充)。detach()将此张量从计算图中分离,避免后续梯度计算
        loss_logits = F.binary_cross_entropy(agent_pred[0], action_label, weight=rewards.unsqueeze(-1)) #看笔记
        loss_feature = F.binary_cross_entropy(agent_pred[1], action_label, weight=rewards.unsqueeze(-1))
        loss = loss_feature + loss_logits
        loss.backward()
        agent_optimizer.step() #将optimizer中的所有参数根据backward计算出的梯度优化一步.

        agent_loss.update(loss.item(), actions.size(0)) 
        # action.size(0)其实就是batch_size，即单个batch的样本数量. 
        # update将当前批次的损失值和样本数量传入，更新内部的累计损失和样本总数，从而动态计算损失的平均值。
        # 可以发现actions仅在这里用到，但是batch_size完全可以通过agent_rewards等得到.
        # 因此我感觉函数传入的logits_agent_actions参数完全是冗余的！

    if args.rank == 0:
        args.logger.info('Epoch:{}, agent Loss:{:.6f}'.format(epoch, agent_loss.avg))
'''

# 由于上面注释掉的部分里提到的logits_agent_actions的冗余问题，因此我直接将其删去以减少内存开销.
# 另: 注释在上面写过一遍，这里就全都不写了
def train_agent(args, epoch, agent_state, agent_rewards, agent, agent_optimizer):
    agent.train()
    agent_loss = AverageMeter('agent_loss', ':.4e')
    for state, rewards in zip(agent_state, agent_rewards):
        
        agent_pred = agent(state)
        agent_optimizer.zero_grad()
        
        action_label = torch.ones_like(agent_pred[0]).detach() 
        loss_logits = F.binary_cross_entropy(agent_pred[0], action_label, weight=rewards.unsqueeze(-1))
        loss_feature = F.binary_cross_entropy(agent_pred[1], action_label, weight=rewards.unsqueeze(-1))
        loss = loss_feature + loss_logits
        loss.backward()
        agent_optimizer.step()

        agent_loss.update(loss.item(), rewards.size(0)) 

    if args.rank == 0:
        args.logger.info('Epoch:{}, agent Loss:{:.6f}'.format(epoch, agent_loss.avg))

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


def train_avg(train_loader, model, criterion_list, optimizer, epoch, device, 
          args, feat_trans, teacher_models):
    
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
            all_teacher_info = []
            for t_model in teacher_models:
                t_features, t_logits = t_model(inputs, is_feat=True)
                t_feature = t_features[-1]
                t_feature = t_feature.detach() 
                t_logits = t_logits.detach() 
                
                teacher_features.append(t_features[-2])
                teacher_logits.append(t_logits)
                teacher_embeddings.append(t_features[-1])

        loss_cls = criterion_ce(logits, targets) #论文中的basic task loss
        
        # # 如果采用 logit ensemble teacher 则将下面这部分注释替换掉loss_kd部分.
        # avg_teacher_logits = torch.mean(torch.stack(teacher_logits), dim=0)
        #     # 若teacher_logits是[batch_size, num_classes]的形状，
        #     # stack后为[num_teachers, batch_size, num_classes]，mean后又恢复原形状
        # loss_kd = criterion_div(logits, avg_teacher_logits.detach())
        loss_kd = torch.tensor(0.).cuda(args.gpu)
        for idx in range(len(teacher_models)):
            loss_kd = loss_kd + criterion_div(logits, teacher_logits[idx].detach())
        loss_kd = loss_kd / len(teacher_models)
        loss_feat = torch.tensor(0.).cuda(args.gpu)
        
        if args.feat_kd == 'mse':
            feat_kd_func = FeatureMSELoss()
        elif args.feat_kd == 'kl':
            feat_kd_func = FeatureKLLoss(args.kd_T)

        for idx in range(len(teacher_models)):
            loss_feat = loss_feat +  (feat_kd_func(trans_student_features[idx], teacher_features[idx])).mean()
        loss_feat = loss_feat / len(teacher_models)
        loss_feat = args.feat_weight * loss_feat
        
        loss = loss_cls + loss_kd + loss_feat
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



def train(train_loader, model, criterion_list, optimizer, epoch, device, 
          args, agent, feat_trans, teacher_models, agent_optimizer):
    
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
    agent_states = [] # 对每个batch的agent网络的输入(agent_state)作记录. 
                    # 每个batch增加一项，累积到args.agent_step项后会在train_agent()中被调用，然后被清空.(后面三个也是这样)
    # logits_agent_actions = [] # 对每个batch的agent网络输出的各教师在logit kd loss中的权重作记录
    # feature_agent_actions = [] # 对每个batch的agent网络输出的各教师在feature kd loss中的权重作记录
    # 上面两项好像根本没必要记录，因此省略以降低内存占用
    agent_rewards = []
    
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
            all_teacher_info = [] #事实上这个变量好像没有用到,因此下面计算的teacher_info也没用
            for t_model in teacher_models:
                t_features, t_logits = t_model(inputs, is_feat=True)
                t_feature = t_features[-1]
                t_feature = t_feature.detach() 
                t_logits = t_logits.detach() 
                
                teacher_features.append(t_features[-2])
                teacher_logits.append(t_logits)
                teacher_embeddings.append(t_features[-1])
                # teacher_info = []
                # teacher_info.append(t_feature)
                # teacher_info.append(t_logits)
                # teacher_info.append(F.cross_entropy(t_logits, targets, reduction='none').unsqueeze(-1)) # 128*1
                # teacher_info = torch.cat(teacher_info, dim=1) # teacher logits , teacher feature , CE_loss , student_teacher_gap
                # all_teacher_info.append(teacher_info)

        agent_state = get_agent_state(trans_student_features, teacher_embeddings, logits, teacher_logits, targets, criterion_div)
        #agent_state是一个tuple: (teacher_infos, t_ces, t_s_logit_div, t_s_feat_div)

        agent_states.append(agent_state) # [bx3, bx3, bx3] (注:这里这个批注应该是错了,agent_state并不是(batch_size,3)的形状)
        
        with torch.no_grad():
            logits_actions, feature_actions, alpha, kd_temp = agent(agent_state) #agent新增了alpha参数做输出
            #agent的输出分别是logit kd loss的各教师权重以及feature kd loss的各教师权重
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

        # logits_agent_actions.append(logits_actions) # 列表中的每一项都是一个形状为(batch_size, teacher_num)的张量
        # feature_agent_actions.append(feature_actions)

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
        loss = loss_cls + alpha * loss_kd + loss_feat # 新增了参数α
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
        #print('normalized_reward', normalized_reward)
        agent_rewards.append(normalized_reward)
        
        
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
            # 根据reward, state, weight更新agent网络，然后进入下一个循环.
            # train_agent(args, epoch, agent_states, agent_rewards, logits_agent_actions, agent, agent_optimizer)
            train_agent(args, epoch, agent_states, agent_rewards, agent, agent_optimizer)
            agent_states = []
            # logits_agent_actions = []
            # feature_agent_actions = []
            agent_rewards = []
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
        # train_agent(args, epoch, agent_states, agent_rewards, logits_agent_actions, agent, agent_optimizer)
        train_agent(args, epoch, agent_states, agent_rewards, agent, agent_optimizer)


def test(epoch, net, device, val_loader, criterion_ce, args, verbose=True): #verbose控制是否输出详细信息
    test_loss_cls = AverageMeter('Loss', ':.4e')
    top1_num = 0
    top5_num = 0
    total = 0
    
    net.eval()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            batch_start_time = time.time()
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            features, logits = net(inputs, is_feat=True)
            loss_cls = torch.tensor(0.).cuda(args.gpu)
            loss_cls = criterion_ce(logits, targets)

            test_loss_cls.update(loss_cls.item(), inputs.size(0))

            top1, top5 = correct_num(logits, targets, topk=(1, 5))
            top1_num += top1
            top5_num += top5
            total += targets.size(0)
            
            if args.rank == 0 and verbose:
                print('Epoch:{}, batch_idx:{}/{}, Duration:{:.2f}, Test Top-1 Acc:{:.4f}'.format(
                    epoch, batch_idx, len(val_loader), time.time()-batch_start_time, (top1_num/(total)*100.).item()))
            
        class_acc1 = round((top1_num/total*100.).item(), 4) #先转化为百分数,然后.item()将tensor转化为标量,round(..., 4)保留四位.
        class_acc5 = round((top5_num/total*100.).item(), 4)

        if args.rank == 0 and verbose:
            args.logger.info('Test epoch:{}\t Test_loss_cls:{:.5f}\nTest top-1 accuracy: {}\nTest top-5 accuracy: {}'
                        .format(epoch, test_loss_cls.avg, str(class_acc1), str(class_acc5)))
    return class_acc1
    