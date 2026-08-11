import torch
import torch.nn as nn
import torch.nn.functional as F

class SACActor(nn.Module):
    def __init__(self, input_size, teacher_num, dynamic=False):
        """
        SAC Actor for MTKD-RL.
        Args:
            input_size: 每个教师的input_size构成的列表。
            teacher_num: 教师模型的数量。
            dynamic: 是否启用动态权重机制。
        """
        super(SACActor, self).__init__()
        self.teacher_num = teacher_num

        all_input_size = 0
        for idx in range(teacher_num):
            all_input_size = all_input_size + input_size[idx]

        # 输入特征处理
        # self.steam = nn.Sequential(
        #     nn.Linear(all_input_size, 128, bias=False),
        #     nn.ReLU()
        # )
        self.steam = nn.Sequential(
            nn.Linear(all_input_size, 256, bias=False),
            nn.ReLU(),
            #nn.Dropout(p=0.2),
            nn.Linear(256, 256, bias=False),
            nn.ReLU(),
            #nn.LayerNorm(256)
        )

        # SAC Actor 特性：输出动作分布的均值和标准差
        # 为什么是2 * teacher_num见forward()
        self.mean_head = nn.Linear(256, teacher_num * 2, bias=True)  # 动作均值（logits_actions 和 feature_actions）
        self.log_std_head = nn.Linear(256, teacher_num * 2, bias=True)  # 动作的log(标准差)

        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)

        # 动态权重机制
        self.dynamic = dynamic
        if dynamic:
            self.logit_weight_factor = nn.Parameter(torch.tensor([1., 1., 1.]), requires_grad=True)
            self.feature_weight_factor = nn.Parameter(torch.tensor([1., 1., 1.]), requires_grad=True)

        # 额外输出：alpha 和 kd_temp
        self.alpha_head = nn.Linear(256, 1, bias=True)
        self.kd_temp_head = nn.Linear(256, 1, bias=True)

    def forward(self, agent_state):
        """
        Forward pass for SAC Actor.
        Args:
            agent_state: 输入状态 (teacher_infos, t_ces, t_s_logit_div, t_s_feat_div)。
        Returns:
            logits_actions: 教师的 logits KD loss 权重。
            feature_actions: 教师的 feature KD loss 权重。
            alpha: 动态调整 KD loss 的权重。
            kd_temp: 蒸馏温度。
            action_mean: 动作分布的均值。
            action_log_std: 动作分布的对数标准差。
        """
        teacher_infos, t_ces, t_s_logit_div, t_s_feat_div = agent_state

        # 拼接所有教师信息
        all_teacher_infos = torch.cat(teacher_infos, dim=1)
        out1 = self.steam(all_teacher_infos)

        # 动作分布的均值和对数标准差
        action_mean = self.mean_head(out1)  # [batch_size, teacher_num * 2]
        #print("action_mean:", action_mean)
        action_log_std = self.log_std_head(out1)  # [batch_size, teacher_num * 2]
        action_log_std = torch.clamp(action_log_std, min=-20, max=2)  # 限制范围[-20, 2](这是SAC的默认)(standard clamp for SAC)

        # 采样动作
        std = action_log_std.exp()
        normal = torch.distributions.Normal(action_mean, std)
        actions = normal.rsample()

        # 将动作分为 logits_actions 和 feature_actions, 并用softmax归一化
        logits_actions_sampled = self.softmax(actions[:, :self.teacher_num])  # 前 teacher_num 个维度
        feature_actions_sampled = self.softmax(actions[:, self.teacher_num:])  # 后 teacher_num 个维度

        # 动态权重机制(和policy.py一致)
        weight_loss_t = (1. - F.softmax(t_ces, dim=1)) / (self.teacher_num - 1)
        weight_loss_t_s_logit_div = F.softmax(t_s_logit_div, dim=1)
        weight_loss_t_s_feat_div = F.softmax(t_s_feat_div, dim=1)

        if self.dynamic:
            l_f = F.softmax(self.logit_weight_factor, dim=0)
            f_f = F.softmax(self.feature_weight_factor, dim=0)
            logits_actions = (l_f[0] * logits_actions_sampled +
                              l_f[1] * weight_loss_t +
                              l_f[2] * weight_loss_t_s_logit_div)
            feature_actions = (f_f[0] * feature_actions_sampled +
                               f_f[1] * weight_loss_t +
                               f_f[2] * weight_loss_t_s_feat_div)
        else:
            logits_actions = (logits_actions_sampled + weight_loss_t + weight_loss_t_s_logit_div) / 3.
            feature_actions = (feature_actions_sampled + weight_loss_t + weight_loss_t_s_feat_div) / 3.

        # alpha 和 kd_temp
        alpha = 2 * self.sigmoid(self.alpha_head(out1))  # [batch_size, 1]
        kd_temp = 4 * self.sigmoid(self.kd_temp_head(out1)) + 1  # [batch_size, 1]

        return logits_actions, feature_actions, alpha, kd_temp, action_mean, action_log_std
        #return logits_actions, feature_actions, action_mean, action_log_std
    
class SACCritic(nn.Module):
    def __init__(self, input_size, teacher_num):
        """
        SAC Critic for MTKD-RL.
        Args:
            input_size: 每个教师的input_size构成的列表。
            teacher_num: 教师模型的数量。
        """
        super(SACCritic, self).__init__()
        self.teacher_num = teacher_num

        all_input_size = 0
        for idx in range(teacher_num):
            all_input_size = all_input_size + input_size[idx]

        # Critic 输入维度 = 状态特征维度 + 动作维度
        self.input_dim = all_input_size + teacher_num * 2  # 状态特征 + logits_actions + feature_actions

        # Critic 网络
        # self.q_net = nn.Sequential(
        #     nn.Linear(self.input_dim, 256),
        #     nn.ReLU(),
        #     nn.Linear(256, 256),
        #     nn.ReLU(),
        #     nn.Linear(256, 1)  # 输出Q值
        # )
        self.q_net = nn.Sequential(
            nn.Linear(self.input_dim, 512),  # 增加隐藏单元数量
            nn.ReLU(),
            nn.Linear(512, 256),  # 增加一层隐藏层
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, state, logits_actions, feature_actions):
        """
        Forward pass for SACCritic.
        Args:
            state: 输入状态 (teacher_infos, t_ces, t_s_logit_div, t_s_feat_div)。
            logits_actions: 教师的 logits KD loss 权重。
            feature_actions: 教师的 feature KD loss 权重。
        Returns:
            Q值: 状态-动作对的Q值。
        """
        # 拼接所有教师信息
        teacher_infos, t_ces, t_s_logit_div, t_s_feat_div = state # teacher_infos中其实已经包含了后三个.
        all_teacher_infos = torch.cat(teacher_infos, dim=1)

        # 拼接状态和动作
        actions = torch.cat([logits_actions, feature_actions], dim=1)  # [batch_size, teacher_num * 2]
        q_input = torch.cat([all_teacher_infos, actions], dim=1)  # [batch_size, input_dim]
        
        # 计算Q值
        q_value = self.q_net(q_input)
        return q_value
    

import random
import numpy as np

class ReplayBuffer:
    def __init__(self, buffer_size, state_dim, action_dim, device):
        """
        Replay Buffer for SAC.
        Args:
            buffer_size: Maximum number of transitions to store in the buffer.
            state_dim: Dimension of the state space.
            action_dim: Dimension of the action space.
            device: Device to store the sampled batches (e.g., 'cuda' or 'cpu').
        """
        self.buffer_size = buffer_size #最大容量
        self.device = device
        self.ptr = 0
        self.size = 0 #当前容量

        # Initialize buffers(让Replay Buffer完全在GPU上运行)
        self.states = torch.zeros((buffer_size, state_dim), dtype=torch.float32, device=device)
        self.actions = torch.zeros((buffer_size, action_dim), dtype=torch.float32, device=device)
        self.rewards = torch.zeros(buffer_size, dtype=torch.float32, device=device)
        self.next_states = torch.zeros((buffer_size, state_dim), dtype=torch.float32, device=device)
        #self.dones = torch.zeros(buffer_size, dtype=np.float32, device=device) #由于dones没啥用这里就忽略了

    def add(self, states, actions, rewards, next_states):
        """
        Add a batch of transitions to the buffer.
        Args:
            states: Current states, shape (batch_size, state_dim).
            actions: Actions taken, shape (batch_size, action_dim).
            rewards: Rewards received, shape (batch_size,).
            next_states: Next states after the actions, shape (batch_size, state_dim).
        """
        batch_size = states.shape[0]
        for i in range(batch_size):
            self.states[self.ptr] = states[i].to(self.device) # 确保张量在正确的device上
            self.actions[self.ptr] = actions[i].to(self.device)
            self.rewards[self.ptr] = rewards[i].to(self.device)
            self.next_states[self.ptr] = next_states[i].to(self.device)
            #self.dones[self.ptr] = done[i].to(self.device)

            self.ptr = (self.ptr + 1) % self.buffer_size
            self.size = min(self.size + 1, self.buffer_size)

    def sample(self, batch_size):
        """
        Sample a batch of transitions from the buffer.
        Args:
            batch_size: Number of transitions to sample.
        Returns:
            A tuple of (states, actions, rewards, next_states, dones) as PyTorch tensors.
        """
        indices = torch.randint(0, self.size, (batch_size,), device=self.device)

        states = self.states[indices]
        actions = self.actions[indices]
        rewards = self.rewards[indices].unsqueeze(1)
        next_states = self.next_states[indices]
        #dones = self.dones[indices].unsqueeze(1)

        return states, actions, rewards, next_states
    
    def clear(self):
        """
        Clear the replay buffer by resetting all stored data and pointers.
        """
        self.ptr = 0
        self.size = 0
        self.states.zero_()
        self.actions.zero_()
        self.rewards.zero_()
        self.next_states.zero_()
        # If you are using dones, uncomment the following line:
        # self.dones.zero_()

    def __len__(self):
        """
        Return the current size of the buffer.
        """
        return self.size