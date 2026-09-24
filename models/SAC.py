import torch
import torch.nn as nn
import torch.nn.functional as F

class SACActor(nn.Module):
    def __init__(self, input_size, teacher_num, dynamic=False):
        """
        SAC Actor for MTKD-RL.
        """
        super(SACActor, self).__init__()
        self.teacher_num = teacher_num

        all_input_size = 0
        for idx in range(teacher_num):
            all_input_size = all_input_size + input_size[idx]

        self.steam = nn.Sequential(
            nn.Linear(all_input_size, 256, bias=False),
            nn.ReLU(),
            nn.Linear(256, 256, bias=False),
            nn.ReLU(),
        )

        self.mean_head = nn.Linear(256, teacher_num * 2, bias=True)  
        self.log_std_head = nn.Linear(256, teacher_num * 2, bias=True)  

        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)

        self.dynamic = dynamic
        if dynamic:
            self.logit_weight_factor = nn.Parameter(torch.tensor([1., 1., 1.]), requires_grad=True)
            self.feature_weight_factor = nn.Parameter(torch.tensor([1., 1., 1.]), requires_grad=True)

        self.alpha_head = nn.Linear(256, 1, bias=True)
        self.kd_temp_head = nn.Linear(256, 1, bias=True)

    def forward(self, agent_state):
        """
        Forward pass for SAC Actor.
        """
        teacher_infos, t_ces, t_s_logit_div, t_s_feat_div = agent_state

        all_teacher_infos = torch.cat(teacher_infos, dim=1)
        out1 = self.steam(all_teacher_infos)

        action_mean = self.mean_head(out1)  # [batch_size, teacher_num * 2]
        action_log_std = self.log_std_head(out1)  # [batch_size, teacher_num * 2]
        action_log_std = torch.clamp(action_log_std, min=-20, max=2)  

        std = action_log_std.exp()
        normal = torch.distributions.Normal(action_mean, std)
        actions = normal.rsample()

        logits_actions_sampled = self.softmax(actions[:, :self.teacher_num])  
        feature_actions_sampled = self.softmax(actions[:, self.teacher_num:]) 

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

        alpha = 2 * self.sigmoid(self.alpha_head(out1))  # [batch_size, 1]
        kd_temp = 4 * self.sigmoid(self.kd_temp_head(out1)) + 1  # [batch_size, 1]

        return logits_actions, feature_actions, alpha, kd_temp, action_mean, action_log_std
    
class SACCritic(nn.Module):
    def __init__(self, input_size, teacher_num):
        """
        SAC Critic for MTKD-RL.
        """
        super(SACCritic, self).__init__()
        self.teacher_num = teacher_num

        all_input_size = 0
        for idx in range(teacher_num):
            all_input_size = all_input_size + input_size[idx]

        self.input_dim = all_input_size + teacher_num * 2 

        self.q_net = nn.Sequential(
            nn.Linear(self.input_dim, 512),  
            nn.ReLU(),
            nn.Linear(512, 256),  
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, state, logits_actions, feature_actions):
        """
        Forward pass for SACCritic.
        """
        teacher_infos, t_ces, t_s_logit_div, t_s_feat_div = state 
        all_teacher_infos = torch.cat(teacher_infos, dim=1)

        actions = torch.cat([logits_actions, feature_actions], dim=1)  # [batch_size, teacher_num * 2]
        q_input = torch.cat([all_teacher_infos, actions], dim=1)  # [batch_size, input_dim]
        
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
        self.buffer_size = buffer_size 
        self.device = device
        self.ptr = 0
        self.size = 0

        self.states = torch.zeros((buffer_size, state_dim), dtype=torch.float32, device=device)
        self.actions = torch.zeros((buffer_size, action_dim), dtype=torch.float32, device=device)
        self.rewards = torch.zeros(buffer_size, dtype=torch.float32, device=device)
        self.next_states = torch.zeros((buffer_size, state_dim), dtype=torch.float32, device=device)

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
            self.states[self.ptr] = states[i].to(self.device) 
            self.actions[self.ptr] = actions[i].to(self.device)
            self.rewards[self.ptr] = rewards[i].to(self.device)
            self.next_states[self.ptr] = next_states[i].to(self.device)

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

    def __len__(self):
        """
        Return the current size of the buffer.
        """
        return self.size