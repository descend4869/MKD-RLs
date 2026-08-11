import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['Policy', 'PolicyTrans', 'HighAgent']

class Policy(nn.Module):
    def __init__(self, input_size, output_size):
        super(Policy, self).__init__()

        self.head = nn.Sequential(
            nn.Linear(input_size, 128, bias=False),
            nn.ReLU(True),
            nn.Linear(128, output_size, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        output = self.head(input)
        return output 


class PolicyTrans(nn.Module):
    def __init__(self, input_size, teacher_num, dynamic=False):
        super(PolicyTrans, self).__init__()
        self.teacher_num = teacher_num
        
        self.sim_trans = nn.ModuleList([])
        all_input_size = 0
        for idx in range(teacher_num):
            all_input_size = all_input_size + input_size[idx]
        
        self.steam = nn.Sequential(
                nn.Linear(all_input_size, 128, bias=False),
                nn.ReLU()) 
        #两层的全连接网络,输入为所有教师的input_size的拼接,输出为128维的特征表示.
        #激活函数为ReLU,即f(x)=max(0,x).
        self.logit_head = nn.Linear(128, teacher_num, bias=True) #输入128维特征表示,输出各个teacher的logit KD loss的权重(因此维数是teacher_num)
        self.feature_head = nn.Linear(128, teacher_num, bias=True) #输出各个teacher的feature KD loss的权重
                
        self.sigmoid = nn.Sigmoid() #f(x) = 1 / (1 + e^(-x))
        self.softmax = nn.Softmax(dim=1) 
        #f(xi) = e^(xi) / Σk (e^(xk)). 这里的dim=1代表对第二个维度进行归一化(如[batch_size,num_classes]就会对num_classes进行归一化)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.dynamic = dynamic
        #动态权重机制：在dynamic为True时,引入两个可训练参数,分别用于控制logit/feature KD loss权重的动态调整.(required_grad=True即代表在训练中可变)
        if dynamic:
            self.logit_weight_factor = torch.nn.Parameter(torch.tensor([1., 1., 1.]), requires_grad=True)
            self.feature_weight_factor = torch.nn.Parameter(torch.tensor([1., 1., 1.]), requires_grad=True)

        #下面这两个是新加的，为了让agent输出参数alpha以及参数kd_temp(蒸馏温度)
        self.alpha_head = nn.Linear(128, 1, bias=True)
        self.kd_temp_head = nn.Linear(128, 1, bias=True)
        # self.kd_temp_head = nn.Linear(128, teacher_num, bias=True)

    def forward(self, agent_state):
        teacher_infos, t_ces, t_s_logit_div, t_s_feat_div = agent_state 
        #这四个都是列表,列表的每一项都是一个张量,对应一个teacher的信息.

        weight_loss_t = (1. - F.softmax(t_ces, dim=1)) / (self.teacher_num - 1)
        weight_loss_t_s_logit_div = F.softmax(t_s_logit_div, dim=1)
        weight_loss_t_s_feat_div = F.softmax(t_s_feat_div, dim=1)

        all_teacher_infos = torch.cat(teacher_infos, dim=1) 
        #原先teacher_infos是一个张量的列表,每个张量的形状为[batch_size, input_size[i]].
        #因此在第一维(dim=1)上拼接后得到的是[batch_size, sum(input_size)]形状的张量.
        out1 = self.steam(all_teacher_infos)
        logit_weights = self.softmax(self.logit_head(out1))
        feature_weights = self.softmax(self.feature_head(out1))
        #新增alpha这个输出, sigmoid会将它限制在[0, 2]中
        alpha = 2 * self.sigmoid(self.alpha_head(out1)) #注：这里输出的alpha是[batch_size, 1]形状的,之后用的话还得取mean()来变为scalar
        #新增蒸馏温度输出
        kd_temp = 4 * self.sigmoid(self.kd_temp_head(out1)) + 1 

        if self.dynamic:
            # self.logit_weight_factor.data = torch.clamp(self.logit_weight_factor.data, min=0.1, max=10.0)
            # self.feature_weight_factor.data = torch.clamp(self.feature_weight_factor.data, min=0.1, max=10.0)
            l_f = F.softmax(self.logit_weight_factor, dim=0) #logit_weight_factor本身只有一个维度,因此这会让它里面的三项和为1.
            f_f = F.softmax(self.feature_weight_factor, dim=0)
            all_logit_weights = (l_f[0]*logit_weights + l_f[1]*weight_loss_t + l_f[2]*weight_loss_t_s_logit_div) 
                # 由于这三个都是经过softmax的(和均为1),因此加权平均后依然满足和为1的性质!!
            all_feature_weights = (f_f[0] * feature_weights + f_f[1] * weight_loss_t + f_f[2] * weight_loss_t_s_feat_div)
        else:
            all_logit_weights = (logit_weights + weight_loss_t + weight_loss_t_s_logit_div)/ 3. 
            all_feature_weights = (feature_weights + weight_loss_t + weight_loss_t_s_feat_div) / 3.

        return all_logit_weights,  all_feature_weights, alpha, kd_temp


# New: 只输出 kd_temp 的高层agent
class HighAgent(nn.Module):
    def __init__(self, input_size, teacher_num):
        super(HighAgent, self).__init__()
        self.teacher_num = teacher_num

        all_input_size = 0
        for idx in range(teacher_num):
            all_input_size = all_input_size + input_size[idx]
        
        # self.steam = nn.Sequential(
        #         nn.Linear(all_input_size, 128, bias=False),
        #         nn.ReLU()) 
        self.steam = nn.Sequential(
            nn.Linear(all_input_size, 256, bias=False),  # 增加隐藏层维度
            nn.ReLU(),
            nn.Linear(256, 128, bias=False),
            nn.ReLU()
        )
        #全连接网络,输入为所有教师的input_size的拼接,输出为128维的特征表示.
        #激活函数为ReLU,即f(x)=max(0,x).
                
        self.sigmoid = nn.Sigmoid() #f(x) = 1 / (1 + e^(-x))
        self.softmax = nn.Softmax(dim=1) 
        #f(xi) = e^(xi) / Σk (e^(xk)). 这里的dim=1代表对第二个维度进行归一化(如[batch_size,num_classes]就会对num_classes进行归一化)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # 注意力机制. 
        # 输入维度为128，使用4个注意力头(每个头独立计算注意力分数,结果最终拼接,用于捕获不同的特征模式)，
        # batch_first表示第一个维度是batch_size
        self.attention = nn.MultiheadAttention(embed_dim=128, num_heads=4, batch_first=True)

        self.kd_temp_head = nn.Linear(128, teacher_num, bias=True)  # 每个教师一个温度

    def forward(self, agent_state):
        teacher_infos, t_ces, t_s_logit_div, t_s_feat_div = agent_state 
        #这四个都是列表,列表的每一项都是一个张量,对应一个teacher的信息.

        all_teacher_infos = torch.cat(teacher_infos, dim=1) 
        #原先teacher_infos是一个张量的列表,每个张量的形状为[batch_size, input_size[i]].
        #因此在第一维(dim=1)上拼接后得到的是[batch_size, sum(input_size)]形状的张量.
        out = self.steam(all_teacher_infos)

        
        # 注意力机制
        # 注意力机制会计算query和key的相关性分数，并根据这些分数对value进行加权求和
        out = out.unsqueeze(1)  # 添加时间维度，从[batch_size, 128]变为[batch_size, 1, 128]
            # 这是为了适配MultiheadAttention的输入格式，它需要输入[batch_size, seq_len, embed_dim]的张量
        out, _ = self.attention(out, out, out) # Q,K,V全是相同的输入，称为自注意力机制，计算输入特征之间的相关性
        out = out.squeeze(1) # 去掉时间维度，将输出恢复到[batch_size, 128]的形状'
        
        
        kd_temp = 4 * self.sigmoid(self.kd_temp_head(out)) + 1 

        return kd_temp
    

# New: 输出 kd_temp 以及 teacher_importance 的高层agent
class HighAgent2(nn.Module):
    def __init__(self, input_size, teacher_num):
        super(HighAgent2, self).__init__()
        self.teacher_num = teacher_num

        all_input_size = 0
        for idx in range(teacher_num):
            all_input_size = all_input_size + input_size[idx]
        
        self.steam = nn.Sequential(
            nn.Linear(all_input_size, 256, bias=False),  # 增加隐藏层维度
            nn.ReLU(),
            nn.Linear(256, 128, bias=False),
            nn.ReLU()
        )
        #全连接网络,输入为所有教师的input_size的拼接,输出为128维的特征表示.
        #激活函数为ReLU,即f(x)=max(0,x).
                
        self.sigmoid = nn.Sigmoid() #f(x) = 1 / (1 + e^(-x))
        self.softmax = nn.Softmax(dim=1) 
        #f(xi) = e^(xi) / Σk (e^(xk)). 这里的dim=1代表对第二个维度进行归一化(如[batch_size,num_classes]就会对num_classes进行归一化)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # 注意力机制. 
        # 输入维度为128，使用4个注意力头(每个头独立计算注意力分数,结果最终拼接,用于捕获不同的特征模式)，
        # batch_first表示第一个维度是batch_size
        self.attention = nn.MultiheadAttention(embed_dim=128, num_heads=4, batch_first=True)

        self.kd_temp_head = nn.Linear(128, teacher_num, bias=True)  # 每个教师一个温度
        self.teacher_importance_head = nn.Linear(128, teacher_num, bias=True)   # 各教师的重要程度

    def forward(self, agent_state):
        teacher_infos, t_ces, t_s_logit_div, t_s_feat_div = agent_state 
        #这四个都是列表,列表的每一项都是一个张量,对应一个teacher的信息.

        all_teacher_infos = torch.cat(teacher_infos, dim=1) 
        #原先teacher_infos是一个张量的列表,每个张量的形状为[batch_size, input_size[i]].
        #因此在第一维(dim=1)上拼接后得到的是[batch_size, sum(input_size)]形状的张量.
        out = self.steam(all_teacher_infos)

        
        # 注意力机制
        # 注意力机制会计算query和key的相关性分数，并根据这些分数对value进行加权求和
        out = out.unsqueeze(1)  # 添加时间维度，从[batch_size, 128]变为[batch_size, 1, 128]
            # 这是为了适配MultiheadAttention的输入格式，它需要输入[batch_size, seq_len, embed_dim]的张量
        out, _ = self.attention(out, out, out) # Q,K,V全是相同的输入，称为自注意力机制，计算输入特征之间的相关性
        out = out.squeeze(1) # 去掉时间维度，将输出恢复到[batch_size, 128]的形状'
        
        
        kd_temp = 4 * self.sigmoid(self.kd_temp_head(out)) + 1 
        teacher_importance = self.softmax(self.teacher_importance_head(out)) # 和为1的teacher_importance张量

        return kd_temp, teacher_importance
