import torch.nn as nn

__all__ = ['FeatureKLLoss', 'FeatureMSELoss']


class FeatureMSELoss(nn.Module):
    """Fitnets: hints for thin deep nets, ICLR 2015"""
    def __init__(self):
        super(FeatureMSELoss, self).__init__()

    def forward(self, f_s, f_t):
        f_s = f_s.view(f_s.size(0), -1) 
            #重新调整为二维张量,形状为(batch_size, -1),其中-1表示自由调整.
            #例如原先feature_student形状是(b,c,h,w),那现在是(b,c*h*w).
        f_t = f_t.view(f_t.size(0), -1) 
        loss = ((f_s - f_t)**2).mean(1) 
            #逐元素差值取平方,形状不变还是(batch_size, num_features)
            #.mean(1)对张量的第一维进行求均值操作,输出形状是(batch_size,)
        return loss


class ChannelNorm(nn.Module):
    def __init__(self):
        super(ChannelNorm, self).__init__()
    def forward(self,featmap):
        n,c,h,w = featmap.shape
        featmap = featmap.reshape((n,c,-1)) #形状变成(n,c,h*w)
        featmap = featmap.softmax(dim=-1) #对最后一维进行softmax,使它的每个元素内的数和为1
        return featmap
    
    
class FeatureKLLoss(nn.Module):
    def __init__(self, temperature=4.0):
        super(FeatureKLLoss, self).__init__()
        self.normalize = ChannelNorm()
        self.criterion = nn.KLDivLoss(reduction='none')
        self.temperature = temperature
       
    def forward(self, f_s, f_t):
        #n,c,h,w = f_s.shape
        norm_s = self.normalize(f_s/self.temperature) #除以温度系数,分布更平滑.再用softmax归一化转化为概率分布.
        norm_t = self.normalize(f_t.detach()/self.temperature)
        norm_s = norm_s.log() #取对数,因为nn.KLDivLoss要求第一个分布必须是对数概率分布,第二个是普通概率分布.

        loss = self.criterion(norm_s, norm_t).sum(-1).mean(-1)
            #norm_s和norm_t形状均为(n,c,h*w).
            #sum(-1)对最后一维(特征维度)求和,得到每个通道的KL散度.
            #mean(-1)对通道维度求均值,得到每个样本的平均KL散度.

        return loss * (self.temperature**2)