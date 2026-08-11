import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

# 为了使timm库中创建的teacher模型支持is_feat参数而添加的一层包装
class Old_TeacherWrapper(nn.Module):
    def __init__(self, model_name, num_classes):
        super().__init__()
        # 需要两个模型，一个features_only=True仅输出中间特征列表，另一个则输出logit
        # 由于这两个都用的是预训练权重，因此权重应当是一致的，不会冲突
        self.backbone = timm.create_model(model_name, pretrained=True, features_only=True, num_classes=num_classes)
        self.full_model = timm.create_model(model_name, pretrained=True, features_only=False, num_classes=num_classes)

        self.backbone.eval()
        for t_n, t_p in self.backbone.named_parameters():
            t_p.requires_grad = False
        self.full_model.eval()
        for t_n, t_p in self.full_model.named_parameters():
            t_p.requires_grad = False

    def forward(self, x, is_feat=False):
        with torch.no_grad():  # 禁用梯度计算，避免显存泄漏
            if is_feat:
                # 中间特征列表 + 分类输出
                features = self.backbone(x)
                # 我们希望features[-1]形状为(1,D)而features[-2]形状为(1,C,H,W)
                # 但事实上如果直接用features的话，features[-1]和[-2]的形状均为(1,C,H,W)，二者仅C,H,W等的大小有所区别
                # 这是因为features[-1]和[-2]实际上是最后两个卷积层的输出，它会跳过最后的GAP(全局平均池化)层和全连接层
                # 为了获得(1,D)的形状，你要手动将features[-1]作池化
                pooled_feature = F.adaptive_avg_pool2d(features[-1], (1, 1))  # 压缩到 [1, C, 1, 1]
                pooled_feature = pooled_feature.view(pooled_feature.size(0), -1)  # 展平为 [1, C]
                
                logits = self.full_model(x)
                
                # 由于之后只会用到倒数两层features，因此不必传递所有的features
                return [features[-1], pooled_feature], logits 
            else:
                # 仅返回分类输出
                return self.full_model(x)

'''
在上面的不断修改中，我突然发现中间特征其实只用到了feature[-1]，
因此或许只需要在full_model中调用forward_features()就好了，
并不需要新创建一个features_only=True的backbone
下面是重写后的TeacherWrapper类
'''

# 为了使timm库中创建的teacher模型支持is_feat参数而添加的一层包装
class TeacherWrapper(nn.Module):
    def __init__(self, model_name, num_classes):
        super().__init__()
        self.full_model = timm.create_model(model_name, pretrained=True, features_only=False, num_classes=num_classes)
        self.full_model.eval()
        for t_n, t_p in self.full_model.named_parameters():
            t_p.requires_grad = False

    def forward(self, x, is_feat=False):
        with torch.no_grad():  # 禁用梯度计算，避免显存泄漏
            if is_feat:
                # 提取中间特征
                feature = self.full_model.forward_features(x)  # 使用 timm 的 forward_features 方法获取最后一层特征
                pooled_feature = F.adaptive_avg_pool2d(feature, (1, 1))  # 压缩到 [1, C, 1, 1]
                pooled_feature = pooled_feature.view(pooled_feature.size(0), -1)  # 展平为 [1, C]
                
                logits = self.full_model(x)  # 获取分类输出
                
                # 返回倒数两层特征和分类输出
                return [feature, pooled_feature], logits
            else:
                # 仅返回分类输出
                return self.full_model(x)
