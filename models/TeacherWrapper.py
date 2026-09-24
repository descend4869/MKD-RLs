import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class TeacherWrapper(nn.Module):
    def __init__(self, model_name, num_classes):
        super().__init__()
        self.full_model = timm.create_model(model_name, pretrained=True, features_only=False, num_classes=num_classes)
        self.full_model.eval()
        for t_n, t_p in self.full_model.named_parameters():
            t_p.requires_grad = False

    def forward(self, x, is_feat=False):
        with torch.no_grad():  
            if is_feat:
                feature = self.full_model.forward_features(x)  
                pooled_feature = F.adaptive_avg_pool2d(feature, (1, 1))  
                pooled_feature = pooled_feature.view(pooled_feature.size(0), -1)  
                logits = self.full_model(x) 
                return [feature, pooled_feature], logits
            else:
                return self.full_model(x)
