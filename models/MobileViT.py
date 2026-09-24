import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List

# ---------------------------
# Helper blocks
# ---------------------------
class ConvBNAct(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=None, act=True):
        super().__init__()
        if p is None:
            p = k // 2
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class DepthwiseConv(nn.Module):
    def __init__(self, in_ch, k=3, s=1):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, kernel_size=k, stride=s, padding=k//2, groups=in_ch, bias=False)
        self.bn = nn.BatchNorm2d(in_ch)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.bn(self.dw(x)))
    
class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super().__init__()
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio)) 
        
        self.use_res_connect = (stride == 1 and inp == oup)

        layers = []
        if expand_ratio != 1:
            layers.append(ConvBNAct(inp, hidden_dim, k=1))  
        layers += [
            DepthwiseConv(hidden_dim, k=3, s=stride), 
            ConvBNAct(hidden_dim, oup, k=1, act=False)
        ]
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)
        
# ---------------------------
# Transformer (lightweight)
# ---------------------------
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim) 
        self.fn = fn

    def forward(self, x):
        return self.fn(self.norm(x))

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim, heads=4, proj_drop=0.0):
        super().__init__()
        self.heads = heads 
        self.head_dim = dim // heads 
        self.scale = (dim // heads) ** -0.5 
        inner_dim = dim 
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False) 
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(proj_drop)
        ) 

    def forward(self, x):
        B, N, D = x.shape
        h = self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1) 
        q, k, v = [t.reshape(B, N, h, D//h).permute(0,2,1,3) for t in qkv] 
        attn = (q @ k.transpose(-2,-1)) * self.scale 
        attn = attn.softmax(dim=-1) 
        out = (attn @ v).permute(0,2,1,3).reshape(B, N, D) 
        return self.to_out(out) 
    
class TransformerEncoder(nn.Module):
    def __init__(self, dim, depth=2, heads=4, mlp_ratio=2.0, dropout=0.0):
        super().__init__()
        layers = []
        for _ in range(depth):
            layers.append(nn.ModuleList([
                PreNorm(dim, MultiHeadSelfAttention(dim, heads=heads, proj_drop=dropout)),
                PreNorm(dim, FeedForward(dim, int(dim*mlp_ratio), dropout))
            ]))
        self.layers = nn.ModuleList(layers)

    def forward(self, x): 
        for attn, ff in self.layers:
            x = x + attn(x) 
            x = x + ff(x)
        return x

# ---------------------------
# MobileViT Block
# ---------------------------
class MobileViTBlock(nn.Module):
    """
    Implements the MobileViT block:
    - Local processing with convolutions
    - Unfold into non-overlapping patches -> Transformer -> fold back
    - Fusion with conv
    """
    def __init__(self, in_channels, transformer_dim, ffn_dim, n_transformer_blocks, patch_size=(2,2), heads=4, 
                 kernel_size=3, dropout=0.0):
        super().__init__()
        p_h, p_w = patch_size
        self.patch_h = p_h
        self.patch_w = p_w

        # local representation: conv -> conv (paper uses convs to get local features)
        self.local_rep = nn.Sequential(
            ConvBNAct(in_channels, in_channels, k=kernel_size, s=1), 
            ConvBNAct(in_channels, transformer_dim, k=1, s=1) 
        )

        # transformer (operates on flattened patches)
        self.transformer = TransformerEncoder(dim=transformer_dim, depth=n_transformer_blocks, heads=heads, 
                                              mlp_ratio=ffn_dim/transformer_dim, dropout=dropout) 

        self.fuse_conv1 = ConvBNAct(transformer_dim, in_channels, k=1, s=1)
        self.fuse_conv2 = ConvBNAct(2*in_channels, in_channels, k=kernel_size, s=1) 

    def forward(self, x):
        B, C, H, W = x.shape
        y = x.clone() 
        '''local representation'''
        local = self.local_rep(x)  
        D = local.shape[1] 

        # pad so H and W divisible by patch sizes
        pad_h = (self.patch_h - (H % self.patch_h)) % self.patch_h 
        pad_w = (self.patch_w - (W % self.patch_w)) % self.patch_w
        if pad_h or pad_w:
            local = F.pad(local, (0, pad_w, 0, pad_h))

        '''unfold'''
        _, _, Hs, Ws = local.shape 
        # reshape into patches: split into (num_patches, patch_h*patch_w, D)
        ph, pw = self.patch_h, self.patch_w
        # reshape -> (B, D, Hs/ph, ph, Ws/pw, pw)
        local_reshaped = local.reshape(B, D, Hs//ph, ph, Ws//pw, pw)
        # permute to bring patches contiguously: (B, num_patches, ph*pw, D)
        local_reshaped = local_reshaped.permute(0,2,4,3,5,1).contiguous()  # (B, Hn, Wn, ph, pw, D)
        B, Hn, Wn, ph, pw, D = local_reshaped.shape
        num_patches = Hn * Wn
        patches = local_reshaped.view(B, num_patches, ph*pw, D) 
        patches_token = patches.mean(dim=2)  # (B, num_patches, D)

        '''transformer'''
        trans_out = self.transformer(patches_token)  # (B, num_patches, D)

        '''fold'''
        trans_broadcast = trans_out.unsqueeze(2).repeat(1,1,ph*pw,1)  
        trans_broadcast = trans_broadcast.view(B, Hn, Wn, ph, pw, D)   
        trans_back = trans_broadcast.permute(0,5,1,3,2,4).contiguous().view(B, D, Hs, Ws)

        if pad_h or pad_w:
            trans_back = trans_back[:, :, :H, :W]

        '''fusion'''
        out = self.fuse_conv1(trans_back)
        out = torch.cat((out, y), 1) 
        out = self.fuse_conv2(out)
        return out
    
# ---------------------------
# MobileViT backbone
# ---------------------------
class MobileViT(nn.Module):
    def __init__(self, dims, channels, mv2_exp, kernel_size=3, patch_size=(2,2), num_classes=1000):
        super().__init__()
        ph, pw = patch_size

        L = [2, 4, 3] 

        self.conv1 = ConvBNAct(3, channels[0], k=3, s=2) 

        # layer 1
        self.layer1 = InvertedResidual(channels[0], channels[1], stride=1, expand_ratio=mv2_exp)

        # layer 2
        self.layer2 = nn.Sequential(
            InvertedResidual(channels[1], channels[2], stride=2, expand_ratio=mv2_exp),
            InvertedResidual(channels[2], channels[3], stride=1, expand_ratio=mv2_exp),
            InvertedResidual(channels[2], channels[3], stride=1, expand_ratio=mv2_exp),
        )

        # layer 3
        self.layer3 = nn.Sequential(
            InvertedResidual(channels[3], channels[4], stride=2, expand_ratio=mv2_exp),
            MobileViTBlock(in_channels=channels[5], transformer_dim=dims[0], ffn_dim=2*dims[0], n_transformer_blocks=L[0])
        )

        # layer 4
        self.layer4 = nn.Sequential(
            InvertedResidual(channels[5], channels[6], stride=2, expand_ratio=mv2_exp),
            MobileViTBlock(in_channels=channels[7], transformer_dim=dims[1], ffn_dim=2*dims[1], n_transformer_blocks=L[1])
        )

        # layer 5
        self.layer5 = nn.Sequential(
            InvertedResidual(channels[7], channels[8], stride=2, expand_ratio=mv2_exp),
            MobileViTBlock(in_channels=channels[9], transformer_dim=dims[2], ffn_dim=2*dims[2], n_transformer_blocks=L[2])
        )

        self.conv2 = ConvBNAct(channels[9], channels[10], k=1)
        self.avgpool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )
        self.classifier = nn.Linear(channels[10], num_classes, bias=False)

    def forward(self, x, is_feat=False):
        features = [] 
        x = self.conv1(x)
        if is_feat:
            features.append(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        if is_feat:
            features.append(x)
        
        x = self.conv2(x)
        if is_feat:
            features.append(x)
        
        x = self.avgpool(x)
        if is_feat:
            features.append(x)
        
        x = self.classifier(x)

        if is_feat:
            return features, x
        return x

def mobilevit_xxs():
    dims = [64, 80, 96]
    channels = [16, 16, 24, 24, 48, 48, 64, 64, 80, 80, 320]
    return MobileViT(dims, channels, mv2_exp=2, num_classes=1000)

def mobilevit_xs():
    dims = [96, 120, 144]
    channels = [16, 32, 48, 48, 64, 64, 80, 80, 96, 96, 384]
    return MobileViT(dims, channels, mv2_exp=4, num_classes=1000)

def mobilevit_s():
    dims = [144, 192, 240]
    channels = [16, 32, 64, 64, 96, 96, 128, 128, 160, 160, 640]
    return MobileViT(dims, channels, mv2_exp=4, num_classes=1000)

def mobilevit(num_classes):
    dims = [96, 120, 144]
    channels = [16, 32, 48, 48, 64, 64, 80, 80, 96, 96, 384]
    return MobileViT(dims, channels, mv2_exp=4, num_classes=num_classes)
