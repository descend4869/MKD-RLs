import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List

# ---------------------------
# Helper blocks
# ---------------------------
class ConvBNAct(nn.Module):
    # 实现了 卷积 + BatchNorm + 激活(可选)
    def __init__(self, in_ch, out_ch, k=3, s=1, p=None, act=True):
        super().__init__()
        if p is None:
            p = k // 2
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False)
            # in_ch是输入图像的通道数，out_ch是卷积产生的通道数，k是卷积核大小，s是卷积步幅，
            # p是填充大小(默认为 k // 2 即自动计算为使输入输出尺寸一致的填充)
            # 注: 这里采用的是标准卷积，有通道混合，与DepthwiseConv模块中每通道独立的深度卷积不同
        self.bn = nn.BatchNorm2d(out_ch)
            # 对(N,C,H,W)这一batch_size为N的图片群在Channel上作归一化
            # 参数num_features = out_ch对输入图片的Channel大小作了规定
        self.act = nn.SiLU() if act else nn.Identity()
            # SiLU(Sigmoid Linear Unit) 又称swish函数
            # 即 f(x) = x * sigmoid(x) = x / (1 + e^(-x))
            # 对任意维度都是逐元素采用此函数

        # 输入为(N, in_ch, H, W)的图片，
        # 经过conv后(默认p使得输入输出图片尺寸一致)变为(N, out_ch, H, W)
        # 再经过bn还是(N, out_ch, H, W)
        # 经过SiLU还是(N, out_ch, H, W)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class DepthwiseConv(nn.Module):
    def __init__(self, in_ch, k=3, s=1):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, kernel_size=k, stride=s, padding=k//2, groups=in_ch, bias=False)
            # 采用分组卷积(与标准卷积的差异体现为group参数)，介绍见笔记中的链接
            # 由于groups = in_ch，这代表着分组数和输入通道数一致，因此每个通道会独立卷积不融合
            # 分组卷积带来的是参数量的大量减小.(链接有解释)
        self.bn = nn.BatchNorm2d(in_ch)
        self.act = nn.SiLU()

        # 类似于ConvBNAct类，输入为(N, C, H, W)的图片，输出和输入形状一致.

    def forward(self, x):
        return self.act(self.bn(self.dw(x)))
    
class InvertedResidual(nn.Module):
    """ MobileNetV2网络中的inverted residual模块, 也是MobileViT论文中的MV2块"""
    # inverted residual是倒置残差的意思，它与残差的区别见笔记
    def __init__(self, inp, oup, stride, expand_ratio):
        super().__init__()
        assert stride in [1, 2]

        # expand_ratio即扩展倍数，将input channel扩展为原来的expand_ratio倍大
        hidden_dim = int(round(inp * expand_ratio)) 
        
        # 只有在stride = 1(空间尺寸不变)并且输入输出通道一致(这是才能使用加法)时才使用残差连接
        self.use_res_connect = (stride == 1 and inp == oup)

        layers = []
        if expand_ratio != 1:
            # 用1*1卷积核来扩展通道(通道数inp -> inp * expand_ratio)
            # 把通道变宽，为后续 depthwise 卷积提供更丰富的特征
            layers.append(ConvBNAct(inp, hidden_dim, k=1))  
        layers += [
            DepthwiseConv(hidden_dim, k=3, s=stride), 
                # 做空间特征提取(每个通道单独卷积不混合)，stride = 1 or 2 → 负责是否下采样.
                # 若 stride = 2 则图片的 H 和 W 都会变为1/2.
            ConvBNAct(hidden_dim, oup, k=1, act=False)
                # 用1*1卷积核来压缩通道(通道数inp * expand_ratio -> inp)
                # 为什么没有激活函数: 论文强调最后一个 1×1 卷积必须是线性的，否则会破坏特征空间，使残差连接失效
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
    # 先做LayerNorm，再把结果送进某个子模块，如 Self-Attention 或 FeedForward(FFN)
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim) #对输入的最后一个维度作标准化，并规定最后一个维度的形状是dim
        self.fn = fn

    def forward(self, x):
        # 输入 x: (B, N, D)
        # B = batch size; N = tokens 个数(patch 数量); D = embedding dim
        # 输入和输出形状一致
        return self.fn(self.norm(x))

class FeedForward(nn.Module):
    # Transformer中的MLP层，介绍见笔记
    # hidden_dim 通常是输入维度的 2~4 倍，在TransformerEncoder类中采用mpl_ratio来控制此倍数
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
    # Transformer中的多头自注意力
    def __init__(self, dim, heads=4, proj_drop=0.0):
        # dim: forward中输入x的第三个维度embedding dim. 也就是说，输入x要是(B, N, dim)的形式.
        # heads: 多头自注意力的头数.会将维度dim分割成heads份.每个head的维度是(dim // heads).
        # proj_drop: 最后输出投影的 dropout 比例，防止过拟合.
        super().__init__()
        self.heads = heads # 记录head个数
        self.head_dim = dim // heads # 每个head的维度
        self.scale = (dim // heads) ** -0.5 # 即注意力公式中的 d^(-0.5)
        inner_dim = dim # 表示Q,K,V的维度都等于输入维度
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False) # 让输入x被转化为Q,K,V的拼接
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(proj_drop)
        ) # 在多头拼接出out后重新将多头信息混合并保持维度不变，即多头自注意力公式中的WO

    def forward(self, x):
        # 输入x: (B, N, D). 
        # B = batch size; N = tokens 个数(patch数量); D = embedding dim
        B, N, D = x.shape
        h = self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1) # to_qkv后输出(B, N, 3*D)，chunk将其切成三份(B, N, D)
        # reshape for heads: (B, h, N, D/h)
        q, k, v = [t.reshape(B, N, h, D//h).permute(0,2,1,3) for t in qkv] # permute就是将维度换位
        attn = (q @ k.transpose(-2,-1)) * self.scale 
            # 即 Q * KT / d^(-0.5).
            # q和k形状均为(B, h, N, D/h)，将k transpose后为(B, h, D/h, N)就是k的转置，矩阵乘后为(B, h, N, N)
        attn = attn.softmax(dim=-1) 
        out = (attn @ v).permute(0,2,1,3).reshape(B, N, D) 
            # attn @ v 结果为 (B, h, N, D/h); permute后为(B, N, h, D/h); reshape后变回(B, N, D).
            # 这事实上已经完成了各个head结果的拼接
        return self.to_out(out) # 最后再通过一个线性变换混合heads
    
class TransformerEncoder(nn.Module):
    # ViT中用到的transformer，其实是完整transformer的编码器
    # 示意图见笔记
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
        # 输入x: (B, N, D). 
        # B = batch size; N = tokens 个数(patch数量); D = embedding dim
        for attn, ff in self.layers:
            x = x + attn(x) # 残差连接
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
            ConvBNAct(in_channels, in_channels, k=kernel_size, s=1), # 一般就是3*3内核的卷积
            ConvBNAct(in_channels, transformer_dim, k=1, s=1) # 1*1卷积
        )

        # transformer (operates on flattened patches)
        self.transformer = TransformerEncoder(dim=transformer_dim, depth=n_transformer_blocks, heads=heads, 
                                              mlp_ratio=ffn_dim/transformer_dim, dropout=dropout) 
                        # ffn_dim就是FeedForward的hidden_dim大小，而mlp_ratio的定义就是从transformer_dim -> hidden_dim的放大倍数

        # fusion convs
        # self.fuse = nn.Sequential(
        #     ConvBNAct(transformer_dim, in_channels, k=1, s=1),
        #     ConvBNAct(in_channels, in_channels, k=kernel_size, s=1) # 一般就是3*3内核的卷积
        # )
        self.fuse_conv1 = ConvBNAct(transformer_dim, in_channels, k=1, s=1)
        self.fuse_conv2 = ConvBNAct(2*in_channels, in_channels, k=kernel_size, s=1) # 一般就是3*3内核的卷积

    def forward(self, x):
        # 输入图片x: (B, C, H, W)
        B, C, H, W = x.shape
        y = x.clone() # 保存x的复制用于最后的拼接
        '''local representation'''
        local = self.local_rep(x)  # 输出形状(B, D, H, W)
        D = local.shape[1] # D就是transformer_dim

        # pad so H and W divisible by patch sizes
        # 当patch_size=2×2时，H 或 W 可能不是偶数.因此尝试对local的长宽进行填充(padding)以使其能被整除.
        pad_h = (self.patch_h - (H % self.patch_h)) % self.patch_h 
            # 括号里算出需要H方向需要填充的数目，再 % self.patch_h 是让已经是整数倍的H不需要再填充.
        pad_w = (self.patch_w - (W % self.patch_w)) % self.patch_w
        if pad_h or pad_w:
            local = F.pad(local, (0, pad_w, 0, pad_h))
            # 填充函数pad()的四个参数分别是(left, right, top, bottom)
            # 因此这表示在左和顶不填充，在右和底填充

        '''patch展开(unfold)'''
        _, _, Hs, Ws = local.shape # 在pad()后重新获取图片的Height和Weight
        # reshape into patches: split into (num_patches, patch_h*patch_w, D)
        ph, pw = self.patch_h, self.patch_w
        # reshape -> (B, D, Hs/ph, ph, Ws/pw, pw)
        local_reshaped = local.reshape(B, D, Hs//ph, ph, Ws//pw, pw)
        # permute to bring patches contiguously: (B, num_patches, ph*pw, D)
        local_reshaped = local_reshaped.permute(0,2,4,3,5,1).contiguous()  # (B, Hn, Wn, ph, pw, D)
        B, Hn, Wn, ph, pw, D = local_reshaped.shape # 其中 Hn = Hs/ph，Wn = Ws/pw
        num_patches = Hn * Wn
        patches = local_reshaped.view(B, num_patches, ph*pw, D) # 得到形状(B, num_patches, tokens_per_patch, D)
        # 接着采用简化版本，将每个patch中ph*pw个token用平均合成一个token，再送入transformer
        patches_token = patches.mean(dim=2)  # (B, num_patches, D)

        '''transformer'''
        trans_out = self.transformer(patches_token)  # (B, num_patches, D)

        '''patch复原(fold)'''
        # 由于之前平均化了，因此这里需要将输出复制ph*pw份以恢复原形状
        trans_broadcast = trans_out.unsqueeze(2).repeat(1,1,ph*pw,1)  
            # unsqueeze(2)在第2维(index=2)插入一个新维度，形状变为(B, num_patches, 1, D)
            # repeat中的参数分别表示在对应维度上重复的次数，因此这里仅在第2维重复ph*pw次，其它保持原样
            # 最终形状变为(B, num_patches, ph*pw, D)
        trans_broadcast = trans_broadcast.view(B, Hn, Wn, ph, pw, D)
            # view()将形状从(B, Hn*Wn, ph*pw, D)变为(B, Hn, Wn, ph, pw, D)
        trans_back = trans_broadcast.permute(0,5,1,3,2,4).contiguous().view(B, D, Hs, Ws)
            # permute()改变张量的维度顺序
            # contiguous()确保张量在内存中的布局是连续的(有时 permute 会导致非连续布局)

        # 如果之前进行过填充(padding)，则将填充后的Hs和Ws的右和底去掉一部分以变回H和W
        if pad_h or pad_w:
            trans_back = trans_back[:, :, :H, :W]

        '''fusion'''
        # fused = self.fuse(trans_back)
        # out = x + fused
        out = self.fuse_conv1(trans_back)
        out = torch.cat((out, y), 1) # 在第一维上对out和输入的复制y进行拼接(与论文图示一致)
        out = self.fuse_conv2(out)
        return out
    
# ---------------------------
# MobileViT backbone
# ---------------------------
class MobileViT(nn.Module):
    def __init__(self, dims, channels, mv2_exp, kernel_size=3, patch_size=(2,2), num_classes=1000):
        '''
        dims: 各个MobileViT block的transformer embedding dims
        channels: 各个block的输入输出通道数
        mv2_exp: Inverted Residual Block 中的 expand ratio
        kernel_size: MobileViT block中的卷积核大小
        patch_size: patch的高度和宽度
        num_classes: 图片的分类数
        '''
        super().__init__()
        ph, pw = patch_size

        L = [2, 4, 3] # 各个MoblileViT Block中的transformer的depth

        # 初始卷积
        self.conv1 = ConvBNAct(3, channels[0], k=3, s=2) # 输入通道数为3，因为图片一般就是三通道

        # layer 1
        self.layer1 = InvertedResidual(channels[0], channels[1], stride=1, expand_ratio=mv2_exp)

        # layer 2
        self.layer2 = nn.Sequential(
            InvertedResidual(channels[1], channels[2], stride=2, expand_ratio=mv2_exp),
                # stride = 2即进行了下采样，即论文图示中的向下箭头再跟2，可以看链接4
            InvertedResidual(channels[2], channels[3], stride=1, expand_ratio=mv2_exp),
            InvertedResidual(channels[2], channels[3], stride=1, expand_ratio=mv2_exp),
                # 这两个是相同的复制，与图示中MV2块上的"2x"一致
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

        # 最后的 1*1卷积 + 全局池化 + 全连接层
        '''
        self.head = nn.Sequential(
            ConvBNAct(channels[9], channels[10], k=1), 
                # 输出形状为(B, channels[10], H, W). 并且由于之前经过了5次下采样，这里的H和W应当是初始图片尺寸的1/32.
            nn.AdaptiveAvgPool2d(1),
                # 全局平均池化，参数1代表输出为1*1. 输出形状(B, channels[10], 1, 1)
            nn.Flatten(),
                # 默认从第1维到末尾展平，(B, channels[10], 1, 1) -> (B, channels[10])
                # 事实上在输入全连接层之前 PyTorch 会自动 flatten，因此这一个或许可以省略
            nn.Linear(channels[10], num_classes, bias=False)
                # (B, channels[10]) -> (B, num_classes)
        )
        '''
        # 为了输出最重要的倒数的两层feature，因此将上面的head拆分一下
        self.conv2 = ConvBNAct(channels[9], channels[10], k=1)
        self.avgpool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )
        self.classifier = nn.Linear(channels[10], num_classes, bias=False)

    def forward(self, x, is_feat=False):
        features = [] # 记录所有中间特征，仿照mobilenetv2
        # 由于事实上重要的只有最后两层feature，因此前面的可以基本省略，像这里就把layer1 ~ layer5中间的feature层全省略了

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

# 用于当前代码的mobilevit
def mobilevit(num_classes):
    # dims = [64, 80, 96]
    # channels = [16, 16, 24, 24, 48, 48, 64, 64, 80, 80, 320]
    # return MobileViT(dims, channels, mv2_exp=2, num_classes=num_classes)
    dims = [96, 120, 144]
    channels = [16, 32, 48, 48, 64, 64, 80, 80, 96, 96, 384]
    return MobileViT(dims, channels, mv2_exp=4, num_classes=num_classes)

# def mobilevit(num_classes):
#     # 调整 dims 和 channels，使模型更轻量化(针对CIFAR-100数据集)
#     dims = [48, 64, 80]  # Transformer embedding dims
#     channels = [16, 16, 24, 24, 32, 32, 48, 48, 64, 64, 256]  # 通道数
#     return MobileViT(dims, channels, mv2_exp=2, patch_size=(4, 4), num_classes=num_classes)