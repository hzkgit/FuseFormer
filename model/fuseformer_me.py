''' Fuseformer for Video Inpainting
'''
import numpy as np
import time
import math
from functools import reduce
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from core.spectral_norm import spectral_norm as _spectral_norm
import matplotlib.pyplot as plt
from torch import Tensor, LongTensor
from typing import Tuple, Optional


def visualize_img(tensor, savename):
    if len(tensor.shape) == 4:
        batch_size, channels, height, width = tensor.shape
        num_images = batch_size
    elif len(tensor.shape) == 5:
        batch1_size, batch2_size, channels, height, width = tensor.shape
        num_images = batch1_size * batch2_size
    else:
        raise ValueError("Unsupported tensor dimensions.")

    num_rows = num_images // 2
    if num_images % 2:
        num_rows += 1

    fig, axes = plt.subplots(num_rows, 2, figsize=(12, 6 * num_rows))

    for i in range(num_images):
        row = i // 2
        col = i % 2
        if len(tensor.shape) == 4:
            single_img = tensor[i]
        else:
            img_idx1 = i // batch2_size
            img_idx2 = i % batch2_size
            single_img = tensor[img_idx1, img_idx2]

        img_np = single_img.permute(1, 2, 0).numpy()
        axes[row, col].imshow(img_np)
        axes[row, col].axis('off')

    plt.tight_layout()
    plt.savefig(savename)
    plt.close()


class BaseNetwork(nn.Module):
    def __init__(self):
        super(BaseNetwork, self).__init__()

    def print_network(self):
        if isinstance(self, list):
            self = self[0]
        num_params = 0
        for param in self.parameters():
            num_params += param.numel()
        print('Network [%s] was created. Total number of parameters: %.1f million. '
              'To see the architecture, do print(network).' % (type(self).__name__, num_params / 1000000))

    def init_weights(self, init_type='normal', gain=0.02):
        '''
        initialize network's weights
        init_type: normal | xavier | kaiming | orthogonal
        https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/9451e70673400885567d08a9e97ade2524c700d0/models/networks.py#L39
        '''

        def init_func(m):
            classname = m.__class__.__name__
            if classname.find('InstanceNorm2d') != -1:
                if hasattr(m, 'weight') and m.weight is not None:
                    nn.init.constant_(m.weight.data, 1.0)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)
            elif hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, gain)  # 模块 m 的权重初始化为一个均值为 0.0、标准差为 gain 倍的正态分布的随机值
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'xavier_uniform':
                    nn.init.xavier_uniform_(m.weight.data, gain=1.0)
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data, gain=gain)
                elif init_type == 'none':  # uses pytorch's default init method
                    m.reset_parameters()
                else:
                    raise NotImplementedError(
                        'initialization method [%s] is not implemented' % init_type)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

        self.apply(init_func)

        # propagate to children
        for m in self.children():
            if hasattr(m, 'init_weights'):
                m.init_weights(init_type, gain)


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.group = [1, 2, 4, 8, 1]
        self.layers = nn.ModuleList([
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1, groups=1),  # 序号8的层
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(640, 512, kernel_size=3, stride=1, padding=1, groups=2),  # 640*w*h 使用512组640*3*3的卷积核 得到512*w'*h'
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(768, 384, kernel_size=3, stride=1, padding=1, groups=4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(640, 256, kernel_size=3, stride=1, padding=1, groups=8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 128, kernel_size=3, stride=1, padding=1, groups=1),
            nn.LeakyReLU(0.2, inplace=True)
        ])

    def forward(self, x):
        bt, c, h, w = x.size()
        h, w = h // 4, w // 4  # h 和 w 的值被调整为原来的四分之一（下采样，//返回结果为去下界的整数）
        out = x
        for i, layer in enumerate(self.layers):  # enumerate用于在遍历一个序列（如列表、元组、字符串等）的同时获取元素的索引（索引的起始值默认为0）
            if i == 8:
                x0 = out  # x0=bt*256*w*h
            if i > 8 and i % 2 == 0:  # 分组、重塑和连接，这样做的目的是为了减少参数量、降低计算复杂度
                g = self.group[(i - 8) // 2]
                x = x0.view(bt, g, -1, h, w)  # x=bt*2*128*w*h
                o = out.view(bt, g, -1, h, w)  # out=bt*384*w*h o=bt*2*192*w*h
                out = torch.cat([x, o], 2).view(bt, -1, h, w)  # bt*2*320*w*h  out=bt*640*h*w
            out = layer(out)
        return out


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x):
        """
        x: NHWC tensor
        """
        x = x.permute(0, 3, 1, 2)  # NCHW
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # NHWC

        return x


class Block(nn.Module):
    def __init__(self, dim,
                 num_heads=8, n_win=7, qk_dim=None, qk_scale=None,
                 topk=4,
                 mlp_ratio=4, mlp_dwconv=False,
                 side_dwconv=5, before_attn_dwconv=3, pre_norm=True, auto_pad=False):
        super().__init__()
        qk_dim = qk_dim or dim

        # modules
        if before_attn_dwconv > 0:
            self.pos_embed = nn.Conv2d(dim, dim, kernel_size=before_attn_dwconv, padding=1, groups=dim)
        else:
            self.pos_embed = lambda x: 0
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)  # important to avoid attention collapsing
        if topk > 0:
            self.attn = BiLevelRoutingAttention_nchw(dim=dim, num_heads=num_heads, n_win=n_win,
                                                     qk_scale=qk_scale,
                                                     topk=topk,
                                                     side_dwconv=side_dwconv,
                                                     auto_pad=auto_pad)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = nn.Sequential(nn.Linear(dim, int(mlp_ratio * dim)),
                                 DWConv(int(mlp_ratio * dim)) if mlp_dwconv else nn.Identity(),
                                 nn.GELU(),
                                 nn.Linear(int(mlp_ratio * dim), dim)
                                 )
        self.use_layer_scale = False
        self.pre_norm = pre_norm

    def forward(self, x):
        """
        x: NCHW tensor
        """
        # conv pos embedding
        x = x + self.pos_embed(x)  # 注意力0层,(16,64,56,56)+(16,64,56,56)=>(16,64,56,56)
        # permute to NHWC tensor for attention & mlp
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = x + self.attn(self.norm1(x))  # (N, H, W, C)
        x = x + self.mlp(self.norm2(x))  # (N, H, W, C)  # 输入x:(16,56,56,64) 经过一次层归一化、经过一个2层全连接得到x:(16,56,56,64)
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)  # =>(16,64,56,56)
        return x


def _grid2seq(x: Tensor, region_size: Tuple[int], num_heads: int):
    """
    Args:
        x: BCHW tensor
        region size: int
        num_heads: number of attention heads
    Return:
        out: rearranged x, has a shape of (bs, nhead, nregion, reg_size, head_dim)
        region_h, region_w: number of regions per col/row
    """
    B, C, H, W = x.size()
    region_h, region_w = H // region_size[0], W // region_size[1]
    x = x.view(B, num_heads, C // num_heads, region_h, region_size[0], region_w, region_size[1])
    x = torch.einsum('bmdhpwq->bmhwpqd', x).flatten(2, 3).flatten(-3, -2)  # (bs, nhead, nregion, reg_size, head_dim)
    return x, region_h, region_w


def _seq2grid(x: Tensor, region_h: int, region_w: int, region_size: Tuple[int]):
    """
    Args:
        x: (bs, nhead, nregion, reg_size^2, head_dim)
    Return:
        x: (bs, C, H, W)
    """
    bs, nhead, nregion, reg_size_square, head_dim = x.size()
    x = x.view(bs, nhead, region_h, region_w, region_size[0], region_size[1], head_dim)
    x = torch.einsum('bmhwpqd->bmdhpwq', x).reshape(bs, nhead * head_dim,
                                                    region_h * region_size[0], region_w * region_size[1])
    return x


def regional_routing_attention_torch(
        query: Tensor, key: Tensor, value: Tensor, scale: float,
        region_graph: LongTensor, region_size: Tuple[int],
        kv_region_size: Optional[Tuple[int]] = None,
        auto_pad=True) -> Tensor:
    """
    Args:
        query, key, value: (B, C, H, W) tensor
        scale: the scale/temperature for dot product attention
        region_graph: (B, nhead, h_q*w_q, topk) tensor, topk <= h_k*w_k
        region_size: region/window size for queries, (rh, rw)
        key_region_size: optional, if None, key_region_size=region_size
        auto_pad: required to be true if the input sizes are not divisible by the region_size
    Return:
        output: (B, C, H, W) tensor
        attn: (bs, nhead, q_nregion, reg_size, topk*kv_region_size) attention matrix
    """
    kv_region_size = kv_region_size or region_size
    bs, nhead, q_nregion, topk = region_graph.size()

    # Auto pad to deal with any input size
    q_pad_b, q_pad_r, kv_pad_b, kv_pad_r = 0, 0, 0, 0
    if auto_pad:
        _, _, Hq, Wq = query.size()
        q_pad_b = (region_size[0] - Hq % region_size[0]) % region_size[0]
        q_pad_r = (region_size[1] - Wq % region_size[1]) % region_size[1]
        if (q_pad_b > 0 or q_pad_r > 0):
            query = F.pad(query, (0, q_pad_r, 0, q_pad_b))  # zero padding

        _, _, Hk, Wk = key.size()
        kv_pad_b = (kv_region_size[0] - Hk % kv_region_size[0]) % kv_region_size[0]
        kv_pad_r = (kv_region_size[1] - Wk % kv_region_size[1]) % kv_region_size[1]
        if (kv_pad_r > 0 or kv_pad_b > 0):
            key = F.pad(key, (0, kv_pad_r, 0, kv_pad_b))  # zero padding
            value = F.pad(value, (0, kv_pad_r, 0, kv_pad_b))  # zero padding

    # to sequence format, i.e. (bs, nhead, nregion, reg_size, head_dim)
    query, q_region_h, q_region_w = _grid2seq(query, region_size=region_size, num_heads=nhead)
    key, _, _ = _grid2seq(key, region_size=kv_region_size, num_heads=nhead)
    value, _, _ = _grid2seq(value, region_size=kv_region_size, num_heads=nhead)

    # gather key and values.
    # TODO: is seperate gathering slower than fused one (our old version) ?
    # torch.gather does not support broadcasting, hence we do it manually
    bs, nhead, kv_nregion, kv_region_size, head_dim = key.size()
    broadcasted_region_graph = region_graph.view(bs, nhead, q_nregion, topk, 1, 1). \
        expand(-1, -1, -1, -1, kv_region_size, head_dim)
    key_g = torch.gather(key.view(bs, nhead, 1, kv_nregion, kv_region_size, head_dim). \
                         expand(-1, -1, query.size(2), -1, -1, -1), dim=3,
                         index=broadcasted_region_graph)  # (bs, nhead, q_nregion, topk, kv_region_size, head_dim)
    value_g = torch.gather(value.view(bs, nhead, 1, kv_nregion, kv_region_size, head_dim). \
                           expand(-1, -1, query.size(2), -1, -1, -1), dim=3,
                           index=broadcasted_region_graph)  # (bs, nhead, q_nregion, topk, kv_region_size, head_dim)

    # token-to-token attention
    # (bs, nhead, q_nregion, reg_size, head_dim) @ (bs, nhead, q_nregion, head_dim, topk*kv_region_size)
    # -> (bs, nhead, q_nregion, reg_size, topk*kv_region_size)
    # TODO: mask padding region
    attn = (query * scale) @ key_g.flatten(-3, -2).transpose(-1, -2)
    attn = torch.softmax(attn, dim=-1)
    # (bs, nhead, q_nregion, reg_size, topk*kv_region_size) @ (bs, nhead, q_nregion, topk*kv_region_size, head_dim)
    # -> (bs, nhead, q_nregion, reg_size, head_dim)
    output = attn @ value_g.flatten(-3, -2)

    # to BCHW format
    output = _seq2grid(output, region_h=q_region_h, region_w=q_region_w, region_size=region_size)

    # remove paddings if needed
    if auto_pad and (q_pad_b > 0 or q_pad_r > 0):
        output = output[:, :, :Hq, :Wq]

    return output, attn


class BiLevelRoutingAttention_nchw(nn.Module):
    """Bi-Level Routing Attention that takes nchw input

    Compared to legacy version, this implementation:
    * removes unused args and components
    * uses nchw input format to avoid frequent permutation

    When the size of inputs is not divisible by the region size, there is also a numerical difference
    than legacy implementation, due to:
    * different way to pad the input feature map (padding after linear projection)
    * different pooling behavior (count_include_pad=False)

    Current implementation is more reasonable, hence we do not keep backward numerical compatiability
    """

    def __init__(self, dim, num_heads=8, n_win=7, qk_scale=None, topk=4, side_dwconv=3, auto_pad=False,
                 attn_backend='torch'):
        super().__init__()
        # local attention setting
        self.dim = dim
        self.num_heads = num_heads
        assert self.dim % num_heads == 0, 'dim must be divisible by num_heads!'
        self.head_dim = self.dim // self.num_heads
        self.scale = qk_scale or self.dim ** -0.5  # NOTE: to be consistent with old models.

        ################side_dwconv (i.e. LCE in Shunted Transformer)###########
        self.lepe = nn.Conv2d(dim, dim, kernel_size=side_dwconv, stride=1, padding=side_dwconv // 2,
                              groups=dim) if side_dwconv > 0 else \
            lambda x: torch.zeros_like(x)

        ################ regional routing setting #################
        self.topk = topk
        self.n_win = n_win  # number of windows per row/col

        ##########################################

        self.qkv_linear = nn.Conv2d(self.dim, 3 * self.dim, kernel_size=1)
        self.output_linear = nn.Conv2d(self.dim, self.dim, kernel_size=1)

        if attn_backend == 'torch':
            self.attn_fn = regional_routing_attention_torch
        else:
            raise ValueError('CUDA implementation is not available yet. Please stay tuned.')

    def forward(self, x: Tensor, ret_attn_mask=False):
        """
        Args:
            x: NCHW tensor, better to be channel_last (https://pytorch.org/tutorials/intermediate/memory_format_tutorial.html)
        Return:
            NCHW tensor
        """
        N, T, C, H, W = x.size()
        region_size = (H // self.n_win, W // self.n_win)  # (64/7,64/7)=(9,9)

        # STEP 1: linear projection
        x_flattened = x.view(N * T, C, H, W)
        qkv = self.qkv_linear.forward(x_flattened)
        q, k, v = qkv.chunk(3, dim=1)  # 沿着指定的维度分割成3个块 (80,64,64,64)

        # STEP 2: region-to-region routing
        # NOTE: ceil_mode=True, count_include_pad=False = auto padding
        # NOTE: gradients backward through token-to-token attention. See Appendix A for the intuition.
        q_r = F.avg_pool2d(q.detach(), kernel_size=region_size, ceil_mode=True,
                           count_include_pad=False)  # detach()：拷贝一份张量，但不影响原始张量  avg_pool2d：平均池化
        k_r = F.avg_pool2d(k.detach(), kernel_size=region_size, ceil_mode=True,
                           count_include_pad=False)  # (80,64,64,64)=>(80,64,8,8)
        q_r: Tensor = q_r.permute(0, 2, 3, 1).flatten(1, 2)  # =>(80,8,8,64)=>(80,64,64)   n(hw)c
        k_r: Tensor = k_r.flatten(2, 3)  # nc(hw)  =>(80,64,64)
        a_r = q_r @ k_r  # n(hw)(hw), adj matrix of regional graph  (80,64,64)
        _, idx_r = torch.topk(a_r, k=self.topk, dim=-1)  # n(hw)k  =>(80,64,4)  # 只获取qi*ki值前top4的注意力信息
        idx_r: LongTensor = idx_r.unsqueeze_(1).expand(-1, self.num_heads, -1, -1)  # =>(80,1,64,4)=>(80,8,64,4)

        # STEP 3: token to token attention (non-parametric function)
        output, attn_mat = self.attn_fn(query=q, key=k, value=v, scale=self.scale,
                                        region_graph=idx_r, region_size=region_size)

        output = output + self.lepe(v)  # ncHW
        output = self.output_linear(output)  # ncHW

        # 恢复形状
        output = output.view(N, T, C, H, W)

        if ret_attn_mask:
            return output, attn_mat

        return output


class InpaintGenerator(BaseNetwork):
    def __init__(self, init_weights=True):
        super(InpaintGenerator, self).__init__()
        channel = 256
        hidden = 512
        stack_num = 8
        num_head = 4
        kernel_size = (7, 7)
        padding = (3, 3)
        stride = (3, 3)
        output_size = (60, 108)
        blocks = []
        dropout = 0.
        n_vecs = 1
        for i, d in enumerate(kernel_size):
            n_vecs *= int((output_size[i] + 2 * padding[i] - (d - 1) - 1) / stride[i] + 1)
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=embed_dim[i],  # * 代表传入参数是多个
                        topk=topks[i],  # topks=[1, 4, 16, -2]
                        num_heads=nheads[i],
                        n_win=n_win,
                        qk_dim=qk_dims[i],
                        qk_scale=qk_scale,
                        mlp_ratio=mlp_ratios[i],
                        mlp_dwconv=mlp_dwconv,
                        side_dwconv=side_dwconv,
                        before_attn_dwconv=before_attn_dwconv,
                        pre_norm=pre_norm,
                        auto_pad=auto_pad) for j in range(depth[i])],  # 注意这里有一个遍历depth的操作 [4, 4, 18, 4]
            )
        self.transformer = nn.Sequential(*blocks)  # 在 nn.Sequential 中，每个模块的输出会自动成为下一个模块的输入，直到最后一个模块的输出被返回
        self.ss = SoftSplitNew(channel // 2, hidden, kernel_size, stride, padding, dropout=dropout)
        # self.add_pos_emb = AddPosEmb(n_vecs, hidden)  # n_vecs:720
        self.sc = SoftCompNew(channel // 2, hidden, output_size, kernel_size, stride, padding)

        self.encoder = Encoder()

        # decoder: decode frames from features
        self.decoder = nn.Sequential(
            deconv(channel // 2, 128, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            deconv(64, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1)
        )

        if init_weights:
            self.init_weights()

    def forward(self, masked_frames):
        # extracting features
        b, t, c, h, w = masked_frames.size()  # b是多少个视频 t是同一视频多少帧  masked_frames:([8, 2, 3, 240, 432])
        # visualize_img(masked_frames.cpu().detach(), "{}/input.png".format("/root/autodl-tmp/FuseFormer-master/visual"))
        enc_feat = self.encoder(masked_frames.view(b * t, c, h, w))  # enc_feat:([16, 128, 60, 108])  (b*t,c,h,w)
        _, c, h, w = enc_feat.size()

        trans_feat = self.ss(enc_feat, b)  # 输入(B*T,C,H,W)=>输出(B,T,H,W,C)
        # trans_feat = self.add_pos_emb(trans_feat)  # 输入trans_feat([8, 1440, 512]) (b,?,c)

        trans_feat = self.transformer(trans_feat)  # 输入trans_feat([8, 1440, 512])

        trans_feat = self.sc(trans_feat, t)  # 输入(B,T,H,W,C) 输出(B*T,C,H,W)

        enc_feat = enc_feat + trans_feat  # 输入trans_feat([16, 128, 60, 108]) enc_feat:([16, 128, 60, 108])
        output = self.decoder(enc_feat)  # 输入enc_feat:([16, 128, 60, 108])=>([16, 3, 240, 432])
        output = torch.tanh(output)  # tanh 的输出值范围在 -1 和 1 之间  output([16, 3, 240, 432])
        # visualize_img(output.cpu().detach(), "{}/output.png".format("/root/autodl-tmp/FuseFormer-master/visual"))
        return output


class deconv(nn.Module):
    def __init__(self, input_channel, output_channel, kernel_size=3, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(input_channel, output_channel,
                              kernel_size=kernel_size, stride=1, padding=padding)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='bilinear',
                          align_corners=True)
        return self.conv(x)


# #############################################################################
# ############################# Transformer  ##################################
# #############################################################################


class Attention(nn.Module):
    """
    Compute 'Scaled Dot Product Attention
    """

    def __init__(self, p=0.1):
        super(Attention, self).__init__()
        self.dropout = nn.Dropout(p=p)

    def forward(self, query, key, value, m=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
        if m is not None:
            scores.masked_fill_(m, -1e9)
        p_attn = F.softmax(scores, dim=-1)
        p_attn = self.dropout(p_attn)
        p_val = torch.matmul(p_attn, value)
        return p_val, p_attn


class AddPosEmb(nn.Module):
    def __init__(self, n, c):
        super(AddPosEmb, self).__init__()
        # 使用正态分布初始化位置嵌入参数，而不是直接使用序号
        # https://www.tiangong.cn/result/24e8e7bc-1a76-4ecc-a8bc-a9acca7f41cc
        self.pos_emb = nn.Parameter(torch.zeros(1, 1, n, c).float().normal_(mean=0, std=0.02), requires_grad=True)
        self.num_vecs = n  # 720

    def forward(self, x):
        b, n, c = x.size()  # x:([8, 1440, 512])
        x = x.view(b, -1, self.num_vecs, c)  # x:([8, 2, 720, 512])  # TODO 这里报错了
        x = x + self.pos_emb  # pos_emb：定义位置嵌入参数，用来提供序列中元素的顺序信息
        x = x.view(b, n, c)  # 再还原回输入时的形状 x:([8, 1440, 512])
        return x


class SoftSplit(nn.Module):
    def __init__(self, channel, hidden, kernel_size, stride, padding, dropout=0.1):
        super(SoftSplit, self).__init__()
        self.kernel_size = kernel_size
        self.t2t = nn.Unfold(kernel_size=kernel_size, stride=stride, padding=padding)
        c_in = reduce((lambda x, y: x * y), kernel_size) * channel
        self.embedding = nn.Linear(c_in, hidden)  # hidden=512
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, b):
        feat = self.t2t(
            x)  # 卷积截取后展平（特征图中2*2的一个区域就变成了一个一维向量）x:([16, 128, 60, 108]) feat:([16, 6272, 720])  # 这里能变成720就是对的
        feat = feat.permute(0, 2, 1)  # 交换顺序后，所有2*2小格的第i(1,2,3,4)个格子的元素就保存在一个向量中了 feat:([16, 720, 6272])
        feat = self.embedding(feat)  # 将这些向量生维到512 feat:([16, 720, 512])
        feat = feat.view(b, -1, feat.size(2))  # feat:([8, 1440, 512])  (b,?,c)
        feat = self.dropout(feat)
        return feat


# ProPainter的SS，支持输入(B*T,C,H,W)=>输出(B,T,H,W,C)
class SoftSplitNew(nn.Module):
    def __init__(self, channel, hidden, kernel_size, stride, padding):
        super(SoftSplitNew, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.t2t = nn.Unfold(kernel_size=kernel_size, stride=stride, padding=padding)
        c_in = reduce((lambda x, y: x * y), kernel_size) * channel
        self.embedding = nn.Linear(c_in, hidden)

    def forward(self, x, b, output_size):
        f_h = int((output_size[0] + 2 * self.padding[0] -
                   (self.kernel_size[0] - 1) - 1) / self.stride[0] + 1)
        f_w = int((output_size[1] + 2 * self.padding[1] -
                   (self.kernel_size[1] - 1) - 1) / self.stride[1] + 1)

        feat = self.t2t(x)  # 输入（B*T，C，H，W）=>(b*t,c,?)  (9,128,60,108)=>(9,6272,720)
        feat = feat.permute(0, 2, 1)  # =>(9,720,6272)  (b*t,?,c)
        # feat shape [b*t, num_vec, ks*ks*c]
        feat = self.embedding(feat)  # =>(9,720,512)
        # feat shape after embedding [b, t*num_vec, hidden]
        feat = feat.view(b, -1, f_h, f_w, feat.size(2))  # feat(9,720,512)=>(1,9,20,36,512)
        return feat


class SoftComp(nn.Module):
    def __init__(self, channel, hidden, output_size, kernel_size, stride, padding):
        super(SoftComp, self).__init__()
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        c_out = reduce((lambda x, y: x * y), kernel_size) * channel  # reduce:求kernel_size的乘积,这里是7*7=49,49*128=6272
        self.embedding = nn.Linear(hidden, c_out)
        self.t2t = torch.nn.Fold(output_size=output_size, kernel_size=kernel_size, stride=stride, padding=padding)
        h, w = output_size
        self.bias = nn.Parameter(torch.zeros((channel, h, w), dtype=torch.float32), requires_grad=True)

    def forward(self, x, t):
        feat = self.embedding(x)  # (8, 1440, 512)=>(8,1440,6272)
        b, n, c = feat.size()
        feat = feat.view(b * t, -1, c).permute(0, 2, 1)  # t:2,(8,1440,6272)=>(16,720,6272)=>(16,6272,720)
        feat = self.t2t(feat) + self.bias[None]  # (16,128,60,108)+(1,128,60,108)=feat:(16, 128, 60, 108)
        return feat


# ProPainter的SS，支持输入(B,T,H,W,C) 输出(B*T,C,H,W)
class SoftCompNew(nn.Module):
    def __init__(self, channel, hidden, kernel_size, stride, padding):
        super(SoftCompNew, self).__init__()
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        c_out = reduce((lambda x, y: x * y), kernel_size) * channel
        self.embedding = nn.Linear(hidden, c_out)
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias_conv = nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1)

    def forward(self, x, t, output_size):
        b_, _, _, _, c_ = x.shape
        x = x.view(b_, -1, c_)  # 转为（B,?,C） (1,6480,512)
        feat = self.embedding(x)  # (1,6480,512)=>(1,6480,6272)
        b, _, c = feat.size()
        feat = feat.view(b * t, -1, c).permute(0, 2, 1)  # (1,6480,6272)=>(9,720,6272)=>(9,6272,720)  (B,?,C)=>(B,C,?)
        feat = F.fold(feat, output_size=output_size, kernel_size=self.kernel_size, stride=self.stride,
                      padding=self.padding)
        feat = self.bias_conv(feat)
        return feat


class MultiHeadedAttention(nn.Module):
    """
    Take in model size and number of heads.
    """

    def __init__(self, d_model, head, p=0.1):
        super().__init__()
        self.query_embedding = nn.Linear(d_model, d_model)
        self.value_embedding = nn.Linear(d_model, d_model)
        self.key_embedding = nn.Linear(d_model, d_model)
        self.output_linear = nn.Linear(d_model, d_model)  # nn.Linear:全连接层
        self.attention = Attention(p=p)
        self.head = head

    def forward(self, x):
        b, n, c = x.size()  # x=(8,1440,512)
        c_h = c // self.head  # c_h=512/4=128
        key = self.key_embedding(x)  # key=(8,1440,512)
        key = key.view(b, n, self.head, c_h).permute(0, 2, 1, 3)  # (8,1440,4,128)=>(8,4,1440,128)
        query = self.query_embedding(x)  # query=(8,1440,512)
        query = query.view(b, n, self.head, c_h).permute(0, 2, 1, 3)  # (8,4,1440,128)
        value = self.value_embedding(x)
        value = value.view(b, n, self.head, c_h).permute(0, 2, 1, 3)  # (8,4,1440,128)
        att, _ = self.attention(query, key, value)  # att=(8,4,1440,128)
        att = att.permute(0, 2, 1, 3).contiguous().view(b, n, c)  # (8,1440,4,128)=>(8,1440,512)
        output = self.output_linear(att)  # output=(8,1440,512)
        return output


class FeedForward(nn.Module):
    def __init__(self, d_model, p=0.1):
        super(FeedForward, self).__init__()
        # We set d_ff as a default to 2048
        self.conv = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(p=p),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(p=p))

    def forward(self, x):
        x = self.conv(x)
        return x


class FusionFeedForward(nn.Module):
    def __init__(self, d_model, p=0.1, n_vecs=None, t2t_params=None):
        super(FusionFeedForward, self).__init__()
        # We set d_ff as a default to 1960
        hd = 1960
        self.conv1 = nn.Sequential(
            nn.Linear(d_model, hd))  # Sequential((0): Linear(in_features=512, out_features=1960, bias=True))
        self.conv2 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Dropout(p=p),
            nn.Linear(hd, d_model),
            nn.Dropout(p=p))
        assert t2t_params is not None and n_vecs is not None
        tp = t2t_params.copy()
        self.fold = nn.Fold(**tp)
        del tp['output_size']
        self.unfold = nn.Unfold(**tp)
        self.n_vecs = n_vecs

    def forward(self, x):
        x = self.conv1(x)  # (8,1440,512)=>(8,1440,1960) torch.nn.Linear层默认应用于输入张量的最后一个维度
        b, n, c = x.size()
        # new_ones:快速创建一个全1的张量 (8,1440,49)=>(16,720,49)=>(16,49,720) 这个是为了归一化用的
        normalizer = x.new_ones(b, n, 49).view(-1, self.n_vecs, 49).permute(0, 2, 1)
        # self.fold(output_size=(60, 108), kernel_size=(7, 7), dilation=1(无空洞,如果dilation大于1，则在卷积核元素之间会插入零), padding=(3, 3), stride=(3, 3)):
        # self.fold(normalizer):将多个Patch融合 (16,49,720)=>(16,1,60,108)
        # self.fold(...):将多个Patch融合 (16,720,1970)=>(16,1970,720)=>(16,40,60,108)
        # 两个融合后结果做一下归一化
        # self.unfold(...):将归一化后结果再分割成Patch
        x = self.unfold(self.fold(x.view(-1, self.n_vecs, c).permute(0, 2, 1)) / self.fold(normalizer)).permute(0, 2,
                                                                                                                1).contiguous().view(
            b, n, c)
        x = self.conv2(x)  # 还原为输入是形状(8,1440,1960)=>(8,1440,512)
        return x


class TransformerBlock(nn.Module):
    """
    Transformer = MultiHead_Attention + Feed_Forward with sublayer connection
    """

    def __init__(self, hidden=128, num_head=4, dropout=0.1, n_vecs=None, t2t_params=None):
        super().__init__()
        self.attention = MultiHeadedAttention(d_model=hidden, head=num_head, p=dropout)
        self.ffn = FusionFeedForward(hidden, p=dropout, n_vecs=n_vecs, t2t_params=t2t_params)
        self.norm1 = nn.LayerNorm(hidden)  # 层归一化，一种批量归一化技术
        self.norm2 = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, input):
        x = self.norm1(input)  # x:([8, 1440, 512])  层归一化 （8个批次，每个批次1440个token，每个token长度为512）
        x = input + self.dropout(self.attention(x))  # 多头注意力
        y = self.norm2(x)  # 输入x:(8,1440,512)
        x = x + self.ffn(y)  # x:(8,1440,512)+y:(8,1440,512)=>x:(8,1440,512)
        return x


# ######################################################################
# ######################################################################


class Discriminator(BaseNetwork):
    def __init__(self, in_channels=3, use_sigmoid=False, use_spectral_norm=True, init_weights=True):
        super(Discriminator, self).__init__()
        self.use_sigmoid = use_sigmoid
        nf = 32

        self.conv = nn.Sequential(
            spectral_norm(
                nn.Conv3d(in_channels=in_channels, out_channels=nf * 1, kernel_size=(3, 5, 5), stride=(1, 2, 2),
                          padding=1, bias=not use_spectral_norm), use_spectral_norm),
            # nn.InstanceNorm2d(64, track_running_stats=False),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv3d(nf * 1, nf * 2, kernel_size=(3, 5, 5), stride=(1, 2, 2), padding=(1, 2, 2),
                                    bias=not use_spectral_norm), use_spectral_norm),
            # nn.InstanceNorm2d(128, track_running_stats=False),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv3d(nf * 2, nf * 4, kernel_size=(3, 5, 5), stride=(1, 2, 2), padding=(1, 2, 2),
                                    bias=not use_spectral_norm), use_spectral_norm),
            # nn.InstanceNorm2d(256, track_running_stats=False),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv3d(nf * 4, nf * 4, kernel_size=(3, 5, 5), stride=(1, 2, 2), padding=(1, 2, 2),
                                    bias=not use_spectral_norm), use_spectral_norm),
            # nn.InstanceNorm2d(256, track_running_stats=False),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv3d(nf * 4, nf * 4, kernel_size=(3, 5, 5), stride=(1, 2, 2), padding=(1, 2, 2),
                                    bias=not use_spectral_norm), use_spectral_norm),
            # nn.InstanceNorm2d(256, track_running_stats=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(nf * 4, nf * 4, kernel_size=(3, 5, 5), stride=(1, 2, 2), padding=(1, 2, 2))
        )

        if init_weights:
            self.init_weights()

    def forward(self, xs):
        # T, C, H, W = xs.shape
        xs_t = torch.transpose(xs, 0, 1)  # (16,3,240,432)=>(3, 16, 240, 432)
        xs_t = xs_t.unsqueeze(0)  # B, C, T, H, W  (1, 3, 16, 240, 432)
        feat = self.conv(xs_t)  # feat=(1, 128, 16, 4, 7)
        if self.use_sigmoid:  # use_sigmoid=False
            feat = torch.sigmoid(feat)  # 归一化到0到1之间
        out = torch.transpose(feat, 1, 2)  # B, T, C, H, W   out=(1, 16, 128, 4, 7)
        return out


def spectral_norm(module, mode=True):
    if mode:
        return _spectral_norm(module)  # 谱归一化，限制函数变化的剧烈程度，提高模型的稳定性和性能
    return module