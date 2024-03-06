# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.

import math
import numpy as np
import torch
import torch.fft
import torch.nn as nn
import torch.nn.functional as F

# Bring your packages onto the path
import sys

# from afno.afno2d import AFNO2D
# from afno.bfno2d import BFNO2D
# from afno.ls import AttentionLS
# from afno.sa import SelfAttention
# from afno.gfn import GlobalFilter


import math
import logging
from functools import partial
from collections import OrderedDict
# from copy import Error, deepcopy
# from re import S
# from numpy.lib.arraypad import pad
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
# from timm.models.layers import  to_2tuple, trunc_normal_
# import torch.fft
from torch.nn.modules.container import Sequential
# from main_afnonet import get_args
# from torch.utils.checkpoint import checkpoint_sequential
from models.filter_networks import LReLu_regular, LReLu_torch
# from models.filter_networks import LReLu
# from models.cno import CNOBlock
from einops import rearrange, repeat
from einops.layers.torch import Rearrange




ACTIVATION = {'gelu':nn.GELU(),'tanh':nn.Tanh(),'sigmoid':nn.Sigmoid(),'relu':nn.ReLU(),'leaky_relu':nn.LeakyReLU(0.1),'softplus':nn.Softplus(),'ELU':nn.ELU(),'silu':nn.SiLU()}




class CNOBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 in_size,
                 out_size,
                 cutoff_den=2.0001,
                 conv_kernel=3,
                 filter_size=6,
                 lrelu_upsampling=2,
                 half_width_mult=0.8,
                 radial=False,
                 activation='cno_lrelu'
                 ):
        super(CNOBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.in_size = in_size
        self.out_size = out_size
        self.conv_kernel = conv_kernel

        # ---------- Filter properties -----------
        self.citically_sampled = False  # We use w_c = s/2.0001 --> NOT critically sampled

        if cutoff_den == 2.0:
            self.citically_sampled = True
        self.in_cutoff = self.in_size / cutoff_den
        self.out_cutoff = self.out_size / cutoff_den

        self.in_halfwidth = half_width_mult * self.in_size - self.in_size / cutoff_den
        self.out_halfwidth = half_width_mult * self.out_size - self.out_size / cutoff_den

        # -----------------------------------------

        # We apply Conv -> BN (optional) -> Activation
        # Up/Downsampling happens inside Activation

        pad = (self.conv_kernel - 1) // 2
        self.convolution = torch.nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels,
                                           kernel_size=self.conv_kernel,
                                           padding=pad)


        if activation == "cno_lrelu":
            # self.activation = LReLu(in_channels=self.in_channels,  # In _channels is not used in these settings
            #                         out_channels=self.out_channels,
            #                         in_size=self.in_size,
            #                         out_size=self.out_size,
            #                         in_sampling_rate=self.in_size,
            #                         out_sampling_rate=self.out_size,
            #                         in_cutoff=self.in_cutoff,
            #                         out_cutoff=self.out_cutoff,
            #                         in_half_width=self.in_halfwidth,
            #                         out_half_width=self.out_halfwidth,
            #                         filter_size=filter_size,
            #                         lrelu_upsampling=lrelu_upsampling,
            #                         is_critically_sampled=self.citically_sampled,
            #                         use_radial_filters=False)
            raise NotImplementedError
        elif activation == "cno_lrelu_torch":
            self.activation = LReLu_torch(in_channels=self.out_channels,  # In _channels is not used in these settings
                                          out_channels=self.out_channels,
                                          in_size=self.in_size,
                                          out_size=self.out_size,
                                          in_sampling_rate=self.in_size,
                                          out_sampling_rate=self.out_size)
        elif activation == "lrelu":
            self.activation = LReLu_regular(in_channels=self.in_channels,  # In _channels is not used in these settings
                                            out_channels=self.out_channels,
                                            in_size=self.in_size,
                                            out_size=self.out_size,
                                            in_sampling_rate=self.in_size,
                                            out_sampling_rate=self.out_size)
        else:
            raise ValueError("Please specify different activation function")


    def filter_frequency(self, x, K):
        """
           Apply low-pass filter to a tensor with shape (B, C, M, M).

           Parameters:
               tensor (torch.Tensor): Input tensor of shape (B, C, M, M).
               K (int): The ratio of frequency components to retain.

           Returns:
               torch.Tensor: The filtered tensor.
           """
        # Step 1: FFT
        fft_tensor = torch.fft.fft2(x)

        # Step 2: Create low-pass filter mask
        B, C, M, _ = x.shape
        mask = torch.zeros((M, M), dtype=torch.bool).to(x.device)
        cutoff = M // K  # Define the cutoff frequency based on the ratio K
        mask[:cutoff, :cutoff] = 1  # Retain 1/K of the frequency components
        mask = mask[None, None, :, :]  # Adjust dimensions to match the tensor (B, C, M, M)
        mask = mask.expand(B, C, M, M)  # Expand mask to match tensor dimensions

        # Step 3: Apply filter
        filtered_fft_tensor = fft_tensor * mask

        # Step 4: Inverse FFT
        filtered_tensor = torch.fft.ifft2(filtered_fft_tensor).real  # Get the real part after inverse FFT

        return filtered_tensor

    def forward(self, x):
        x = self.filter_frequency(x, self.conv_kernel)
        x = self.convolution(x)
        return self.activation(x)


class AFNO2D(nn.Module):
    """
    hidden_size: channel dimension size
    num_blocks: how many blocks to use in the block diagonal weight matrices (higher => less complexity but less parameters)
    sparsity_threshold: lambda for softshrink
    hard_thresholding_fraction: how many frequencies you want to completely mask out (lower => hard_thresholding_fraction^2 less FLOPs)
    """
    def __init__(self, width = 32, in_timesteps = 10, out_timesteps = 1, n_channels = 1, num_blocks=8, channel_first = False,sparsity_threshold=0.01, modes = 32,hard_thresholding_fraction=1, hidden_size_factor=1, act='gelu'):
        super().__init__()
        assert width % num_blocks == 0, f"hidden_size {width} should be divisble by num_blocks {num_blocks}"

        # self.in_timesteps = in_timesteps
        # self.out_timesteps = out_timesteps
        # self.n_channels = n_channels

        self.hidden_size = width
        self.sparsity_threshold = sparsity_threshold
        self.num_blocks = num_blocks
        self.block_size = self.hidden_size // self.num_blocks
        self.channel_first = channel_first
        # self.hard_thresholding_fraction = hard_thresholding_fraction
        self.modes = modes
        self.hidden_size_factor = hidden_size_factor
        # self.scale = 0.02
        self.scale = 1 / (self.block_size * self.block_size * self.hidden_size_factor)

        self.act = ACTIVATION[act]

        self.w1 = nn.Parameter(self.scale * torch.rand(2, self.num_blocks, self.block_size, self.block_size * self.hidden_size_factor))
        self.b1 = nn.Parameter(self.scale * torch.rand(2, self.num_blocks, self.block_size * self.hidden_size_factor))
        self.w2 = nn.Parameter(self.scale * torch.rand(2, self.num_blocks, self.block_size * self.hidden_size_factor, self.block_size))
        self.b2 = nn.Parameter(self.scale * torch.rand(2, self.num_blocks, self.block_size))

    ### N, C, X, Y
    def forward(self, x, spatial_size=None):
        if self.channel_first:
            B, C, H, W = x.shape
            x = x.permute(0, 2, 3, 1)  ### ->N, X, Y, C
        else:
            B, H, W, C = x.shape
        # x = x.view(*x.shape[:-2], -1)  #### B, X, Y, T*C
        # grid = self.get_grid(x)
        # x = torch.cat((x, grid), dim=-1)  #### B, X, Y, T, C
        x_orig = x

        x = torch.fft.rfft2(x, dim=(1, 2), norm="ortho")
        # x = torch.fft.rfft2(x, dim=(1, 2))

        x = x.reshape(B, x.shape[1], x.shape[2], self.num_blocks, self.block_size)

        o1_real = torch.zeros([B, x.shape[1], x.shape[2], self.num_blocks, self.block_size * self.hidden_size_factor], device=x.device)
        o1_imag = torch.zeros([B, x.shape[1], x.shape[2], self.num_blocks, self.block_size * self.hidden_size_factor], device=x.device)
        o2_real = torch.zeros(x.shape, device=x.device)
        o2_imag = torch.zeros(x.shape, device=x.device)

        # total_modes = H*W // 2 + 1
        kept_modes = self.modes

        o1_real[:, :kept_modes, :kept_modes] = self.act(
            torch.einsum('...bi,bio->...bo', x[:, :kept_modes, :kept_modes].real, self.w1[0]) - \
            torch.einsum('...bi,bio->...bo', x[:, :kept_modes, :kept_modes].imag, self.w1[1]) + \
            self.b1[0]
        )

        o1_imag[:, :kept_modes, :kept_modes] = self.act(
            torch.einsum('...bi,bio->...bo', x[:, :kept_modes, :kept_modes].imag, self.w1[0]) + \
            torch.einsum('...bi,bio->...bo', x[:, :kept_modes, :kept_modes].real, self.w1[1]) + \
            self.b1[1]
        )

        o2_real[:, :kept_modes, :kept_modes] = (
            torch.einsum('...bi,bio->...bo', o1_real[:, :kept_modes, :kept_modes], self.w2[0]) - \
            torch.einsum('...bi,bio->...bo', o1_imag[:, :kept_modes, :kept_modes], self.w2[1]) + \
            self.b2[0]
        )

        o2_imag[:, :kept_modes, :kept_modes] = (
            torch.einsum('...bi,bio->...bo', o1_imag[:, :kept_modes, :kept_modes], self.w2[0]) + \
            torch.einsum('...bi,bio->...bo', o1_real[:, :kept_modes, :kept_modes], self.w2[1]) + \
            self.b2[1]
        )

        x = torch.stack([o2_real, o2_imag], dim=-1)
        ## for ab study
        # x = F.softshrink(x, lambd=self.sparsity_threshold)

        x = torch.view_as_complex(x)
        x = x.reshape(B, x.shape[1], x.shape[2], C)
        x = torch.fft.irfft2(x, s=(H, W), dim=(1, 2), norm="ortho")
        # x = torch.fft.irfft2(x, s=(H, W), dim=(1, 2))

        # x = x.reshape(B, N, C)
        # x = x.type(dtype)

        # x = x.reshape(*x.shape[: -1], self.out_timesteps, C)

        x = x + x_orig
        if self.channel_first:
            x = x.permute(0, 3, 1, 2)     ### N, C, X, Y

        return x

    # def get_grid(self, x):
    #     batchsize, size_x, size_y = x.shape[0], x.shape[1], x.shape[2]
    #     gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
    #     gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
    #     gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
    #     gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
    #     grid = torch.cat((gridx, gridy), dim=-1).to(x.device)
    #     return grid





_logger = logging.getLogger(__name__)




class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act='gelu', drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = ACTIVATION[act]
        self.fc2 = nn.Linear(hidden_features, out_features)
        # self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        # x = self.drop(x)
        x = self.fc2(x)
        # x = self.drop(x)
        return x


# class AdaptiveFourierNeuralOperator(nn.Module):
#     def __init__(self, dim, n_blocks = 4, h=14, w=8, bias=True, softshrink=0.1):
#         super().__init__()
#
#         self.hidden_size = dim
#         self.h = h
#         self.w = w
#         self.num_blocks = n_blocks
#         self.bias = bias
#         self.block_size = self.hidden_size // self.num_blocks
#         assert self.hidden_size % self.num_blocks == 0
#
#         self.scale = 0.02
#         self.w1 = torch.nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size, self.block_size))
#         self.b1 = torch.nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size))
#         self.w2 = torch.nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size, self.block_size))
#         self.b2 = torch.nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size))
#         self.relu = nn.ReLU()
#
#         if bias:
#             self.bias = nn.Conv1d(self.hidden_size, self.hidden_size, 1)
#         else:
#             self.bias = None
#
#         self.softshrink = softshrink
#
#     def multiply(self, input, weights):
#         return torch.einsum('...bd,bdk->...bk', input, weights)
#
#     def forward(self, x, spatial_size=None):
#         B, N, C = x.shape
#         if spatial_size is None:
#             a = b = int(math.sqrt(N))
#         else:
#             a, b = spatial_size
#
#         if self.bias:
#             bias = self.bias(x.permute(0, 2, 1)).permute(0, 2, 1)
#         else:
#             bias = torch.zeros(x.shape, device=x.device)
#
#         x = x.reshape(B, a, b, C).float()
#         x = torch.fft.rfft2(x, dim=(1, 2), norm='ortho')
#         x = x.reshape(B, x.shape[1], x.shape[2], self.num_blocks, self.block_size)
#
#         x_real_1 = F.relu(self.multiply(x.real, self.w1[0]) - self.multiply(x.imag, self.w1[1]) + self.b1[0])
#         x_imag_1 = F.relu(self.multiply(x.real, self.w1[1]) + self.multiply(x.imag, self.w1[0]) + self.b1[1])
#         x_real_2 = self.multiply(x_real_1, self.w2[0]) - self.multiply(x_imag_1, self.w2[1]) + self.b2[0]
#         x_imag_2 = self.multiply(x_real_1, self.w2[1]) + self.multiply(x_imag_1, self.w2[0]) + self.b2[1]
#
#         x = torch.stack([x_real_2, x_imag_2], dim=-1).float()
#         x = F.softshrink(x, lambd=self.softshrink) if self.softshrink else x
#
#         x = torch.view_as_complex(x)
#         x = x.reshape(B, x.shape[1], x.shape[2], self.hidden_size)
#         x = torch.fft.irfft2(x, s=(a, b), dim=(1, 2), norm='ortho')
#         x = x.reshape(B, N, C)
#
#         return x + bias
#

class Block(nn.Module):
    def __init__(self, mixing_type = 'afno', double_skip = True, width = 32, n_blocks = 4, mlp_ratio=1., channel_first = True, modes = 32, drop=0., drop_path=0., act='gelu', h=14, w=8,):
        super().__init__()
        # self.norm1 = norm_layer(width)
        # self.norm1 = torch.nn.LayerNorm([width])
        self.norm1 = torch.nn.GroupNorm(8, width)
        # self.norm1 = torch.nn.InstanceNorm2d(width,affine=True,track_running_stats=False)
        self.width = width
        self.modes = modes
        self.act = ACTIVATION[act]

        if mixing_type == "afno":
            self.filter = AFNO2D(width = width, num_blocks=n_blocks, sparsity_threshold=0.01, channel_first = channel_first, modes = modes,
                                 hard_thresholding_fraction=1, hidden_size_factor=1, act=act)
            # self.filter = AdaptiveFourierNeuralOperator(dim, h=h, w=w)
        # elif args.mixing_type == "bfno":
        #     self.filter = BFNO2D(hidden_size=768, num_blocks=8, hard_thresholding_fraction=1)
        # elif args.mixing_type == "sa":
        #     self.filter = SelfAttention(dim=768, h=14, w=8)
        # if args.mixing_type == "gfn":
        #     self.filter = GlobalFilter(dim=768, h=14, w=8)
        # elif args.mixing_type == "ls":
        #     self.filter = AttentionLS(dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,rpe=False, nglo=1, dp_rank=2, w=2)

        # self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        # self.drop_path = nn.Identity()
        # self.norm2 = norm_layer(width)
        self.norm2 = torch.nn.GroupNorm(8, width)
        # self.norm2 = torch.nn.InstanceNorm2d(width,affine=True,track_running_stats=False)



        mlp_hidden_dim = int(width * mlp_ratio)
        # self.mlp = Mlp(in_features=width, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels=width, out_channels=mlp_hidden_dim, kernel_size=1, stride=1),
            self.act,
            # nn.Conv2d(in_channels=mlp_hidden_dim, out_channels=mlp_hidden_dim, kernel_size=1, stride=1),
            # nn.GELU(),
            nn.Conv2d(in_channels=mlp_hidden_dim, out_channels=width, kernel_size=1, stride=1),
        )

        self.double_skip = double_skip

    def forward(self, x):
        residual = x
        x = self.norm1(x)
        x = self.filter(x)


        if self.double_skip:
            x = x + residual
            residual = x

        # x = self.mlp(x.permute(0,2,3,1)).permute(0,3,1,2)
        x = self.norm2(x)
        x = self.mlp(x)

        # x = self.drop_path(x)
        x = x + residual

        return x


class CNOPatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, out_dim=128,act='cno_lrelu_torch'):
        super().__init__()
        # img_size = to_2tuple(img_size)
        # patch_size = to_2tuple(patch_size)
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.out_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.out_dim = out_dim
        self.act = ACTIVATION['gelu']
        # if act == 'cno_lrelu_torch':
        self.act_patching = LReLu_torch(embed_dim, embed_dim, self.out_size[0],self.out_size[0], self.out_size[0], self.out_size[1])


        self.proj = nn.Sequential(
            # CNOBlock(in_chans,embed_dim,self.out_size[0],self.out_size[0],cutoff_den=2.0001,conv_kernel=patch_size[0],activation=act),
            nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size),
            self.act_patching,
            # self.act,
            nn.Conv2d(embed_dim, out_dim, kernel_size=1, stride=1)
        )

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        # x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x



class TimeAggregator(nn.Module):
    def __init__(self, n_channels, n_timesteps, out_channels, type='mlp'):
        super(TimeAggregator, self).__init__()
        self.n_channels = n_channels
        self.n_timesteps = n_timesteps
        self.out_channels = out_channels
        self.type = type
        if self.type == 'mlp':
            self.w = nn.Parameter(1/(n_timesteps * out_channels**0.5) *torch.randn(n_timesteps, out_channels, out_channels),requires_grad=True)   # initialization could be tuned
        elif self.type == 'exp_mlp':
            self.w = nn.Parameter(1/(n_timesteps * out_channels**0.5) *torch.randn(n_timesteps, out_channels, out_channels),requires_grad=True)   # initialization could be tuned
            self.gamma = nn.Parameter(2**torch.linspace(-10,10, out_channels).unsqueeze(0),requires_grad=True)  # 1, C
    ##  B, X, Y, T, C
    def forward(self, x):
        if self.type == 'mlp':
            x = torch.einsum('tij, ...ti->...j', self.w, x)
        elif self.type == 'exp_mlp':
            t = torch.linspace(0, 1, x.shape[-2]).unsqueeze(-1).to(x.device) # T, 1
            t_embed = torch.cos(t @ self.gamma)
            x = torch.einsum('tij,...ti->...j', self.w, x * t_embed)

        return x








'''
    AFNO version. 12.6
    summary: legacy, using time, channel mixing
'''
# class AFNONet(nn.Module):
#     def __init__(self, img_size=224, patch_size=16, mixing_type = 'afno',in_channels=1, out_channels = 3, in_timesteps=1, out_timesteps=1, n_blocks=4, embed_dim=768, depth=12,modes=32,
#                  mlp_ratio=1., representation_size=None, uniform_drop=False,
#                  drop_rate=0., drop_path_rate=0., dropcls=0, n_cls = 1, normalize=False,time_agg='mlp'):
#         """
#         Args:
#             img_size (int, tuple): input image size
#             patch_size (int, tuple): patch size
#             in_chans (int): number of input channels
#             num_classes (int): number of classes for classification head
#             embed_dim (int): embedding dimension
#             depth (int): depth of transformer
#             num_heads (int): number of attention heads
#             mlp_ratio (int): ratio of mlp hidden dim to embedding dim
#             qkv_bias (bool): enable bias for qkv if True
#             qk_scale (float): override default qk scale of head_dim ** -0.5 if set
#             representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
#             drop_rate (float): dropout rate
#             attn_drop_rate (float): attention dropout rate
#             drop_path_rate (float): stochastic depth rate
#             hybrid_backbone (nn.Module): CNN backbone to use in-place of PatchEmbed module
#             norm_layer: (nn.Module): normalization layer
#         """
#         super().__init__()
#
#         # self.num_classes = num_classes
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.in_timesteps = in_timesteps
#         self.out_timesteps = out_timesteps
#
#         self.n_blocks = n_blocks
#         self.modes = modes
#         self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
#
#         self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_channels * in_timesteps + 2, embed_dim=in_timesteps * out_channels * patch_size + 2, out_dim=embed_dim)
#
#         self.latent_size = self.patch_embed.out_size
#
#         # self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
#         self.pos_embed = nn.Parameter(torch.zeros(1, embed_dim, self.patch_embed.out_size[0], self.patch_embed.out_size[1]))
#         # self.pos_drop = nn.Dropout(p=drop_rate)
#         self.normalize = normalize
#         self.time_agg = time_agg
#         self.n_cls = n_cls
#
#
#         h = img_size // patch_size
#         w = h // 2 + 1
#
#         if uniform_drop:
#             print('using uniform droppath with expect rate', drop_path_rate)
#             dpr = [drop_path_rate for _ in range(depth)]  # stochastic depth decay rule
#         else:
#             print('using linear droppath with expect rate', drop_path_rate * 0.5)
#             dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
#         # dpr = [drop_path_rate for _ in range(depth)]  # stochastic depth decay rule
#
#         self.blocks = nn.ModuleList([
#             Block(mixing_type=mixing_type,modes=modes,
#                 width=embed_dim, mlp_ratio=mlp_ratio, channel_first = True, n_blocks=n_blocks,double_skip=False,
#                 drop=drop_rate, drop_path=dpr[i], h=h, w=w,)
#             for i in range(depth)])
#
#         if self.normalize:
#             self.scale_feats = nn.Linear(2 * in_channels, embed_dim)
#         self.cls_head = nn.Sequential(
#             nn.Linear(embed_dim, embed_dim),
#             nn.GELU(),
#             nn.Linear(embed_dim, embed_dim),
#             nn.GELU(),
#             nn.Linear(embed_dim, n_cls)
#         )
#
#         ###TODO: time agg layer
#         # if self.time_agg == 'mlp':
#             #self.time_agg_layer = nn.Linear()
#
#         # self.norm = norm_layer(embed_dim)
#
#         # up sampler size
#         # self.out_layer = nn.Sequential(
#         #     nn.ConvTranspose2d(in_channels=embed_dim, out_channels=embed_dim, kernel_size=patch_size, stride=patch_size),
#         #     nn.GELU(),
#         #     nn.Conv2d(in_channels=embed_dim, out_channels=embed_dim, kernel_size=1, stride=1),
#         #     nn.GELU(),
#         #     nn.Conv2d(in_channels=embed_dim, out_channels=self.out_channels * self.out_timesteps,kernel_size=1, stride=1)
#         # )
#         # self.out_layer = nn.Sequential(
#         #     nn.Conv2d(in_channels=embed_dim, out_channels=in_channels * out_timesteps *patch_size**2 , kernel_size=1, stride=1),
#         #     nn.GELU(),
#         #     nn.PixelShuffle(patch_size),
#         #     nn.Conv2d(in_channels=in_channels * out_timesteps , out_channels=in_channels * out_timesteps, kernel_size=3, stride=1,padding=1),
#         #     # nn.GELU(),
#         #     # nn.ConvTranspose2d(in_channels=, out_channels=in_channels * out_timesteps, kernel_size=patch_size, stride=patch_size),
#         #     # nn.ConvTranspose2d(in_channels=in_channels * out_timesteps * patch_size,out_channels=in_channels * out_timesteps, kernel_size=patch_size, stride=patch_size),
#         #     # nn.GELU(),
#         #     # nn.Conv2d(in_channels=in_channels * out_timesteps * patch_size, out_channels=in_channels * out_timesteps, kernel_size=1,stride=1)
#         # )
#         ### attempt load balancing for high resolution
#         self.out_layer = nn.Sequential(
#             nn.ConvTranspose2d(in_channels=embed_dim, out_channels=32, kernel_size=patch_size, stride=patch_size),
#             nn.GELU(),
#             nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, stride=1),
#             nn.GELU(),
#             nn.Conv2d(in_channels=32, out_channels=self.out_channels * self.out_timesteps,kernel_size=1, stride=1)
#         )
#
#
#         if dropcls > 0:
#             print('dropout %.2f before classifier' % dropcls)
#             self.final_dropout = nn.Dropout(p=dropcls)
#         else:
#             self.final_dropout = nn.Identity()
#
#         torch.nn.init.trunc_normal_(self.pos_embed, std=.02)
#         # self.apply(self._init_weights)
#
#         self.mixing_type = mixing_type
#
#     def _init_weights(self, m):
#         if isinstance(m, nn.Linear):
#             torch.nn.init.trunc_normal_(m.weight, std=.02)
#             if isinstance(m, nn.Linear) and m.bias is not None:
#                 nn.init.constant_(m.bias, 0)
#         elif isinstance(m, nn.LayerNorm):
#             nn.init.constant_(m.bias, 0)
#             nn.init.constant_(m.weight, 1.0)
#
#
#
#     def get_grid(self, x):
#         batchsize, size_x, size_y = x.shape[0], x.shape[1], x.shape[2]
#         gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
#         gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
#         gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
#         gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
#         grid = torch.cat((gridx, gridy), dim=-1).to(x.device)
#         return grid
#
#
#     def get_grid_3d(self, x):
#         batchsize, size_x, size_y, size_z = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
#         gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
#         gridx = gridx.reshape(1, size_x, 1, 1, 1).to(x.device).repeat([batchsize, 1, size_y, size_z, 1])
#         gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
#         gridy = gridy.reshape(1, 1, size_y, 1, 1).to(x.device).repeat([batchsize, size_x, 1, size_z, 1])
#         gridz = torch.tensor(np.linspace(0, 1, size_z), dtype=torch.float)
#         gridz = gridz.reshape(1, 1, 1, size_z, 1).to(x.device).repeat([batchsize, size_x, size_y, 1, 1])
#
#         grid = torch.cat((gridx, gridy, gridz), dim=-1)
#         return grid
#
#
#     def time_aggregate(self, x):
#         if self.time_agg == 'mlp':
#             x = x.view(*x.shape[:-2], -1)  #### B, X, Y, T*C
#             grid = self.get_grid(x)
#             x = torch.cat((x, grid), dim=-1)
#
#
#
#     ### in/out: B, X, Y, T, C
#     def forward(self, x):
#         if self.normalize:
#             mu, sigma = x.mean(dim=(1,2,3),keepdim=True), x.std(dim=(1,2,3),keepdim=True) + 1e-6    # B,1,1,1,C
#             x = (x - mu)/ sigma
#             scale_feats = self.scale_feats(torch.cat([mu, sigma],dim=-1)).squeeze(-2).permute(0,3,1,2)   # B, 1, 1, C
#         else:
#             scale_feats = 0.0
#
#
#         x = x.view(*x.shape[:-2], -1)     #### B, X, Y, T*C
#         grid = self.get_grid(x)
#
#
#
#         x = torch.cat((x, grid), dim=-1).permute(0,3,1,2).contiguous()  #### B, X, Y, T*C +2 -> B, T*C+2, X, Y
#         x = self.patch_embed(x) + scale_feats
#         x = x + self.pos_embed
#         # x = self.pos_drop(x)
#
#         for blk in self.blocks:
#             x = blk(x)
#
#         # if not get_args().checkpoint_activations:
#         #     for blk in self.blocks:
#         #         x = blk(x)
#         # else:
#         #     x = checkpoint_sequential(self.blocks, 4, x)
#
#             # classification
#
#         cls_token = x.mean(dim=(2, 3), keepdim=False)
#         # print(cls_token.shape)
#         cls_pred = self.cls_head(cls_token)
#
#         x = self.out_layer(x).permute(0, 2, 3, 1)
#         x = x.reshape(*x.shape[:3], self.out_timesteps, self.out_channels).contiguous()
#
#         if self.normalize:
#             x = x * sigma  + mu
#
#         return x, cls_pred

'''
    AFNO version. 
    summary: stable, using time aggregator
'''
class CDPOTNet(nn.Module):
    def __init__(self, img_size=224, patch_size=16, mixing_type = 'afno',in_channels=1, out_channels = 3, in_timesteps=1, out_timesteps=1, n_blocks=4, embed_dim=768,out_layer_dim=32, depth=12,modes=32,
                 mlp_ratio=1., representation_size=None, uniform_drop=False,
                 drop_rate=0., drop_path_rate=0., dropcls=0, n_cls = 1, normalize=False,act='gelu',time_agg='exp_mlp'):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            hybrid_backbone (nn.Module): CNN backbone to use in-place of PatchEmbed module
            norm_layer: (nn.Module): normalization layer
        """
        super().__init__()

        # self.num_classes = num_classes
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.in_timesteps = in_timesteps
        self.out_timesteps = out_timesteps

        self.n_blocks = n_blocks
        self.modes = modes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.mlp_ratio = mlp_ratio
        self.act = ACTIVATION[act]
        patching_act = 'cno_lrelu_torch'
        # patching_act = 'lrelu'
        self.patch_embed = CNOPatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_channels + 3, embed_dim=out_channels * patch_size + 3, out_dim=embed_dim,act=patching_act)

        self.img_size = img_size
        self.latent_size = self.patch_embed.out_size


        # self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, embed_dim, self.patch_embed.out_size[0], self.patch_embed.out_size[1]))
        # self.pos_drop = nn.Dropout(p=drop_rate)
        self.normalize = normalize
        self.time_agg = time_agg
        self.n_cls = n_cls


        h = img_size // patch_size
        w = h // 2 + 1

        if uniform_drop:
            print('using uniform droppath with expect rate', drop_path_rate)
            dpr = [drop_path_rate for _ in range(depth)]  # stochastic depth decay rule
        else:
            print('using linear droppath with expect rate', drop_path_rate * 0.5)
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        # dpr = [drop_path_rate for _ in range(depth)]  # stochastic depth decay rule

        self.blocks = nn.ModuleList([
            Block(mixing_type=mixing_type,modes=modes,
                width=embed_dim, mlp_ratio=mlp_ratio, channel_first = True, n_blocks=n_blocks,double_skip=False,
                drop=drop_rate, drop_path=dpr[i], h=h, w=w,act = act)
            for i in range(depth)])

        if self.normalize:
            self.scale_feats_mu = nn.Linear(2 * in_channels, embed_dim)
            self.scale_feats_sigma = nn.Linear(2 * in_channels, embed_dim)
        self.cls_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            self.act,
            nn.Linear(embed_dim, embed_dim),
            self.act,
            nn.Linear(embed_dim, n_cls)
        )

        self.time_agg_layer = TimeAggregator(in_channels, in_timesteps, embed_dim, time_agg)

        # self.norm = norm_layer(embed_dim)

        ### attempt load balancing for high resolution
        # self.act_patching = LReLu_torch(out_layer_dim, out_layer_dim, self.img_size,self.img_size, self.img_size, self.img_size)

        self.out_layer = nn.Sequential(
            # nn.ConvTranspose2d(in_channels=embed_dim, out_channels=out_layer_dim, kernel_size=patch_size, stride=patch_size),
            CNOBlock(embed_dim, out_layer_dim, self.latent_size[0], self.img_size,cutoff_den=2.0001,conv_kernel=1,activation=patching_act),
            # self.act_patching,
            # self.act,
            nn.Conv2d(in_channels=out_layer_dim, out_channels=out_layer_dim, kernel_size=1, stride=1),
            self.act,
            nn.Conv2d(in_channels=out_layer_dim, out_channels=self.out_channels * self.out_timesteps,kernel_size=1, stride=1)
        )


        if dropcls > 0:
            print('dropout %.2f before classifier' % dropcls)
            self.final_dropout = nn.Dropout(p=dropcls)
        else:
            self.final_dropout = nn.Identity()

        torch.nn.init.trunc_normal_(self.pos_embed, std=.02)
        # self.apply(self._init_weights)

        self.mixing_type = mixing_type

    def _init_weights(self, m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            torch.nn.init.trunc_normal_(m.weight, std=.002)    # .02
            if m.bias is not None:
            # if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)



    def get_grid(self, x):
        batchsize, size_x, size_y = x.shape[0], x.shape[1], x.shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        grid = torch.cat((gridx, gridy), dim=-1).to(x.device)
        return grid


    def get_grid_3d(self, x):
        batchsize, size_x, size_y, size_z = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1, 1).to(x.device).repeat([batchsize, 1, size_y, size_z, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1, 1).to(x.device).repeat([batchsize, size_x, 1, size_z, 1])
        gridz = torch.tensor(np.linspace(0, 1, size_z), dtype=torch.float)
        gridz = gridz.reshape(1, 1, 1, size_z, 1).to(x.device).repeat([batchsize, size_x, size_y, 1, 1])

        grid = torch.cat((gridx, gridy, gridz), dim=-1)
        return grid


    ### in/out: B, X, Y, T, C
    def forward(self, x):
        B, _, _, T, _ = x.shape
        if self.normalize:
            mu, sigma = x.mean(dim=(1,2,3),keepdim=True), x.std(dim=(1,2,3),keepdim=True) + 1e-6    # B,1,1,1,C
            x = (x - mu)/ sigma
            scale_mu = self.scale_feats_mu(torch.cat([mu, sigma],dim=-1)).squeeze(-2).permute(0,3,1,2)   #-> B, C, 1, 1
            scale_sigma = self.scale_feats_sigma(torch.cat([mu, sigma], dim=-1)).squeeze(-2).permute(0, 3, 1, 2)


        grid = self.get_grid_3d(x)
        x = torch.cat((x, grid), dim=-1).contiguous() # B, X, Y, T, C+3
        x = rearrange(x, 'b x y t c -> (b t) c x y')
        x = self.patch_embed(x)

        x = x + self.pos_embed

        x = rearrange(x, '(b t) c x y -> b x y t c', b=B, t=T)

        x = self.time_agg_layer(x)

        # x = self.pos_drop(x)
        x = rearrange(x, 'b x y c -> b c x y')

        if self.normalize:
            x = scale_sigma * x + scale_mu   ### Ada_in layer

        for blk in self.blocks:
            x = blk(x)

        # if not get_args().checkpoint_activations:
        #     for blk in self.blocks:
        #         x = blk(x)
        # else:
        #     x = checkpoint_sequential(self.blocks, 4, x)

            # classification

        cls_token = x.mean(dim=(2, 3), keepdim=False)
        # print(cls_token.shape)
        cls_pred = self.cls_head(cls_token)

        x = self.out_layer(x).permute(0, 2, 3, 1)
        x = x.reshape(*x.shape[:3], self.out_timesteps, self.out_channels).contiguous()

        if self.normalize:
            x = x * sigma  + mu

        return x, cls_pred

    def extra_repr(self) -> str:

        named_modules = set()
        for p in self.named_modules():
            named_modules.update([p[0]])
        named_modules = list(named_modules)

        string_repr = ''
        for p in self.named_parameters():
            name = p[0].split('.')[0]
            if name not in named_modules:
                string_repr = string_repr + '(' + name + '): ' \
                              + 'tensor(' + str(tuple(p[1].shape)) + ', requires_grad=' + str(
                    p[1].requires_grad) + ')\n'

        return string_repr



def resize_pos_embed(posemb, posemb_new):
    # Rescale the grid of position embeddings when loading from state_dict. Adapted from
    # https://github.com/google-research/vision_transformer/blob/00883dd691c63a6830751563748663526e811cee/vit_jax/checkpoint.py#L224
    _logger.info('Resized position embedding: %s to %s', posemb.shape, posemb_new.shape)
    ntok_new = posemb_new.shape[1]
    if True:
        posemb_tok, posemb_grid = posemb[:, :1], posemb[0, 1:]
        ntok_new -= 1
    else:
        posemb_tok, posemb_grid = posemb[:, :0], posemb[0]
    gs_old = int(math.sqrt(len(posemb_grid)))
    gs_new = int(math.sqrt(ntok_new))
    _logger.info('Position embedding grid-size from %s to %s', gs_old, gs_new)
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(posemb_grid, size=(gs_new, gs_new), mode='bilinear')
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, gs_new * gs_new, -1)
    posemb = torch.cat([posemb_tok, posemb_grid], dim=1)
    return posemb


def checkpoint_filter_fn(state_dict, model):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    if 'model' in state_dict:
        # For deit models
        state_dict = state_dict['model']
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k and len(v.shape) < 4:
            # For old models that I trained prior to conv based patchification
            O, I, H, W = model.patch_embed.proj.weight.shape
            v = v.reshape(O, -1, H, W)
        elif k == 'pos_embed' and v.shape != model.pos_embed.shape:
            # To resize pos embedding when using model at different size from pretrained weights
            v = resize_pos_embed(v, model.pos_embed)
        out_dict[k] = v
    return out_dict


if __name__ == "__main__":
    # x = torch.rand(4, 20, 20, 100)
    # net = AFNO2D(in_timesteps=3, out_timesteps=1, n_channels=2, width=100, num_blocks=5)
    x = torch.rand(4, 20, 20, 6, 3)
    net = CDPOTNet(img_size=20, patch_size=5, in_channels=3, out_channels=3, in_timesteps=6, out_timesteps=1, embed_dim=32, normalize=True)
    y,_ = net(x)
    print(y.shape)