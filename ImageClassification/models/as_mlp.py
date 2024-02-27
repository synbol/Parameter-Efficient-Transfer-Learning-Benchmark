# --------------------------------------------------------
# AS-MLP
# Licensed under The MIT License [see LICENSE for details]
# Written by Zehao Yu and Dongze Lian (AS-MLP)
# --------------------------------------------------------

import logging
import math
from copy import deepcopy
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.fx_features import register_notrace_function
from timm.models.helpers import build_model_with_cfg 
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.layers import _assert
from timm.models.registry import register_model

from timm.models.vision_transformer import checkpoint_filter_fn 



_logger = logging.getLogger(__name__)

def _cfg(url='', file='', **kwargs):
    return {
        'url': url,
        'file': file,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic', 'fixed_input_size': True,
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }

default_cfgs = {
    'as_base_patch4_window7_224': _cfg(
        file='/path/to/asmlp_base_patch4_shift5_224.pth'
    ),
}


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., tuning_mode=None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1, 1)
        self.drop = nn.Dropout(drop)


        self.tuning_mode = tuning_mode
        if tuning_mode == 'ssf':        
            self.ssf_scale_1, self.ssf_shift_1 = init_ssf_scale_shift(hidden_features)
            self.ssf_scale_2, self.ssf_shift_2 = init_ssf_scale_shift(out_features)


    def forward(self, x):
        x = self.fc1(x)
        if self.tuning_mode == 'ssf':
            x = ssf_ada(x, self.ssf_scale_1, self.ssf_shift_1)

        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x) 
        if self.tuning_mode == 'ssf':
            x = ssf_ada(x, self.ssf_scale_2, self.ssf_shift_2)

        x = self.drop(x)
        
        return x



class AxialShift(nn.Module):
    r""" Axial shift  

    Args:
        dim (int): Number of input channels.
        shift_size (int): shift size .
        as_bias (bool, optional):  If True, add a learnable bias to as mlp. Default: True
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, shift_size, as_bias=True, proj_drop=0., tuning_mode=None):

        super().__init__()
        self.dim = dim
        self.shift_size = shift_size
        self.pad = shift_size // 2
        self.conv1 = nn.Conv2d(dim, dim, 1, 1, 0, groups=1, bias=as_bias)
        self.conv2_1 = nn.Conv2d(dim, dim, 1, 1, 0, groups=1, bias=as_bias)
        self.conv2_2 = nn.Conv2d(dim, dim, 1, 1, 0, groups=1, bias=as_bias)
        self.conv3 = nn.Conv2d(dim, dim, 1, 1, 0, groups=1, bias=as_bias)

        self.actn = nn.GELU()

        self.norm1 = MyNorm(dim)
        self.norm2 = MyNorm(dim)
        self.tuning_mode = tuning_mode


        if tuning_mode == 'ssf':     
            self.ssf_scale_1, self.ssf_shift_1 = init_ssf_scale_shift(dim)
            self.ssf_scale_2, self.ssf_shift_2 = init_ssf_scale_shift(dim)
            self.ssf_scale_3, self.ssf_shift_3 = init_ssf_scale_shift(dim)
            self.ssf_scale_4, self.ssf_shift_4 = init_ssf_scale_shift(dim)


    def forward(self, x):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, C, H, W = x.shape
        x = self.conv1(x)
        if self.tuning_mode == 'ssf':   
            x = ssf_ada(x, self.ssf_scale_1, self.ssf_shift_1)
        
        x = self.norm1(x)
        x = self.actn(x)
       
        
        x = F.pad(x, (self.pad, self.pad, self.pad, self.pad) , "constant", 0)
        
        xs = torch.chunk(x, self.shift_size, 1)

        def shift(dim):
            x_shift = [ torch.roll(x_c, shift, dim) for x_c, shift in zip(xs, range(-self.pad, self.pad+1))]
            x_cat = torch.cat(x_shift, 1)
            x_cat = torch.narrow(x_cat, 2, self.pad, H)
            x_cat = torch.narrow(x_cat, 3, self.pad, W)
            return x_cat

        x_shift_lr = shift(3)
        x_shift_td = shift(2)
    

        if self.tuning_mode == 'ssf':   
            x_lr = ssf_ada(self.conv2_1(x_shift_lr), self.ssf_scale_2, self.ssf_shift_2)
            x_td = ssf_ada(self.conv2_2(x_shift_td), self.ssf_scale_3, self.ssf_shift_3)
        else:
            x_lr = self.conv2_1(x_shift_lr)
            x_td = self.conv2_2(x_shift_td)

        x_lr = self.actn(x_lr)
        x_td = self.actn(x_td)

        x = x_lr + x_td
        x = self.norm2(x)

        x = self.conv3(x)
        if self.tuning_mode == 'ssf': 
            x = ssf_ada(x, self.ssf_scale_4, self.ssf_shift_4)

        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, shift_size={self.shift_size}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # conv1 
        flops += N * self.dim * self.dim
        # norm 1
        flops += N * self.dim
        # conv2_1 conv2_2
        flops += N * self.dim * self.dim * 2
        # x_lr + x_td
        flops += N * self.dim
        # norm2
        flops += N * self.dim
        # norm3
        flops += N * self.dim * self.dim
        return flops


class AxialShiftedBlock(nn.Module):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        shift_size (int): Shift size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        as_bias (bool, optional): If True, add a learnable bias to Axial Mlp. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, input_resolution, shift_size=7,
                 mlp_ratio=4., as_bias=True, drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, tuning_mode=None):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        self.norm1 = norm_layer(dim)
        self.axial_shift = AxialShift(dim, shift_size=shift_size, as_bias=as_bias, proj_drop=drop, tuning_mode=tuning_mode)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop, tuning_mode=tuning_mode)


        self.tuning_mode = tuning_mode

        if tuning_mode == 'ssf':    
            self.ssf_scale_1, self.ssf_shift_1 = init_ssf_scale_shift(dim)
            self.ssf_scale_2, self.ssf_shift_2 = init_ssf_scale_shift(dim)


    def forward(self, x):
        B, C, H, W = x.shape

        shortcut = x
        
        x = self.norm1(x)
        if self.tuning_mode == 'ssf':
            x = ssf_ada(x, self.ssf_scale_1, self.ssf_shift_1)

        # axial shift block
        x = self.axial_shift(x)  # B, C, H, W

        # FFN
        x = shortcut + self.drop_path(x)
        if self.tuning_mode == 'ssf':
            x = x + self.drop_path(self.mlp(ssf_ada(self.norm2(x), self.ssf_scale_2, self.ssf_shift_2)))
        else:
            x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, " \
               f"shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # shift mlp 
        flops += self.axial_shift.flops(H * W)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops


class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm, tuning_mode=None):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Conv2d(4 * dim, 2 * dim, 1, 1, bias=False)
        self.norm = norm_layer(4 * dim)
        self.tuning_mode = tuning_mode
        if tuning_mode == 'ssf':     
            self.ssf_scale_1, self.ssf_shift_1 = init_ssf_scale_shift(4 * dim)


    def forward(self, x):
        """
        x: B, H*W, C
        """
        B, C, H, W = x.shape
        #assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, C, H, W)

        x0 = x[:, :, 0::2, 0::2]  # B C H/2 W/2 
        x1 = x[:, :, 1::2, 0::2]  # B C H/2 W/2 
        x2 = x[:, :, 0::2, 1::2]  # B C H/2 W/2 
        x3 = x[:, :, 1::2, 1::2]  # B C H/2 W/2 
        x = torch.cat([x0, x1, x2, x3], 1)  # B 4*C H/2 W/2 

        x = self.norm(x)
        if self.tuning_mode == 'ssf':
            x = ssf_ada(x, self.ssf_scale_1, self.ssf_shift_1)

        x = self.reduction(x)

        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.dim
        flops += (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        return flops


class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, input_resolution, depth, shift_size,
                 mlp_ratio=4., as_bias=True, drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False, tuning_mode=None):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            AxialShiftedBlock(dim=dim, input_resolution=input_resolution,
                              shift_size=shift_size,
                              mlp_ratio=mlp_ratio,
                              as_bias=as_bias,
                              drop=drop, 
                              drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                              norm_layer=norm_layer, tuning_mode=tuning_mode[i])
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer, tuning_mode=tuning_mode)
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops


class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None, tuning_mode=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None
        
        self.tuning_mode = tuning_mode
        if tuning_mode == 'ssf':     
            self.ssf_scale_1, self.ssf_shift_1 = init_ssf_scale_shift(embed_dim)

            

            if norm_layer:
                self.ssf_scale_2, self.ssf_shift_2 = init_ssf_scale_shift(embed_dim)


    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)#.flatten(2).transpose(1, 2)  # B Ph*Pw C

        if self.tuning_mode == 'ssf':  
            x = ssf_ada(x, self.ssf_scale_1, self.ssf_shift_1)
            if self.norm is not None:
                x = self.norm(x)
                x = ssf_ada(x, self.ssf_scale_2, self.ssf_shift_2)
        else:
            if self.norm is not None:
                x = self.norm(x)

        return x

    def flops(self):
        Ho, Wo = self.patches_resolution
        flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops


def MyNorm(dim):
    return nn.GroupNorm(1, dim)


def init_ssf_scale_shift(dim):
    scale = nn.Parameter(torch.ones(dim))
    shift = nn.Parameter(torch.zeros(dim))

    nn.init.normal_(scale, mean=1, std=.02)
    nn.init.normal_(shift, std=.02)

    return scale, shift


def ssf_ada(x, scale, shift):
    assert scale.shape == shift.shape
    if x.shape[-1] == scale.shape[0]:
        return x * scale + shift
    elif x.shape[1] == scale.shape[0]:
        return x * scale.view(1, -1, 1, 1) + shift.view(1, -1, 1, 1)
    else:
        raise ValueError('the input tensor shape does not match the shape of the scale factor.')


class AS_MLP(nn.Module):
    r""" AS-MLP
        A PyTorch impl of : `AS-MLP: An Axial Shifted MLP Architecture for Vision`  -
          https://arxiv.org/pdf/xxx.xxx

    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each AS-MLP layer.
        window_size (int): shift size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        as_bias (bool): If True, add a learnable bias to as-mlp block. Default: True
        drop_rate (float): Dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.GroupNorm with group=1.
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000,
                 embed_dim=96, depths=[2, 2, 6, 2], 
                 shift_size=5, mlp_ratio=4., as_bias=True, 
                 drop_rate=0., drop_path_rate=0.1,
                 norm_layer=MyNorm, patch_norm=True,
                 use_checkpoint=False, tuning_mode=None, **kwargs):
        super().__init__()

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        
        self.tuning_mode = tuning_mode
        tuning_mode_list = [[tuning_mode] * depths[i_layer] for i_layer in range(self.num_layers)]
        if tuning_mode == 'ssf':   
            self.ssf_scale_1, self.ssf_shift_1 = init_ssf_scale_shift(self.num_features)
            

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                               input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                 patches_resolution[1] // (2 ** i_layer)),
                               depth=depths[i_layer],
                               shift_size=shift_size,
                               mlp_ratio=self.mlp_ratio,
                               as_bias=as_bias,
                               drop=drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                               use_checkpoint=use_checkpoint,
                               tuning_mode=tuning_mode_list[i_layer])
            self.layers.append(layer)

        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        #self.apply(self._init_weights)
    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()


    def forward_features(self, x):
        x = self.patch_embed(x)
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x)
        
        x = self.norm(x)  # B L C
        if self.tuning_mode == 'ssf': 
            x = ssf_ada(x, self.ssf_scale_1, self.ssf_shift_1)
        
        x = self.avgpool(x)  # B C 1 1
        x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x

    def flops(self):
        flops = 0
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += self.num_features * self.patches_resolution[0] * self.patches_resolution[1] // (2 ** self.num_layers)
        flops += self.num_features * self.num_classes
        return flops


def _create_as_mlp(variant, pretrained=False, **kwargs):
    model = build_model_with_cfg(
        AS_MLP, variant, pretrained,
        pretrained_filter_fn=checkpoint_filter_fn,
        **kwargs)

    return model


@register_model
def as_base_patch4_window7_224(pretrained=False, **kwargs):
    """ AS-MLP-B @ 224x224, pretrained ImageNet-1k
    """
    model_kwargs = dict(
        patch_size=4, shift_size=5, embed_dim=128, depths=(2, 2, 18, 2), **kwargs)
    return _create_as_mlp('as_base_patch4_window7_224', pretrained=pretrained, **model_kwargs)
