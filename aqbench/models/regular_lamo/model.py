import sys
import os
import math
from typing import Optional, Callable, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, trunc_normal_


DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"
try:
    from .csm_triton import cross_scan_fn, cross_merge_fn
except ImportError:
    from csm_triton import cross_scan_fn, cross_merge_fn
    
from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined


from aqbench.registry.model_registry import register_model
from aqbench.models.base import (
    AbsoluteTimeEmbedding,
    get_grid,
    stack_time_into_channels,
    unstack_channels_into_time,
)

class Permute(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.args = args

    def forward(self, x: torch.Tensor):
        return x.permute(*self.args)

class mamba_4D(nn.Module):
    def __init__(
        self,
        d_model=96,
        d_state=16, 
        ssm_ratio=2.0,
        dt_rank="auto",
        act_layer=nn.GELU,
        d_conv=3, 
        conv_bias=True,
        dropout=0.0,
        bias=False,
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        initialize="v2",
        forward_type="m0",
        with_initial_state=False,
        force_fp32=False, 
        chunk_size=64,
        selective_scan_backend=None,
        scan_mode="cross2d",
        scan_force_torch=False,
        **kwargs,    
    ):
        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()
        
        d_inner = int(ssm_ratio * d_model)
        dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank
        
        assert d_inner % dt_rank == 0
        assert scan_mode in ["unidi", "bidi", "cross2d"]
        assert selective_scan_backend in [None, "triton", "torch"]
        
        self.with_dconv = d_conv > 1
        self.d_state = d_state
        self.force_fp32 = force_fp32
        self.chunk_size = chunk_size
        Linear = nn.Linear

        self.disable_z = False
        self.disable_z_act = False
        self.oact = False
        self.out_norm = nn.LayerNorm(d_inner)
        
        self.selective_scan_backend = selective_scan_backend
        self.scan_mode = scan_mode
        self.scan_force_torch = scan_force_torch

        k_group = 4

        # in proj
        d_proj = d_inner if self.disable_z else (d_inner * 2)
        self.in_proj = Linear(d_model, d_proj, bias=bias)
        self.act: nn.Module = act_layer()
        
        # conv
        if self.with_dconv:
            self.conv2d = nn.Sequential(
                Permute(0, 3, 1, 2),
                nn.Conv2d(
                    in_channels=d_inner, out_channels=d_inner,
                    groups=d_inner, bias=conv_bias, kernel_size=d_conv,
                    padding=(d_conv - 1) // 2, **factory_kwargs,
                ),
                Permute(0, 2, 3, 1),
            ) 
        
        # x proj
        self.x_proj = [
            nn.Linear(d_inner, (dt_rank + d_state * 2), bias=False)
            for _ in range(k_group)
        ]
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))
        del self.x_proj
        
        # out proj
        self.out_act = nn.GELU() if self.oact else nn.Identity()
        self.out_proj = Linear(d_inner, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

        if initialize in ["v1"]:
            self.Ds = nn.Parameter(torch.ones((k_group, dt_rank, int(d_inner // dt_rank))))
            self.A_logs = nn.Parameter(torch.randn((k_group, dt_rank)))
            self.dt_projs_bias = nn.Parameter(0.1 * torch.randn((k_group, dt_rank))) 
        elif initialize in ["v2"]:
            self.Ds = nn.Parameter(torch.ones((k_group, dt_rank, int(d_inner // dt_rank))))
            self.A_logs = nn.Parameter(torch.zeros((k_group, dt_rank))) 
            self.dt_projs_bias = nn.Parameter(0.1 * torch.rand((k_group, dt_rank)))

        # init state
        self.initial_state = None
        if with_initial_state:
            self.initial_state = nn.Parameter(torch.zeros((1, k_group * dt_rank, int(d_inner // dt_rank), d_state)), requires_grad=False)

    def forward_core(self, x: torch.Tensor=None, **kwargs):
        x_proj_bias = getattr(self, "x_proj_bias", None)
        to_fp32 = lambda *args: (_a.to(torch.float32) for _a in args)

        N = self.d_state
        B, H, W, RD = x.shape
        K, R = self.A_logs.shape
        K, R, D = self.Ds.shape
        
        assert RD == R * D
        
        L = H * W
        KR = K * R
        _scan_mode = dict(cross2d=0, unidi=1, bidi=2, cascade2d=3)[self.scan_mode]

        initial_state = None
        if self.initial_state is not None:
            assert self.initial_state.shape[-1] == self.d_state
            initial_state = self.initial_state.detach().repeat(B, 1, 1, 1)
            
        xs = cross_scan_fn(x.view(B, H, W, RD), in_channel_first=False, out_channel_first=False, scans=_scan_mode, force_torch=self.scan_force_torch) 
        
        x_dbl = torch.einsum("b l k d, k c d -> b l k c", xs, self.x_proj_weight)
        if x_proj_bias is not None:
            x_dbl = x_dbl + x_proj_bias.view(1, -1, K, 1)
        
        dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=3)
        xs = xs.contiguous().view(B, L, KR, D)
        dts = dts.contiguous().view(B, L, KR)
        Bs = Bs.contiguous().view(B, L, K, N)
        Cs = Cs.contiguous().view(B, L, K, N)
        
        if self.force_fp32:
            xs, dts, Bs, Cs = to_fp32(xs, dts, Bs, Cs)

        As = -self.A_logs.to(torch.float).exp().view(KR)
        Ds = self.Ds.to(torch.float).view(KR, D)
        dt_bias = self.dt_projs_bias.view(KR)

        if self.force_fp32:
            xs, dts, Bs, Cs = to_fp32(xs, dts, Bs, Cs)
        
        ys, final_state = mamba_chunk_scan_combined(
            xs, dts, As, Bs, Cs, chunk_size = self.chunk_size, D=Ds, dt_bias=dt_bias, 
            initial_states=initial_state, dt_softplus=True, return_final_states=True,
        )
                
        y: torch.Tensor = cross_merge_fn(ys.view(B, H, W, K, RD), in_channel_first=False, out_channel_first=False, scans=_scan_mode, force_torch=self.scan_force_torch)
            
        if self.initial_state is not None:
            self.initial_state = nn.Parameter(final_state.detach().sum(0, keepdim=True), requires_grad=False)

        y = self.out_norm(y.view(B, H, W, -1))
        return y.to(x.dtype)

    def forward(self, x: torch.Tensor, **kwargs):
        x = self.in_proj(x)
        if not self.disable_z:
            x, z = x.chunk(2, dim=(-1)) 
            if not self.disable_z_act:
                z = self.act(z)
        if self.with_dconv:
            x = self.conv2d(x) 
        x = self.act(x)
        
        y = self.forward_core(x)
        y = self.out_act(y)
        if not self.disable_z:
            y = y * z
        out = self.dropout(self.out_proj(y))
        return out


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Patchify(nn.Module):
    def __init__(self, patch_size=3, in_chans=3, embed_dim=768):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        B, C, H_p, W_p = x.shape
        x = x.flatten(2).transpose(1, 2)  
        x = self.norm(x)
        return x, H_p, W_p

class DePatchify(nn.Module):
    def __init__(self, patch_size=3, embed_dim=768, out_chans=3):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.proj = nn.ConvTranspose2d(embed_dim, out_chans, kernel_size=patch_size, stride=patch_size)

    def forward(self, x, H_p, W_p):
        x = self.norm(x)
        B, L, C = x.shape
        x = x.transpose(1, 2).reshape(B, C, H_p, W_p)
        x = self.proj(x) 
        return x

class Block(nn.Module):
    def __init__(self, dim, mlp_ratio=4., drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.hydra = mamba_4D(d_model=dim, d_state=64, ssm_ratio=2.0)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.dim = dim

    def forward(self, x, H_p, W_p):
        inp = x
        x_norm = self.norm1(x).reshape(-1, H_p, W_p, self.dim)
        outp = self.drop_path(self.hydra(x_norm).reshape(-1, H_p * W_p, self.dim))
        
        x = inp + outp
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


@register_model("regular_lamo")
class LaMO(nn.Module):
    def __init__(
        self,
        in_channels,                 
        height,
        width,
        Tin          = 6,
        Tout         = 6,
        num_targets  = 1,
        time_embed_d = 16,
        embed_dims   = [192, 256, 320], 
        depths       = [3, 4, 6],
        patch_sizes  = [2, 4, 4],
        mlp_ratios   = [4, 4, 4],
        drop_rate    = 0.05,
        drop_path_rate= 0.1,
        **kwargs,
    ):
        super().__init__()
        
        self.Tin = Tin
        self.Tout = Tout
        self.H = height
        self.W = width
        self.num_targets = num_targets
        
        self.time_embed = AbsoluteTimeEmbedding(d=time_embed_d)
        d_time  = self.time_embed.embed_dim          
        eff_in  = Tin * (in_channels + 2 + d_time)   
        out_dim = Tout * num_targets                 

        self.num_scales = len(embed_dims)
        self.max_patch = max(patch_sizes)
        self.H_pad = math.ceil(self.H / self.max_patch) * self.max_patch
        self.W_pad = math.ceil(self.W / self.max_patch) * self.max_patch

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        cur_depth = 0

        for i in range(self.num_scales):
            in_ch = eff_in if i == 0 else embed_dims[i-1]
            out_ch = out_dim if i == self.num_scales - 1 else embed_dims[i]

            patch_embed = Patchify(patch_size=patch_sizes[i], in_chans=in_ch, embed_dim=embed_dims[i])
            
            num_patches = (self.H_pad // patch_sizes[i]) * (self.W_pad // patch_sizes[i])
            pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dims[i]))
            pos_drop = nn.Dropout(p=drop_rate)
            
            blocks = nn.ModuleList([
                Block(dim=embed_dims[i], mlp_ratio=mlp_ratios[i], drop=drop_rate, drop_path=dpr[cur_depth + j])
                for j in range(depths[i])
            ])
            cur_depth += depths[i]
            
            depatch_embed = DePatchify(patch_size=patch_sizes[i], embed_dim=embed_dims[i], out_chans=out_ch)
            
            setattr(self, f"patch_embed{i}", patch_embed)
            setattr(self, f"pos_embed{i}", pos_embed)
            setattr(self, f"pos_drop{i}", pos_drop)
            setattr(self, f"block{i}", blocks)
            setattr(self, f"depatch_embed{i}", depatch_embed)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, t_in: torch.Tensor = None) -> torch.Tensor:
        B, Tin, H, W, _ = x.shape
        
        grid = get_grid(B, H, W, x.device)                  
        te   = self.time_embed(t_in) if t_in is not None else None
        x_s  = stack_time_into_channels(x, grid=grid, time_embed=te) 
        
        fx = x_s.permute(0, 3, 1, 2)
        
        # padding
        pad_h = self.H_pad - H
        pad_w = self.W_pad - W
        if pad_h > 0 or pad_w > 0:
            fx = F.pad(fx, (0, pad_w, 0, pad_h), mode='constant', value=0)

        # hierarchial processing
        for i in range(self.num_scales):
            patch_embed   = getattr(self, f"patch_embed{i}")
            pos_embed     = getattr(self, f"pos_embed{i}")
            pos_drop      = getattr(self, f"pos_drop{i}")
            blocks        = getattr(self, f"block{i}")
            depatch_embed = getattr(self, f"depatch_embed{i}")

            fx, Hp, Wp = patch_embed(fx)
            fx = pos_drop(fx + pos_embed[:, :Hp*Wp, :])
            
            for blk in blocks:
                fx = blk(fx, Hp, Wp)
                
            fx = depatch_embed(fx, Hp, Wp)

        # crop back to original dimensions
        if pad_h > 0 or pad_w > 0:
            fx = fx[:, :, :H, :W]

        y = fx.permute(0, 2, 3, 1) 
        return unstack_channels_into_time(y, self.Tout, self.num_targets)