import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

from aqbench.registry.model_registry import register_model
from aqbench.models.base import (
    AbsoluteTimeEmbedding,
    get_grid,
    stack_time_into_channels,
    unstack_channels_into_time,
)


def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)


def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)


class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None,
                 attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim         = dim
        self.window_size = window_size
        self.num_heads   = num_heads
        head_dim         = dim // num_heads
        self.scale       = qk_scale or head_dim ** -0.5

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))
        trunc_normal_(self.relative_position_bias_table, std=0.02)

        coords_h       = torch.arange(self.window_size[0])
        coords_w       = torch.arange(self.window_size[1])
        coords         = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))
        coords_flatten = torch.flatten(coords, 1)
        rc             = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        rc             = rc.permute(1, 2, 0).contiguous()
        rc[:, :, 0]   += self.window_size[0] - 1
        rc[:, :, 1]   += self.window_size[1] - 1
        rc[:, :, 0]   *= 2 * self.window_size[1] - 1
        self.register_buffer("relative_position_index", rc.sum(-1))

        self.qkv       = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj      = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax   = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads,
                                    C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q * self.scale) @ k.transpose(-2, -1)

        rpb = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(self.window_size[0] * self.window_size[1],
               self.window_size[0] * self.window_size[1], -1).permute(2, 0, 1).contiguous()
        attn = attn + rpb.unsqueeze(0)

        if mask is not None:
            nW   = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + \
                   mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = self.attn_drop(self.softmax(attn))
        return self.proj_drop(self.proj((attn @ v).transpose(1, 2).reshape(B_, N, C)))


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features    = out_features    or in_features
        hidden_features = hidden_features or in_features
        self.fc1  = nn.Linear(in_features, hidden_features)
        self.act  = act_layer()
        self.fc2  = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size=4, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.window_size      = window_size
        self.shift_size       = shift_size

        if min(self.input_resolution) <= self.window_size:
            self.shift_size  = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size

        self.norm1 = norm_layer(dim)
        self.attn  = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp   = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio),
                         act_layer=nn.GELU, drop=drop)

        if self.shift_size > 0:
            H, W     = self.input_resolution
            img_mask = torch.zeros(1, H, W, 1)
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1
            mask_windows = window_partition(img_mask, self.window_size).view(
                -1, self.window_size ** 2)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, -100.).masked_fill(
                attn_mask == 0, 0.)
        else:
            attn_mask = None
        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        H, W    = self.input_resolution
        B, L, C = x.shape
        shortcut = x
        x = self.norm1(x).view(B, H, W, C)

        sx = torch.roll(x, (-self.shift_size, -self.shift_size), (1, 2)) \
             if self.shift_size > 0 else x

        xw = window_partition(sx, self.window_size).view(-1, self.window_size ** 2, C)
        aw = self.attn(xw, mask=self.attn_mask).view(
            -1, self.window_size, self.window_size, C)
        x  = window_reverse(aw, self.window_size, H, W)

        if self.shift_size > 0:
            x = torch.roll(x, (self.shift_size, self.shift_size), (1, 2))

        x = shortcut + self.drop_path(x.view(B, L, C))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x



class PatchEmbed(nn.Module):
    def __init__(self, in_channels, embed_dim, patch_size):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim,
                              kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):   # x: (B, H, W, C)
        x = self.proj(x.permute(0, 3, 1, 2))
        return self.norm(x.flatten(2).transpose(1, 2))


class PatchMerging(nn.Module):
    def __init__(self, input_resolution, dim):
        super().__init__()
        self.input_resolution = input_resolution
        self.norm      = nn.LayerNorm(4 * dim)
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        x = x.view(B, H, W, C)
        x = torch.cat([x[:, 0::2, 0::2], x[:, 1::2, 0::2],
                       x[:, 0::2, 1::2], x[:, 1::2, 1::2]], dim=-1)
        return self.reduction(self.norm(x.view(B, -1, 4 * C)))


class PatchExpand(nn.Module):
    def __init__(self, input_resolution, dim):
        super().__init__()
        self.input_resolution = input_resolution
        self.expand = nn.Linear(dim, 2 * dim, bias=False)
        self.norm   = nn.LayerNorm(dim // 2)

    def forward(self, x):
        H, W = self.input_resolution
        x = self.expand(x)
        B, L, C = x.shape
        x = x.view(B, H, W, 2, 2, C // 4).permute(0, 1, 3, 2, 4, 5).contiguous()
        return self.norm(x.view(B, 2 * H * 2 * W, C // 4))


class FinalPatchExpand(nn.Module):
    def __init__(self, input_resolution, dim, patch_size):
        super().__init__()
        self.input_resolution = input_resolution
        self.patch_size = patch_size
        self.expand = nn.Linear(dim, patch_size ** 2 * dim, bias=False)
        self.norm   = nn.LayerNorm(dim)

    def forward(self, x):
        H, W = self.input_resolution
        p    = self.patch_size
        x    = self.expand(x)
        B, L, C_exp = x.shape
        C    = C_exp // (p * p)
        x    = x.view(B, H, W, p, p, C).permute(0, 1, 3, 2, 4, 5).contiguous()
        return self.norm(x.view(B, p * H, p * W, C))


class BasicLayer(nn.Module):
    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., drop=0., attn_drop=0., drop_path=0.,
                 downsample=None, use_checkpoint=False):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim, input_resolution=input_resolution,
                num_heads=num_heads, window_size=window_size,
                shift_size=0 if i % 2 == 0 else window_size // 2,
                mlp_ratio=mlp_ratio, drop=drop, attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path)
            for i in range(depth)
        ])
        self.downsample = downsample

    def forward(self, x):
        for blk in self.blocks:
            x = checkpoint.checkpoint(blk, x, use_reentrant=False) \
                if self.use_checkpoint else blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x


class BasicLayer_up(nn.Module):
    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., drop=0., attn_drop=0., drop_path=0.,
                 upsample=None, use_checkpoint=False):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim, input_resolution=input_resolution,
                num_heads=num_heads, window_size=window_size,
                shift_size=0 if i % 2 == 0 else window_size // 2,
                mlp_ratio=mlp_ratio, drop=drop, attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path)
            for i in range(depth)
        ])
        self.upsample = upsample

    def forward(self, x):
        for blk in self.blocks:
            x = checkpoint.checkpoint(blk, x, use_reentrant=False) \
                if self.use_checkpoint else blk(x)
        if self.upsample is not None:
            x = self.upsample(x)
        return x



@register_model("swin")
class Swin(nn.Module):
    def __init__(
        self,
        in_channels,                 
        height,
        width,
        embed_dim      = 72,
        depths         = (2, 2, 2, 2),
        depths_decoder = None,
        num_heads      = (3, 6, 12, 24),
        window_size    = 4,
        patch_size     = 2,
        mlp_ratio      = 4,
        dropout        = 0.0,
        attn_drop      = 0.0,
        drop_path_rate = 0.1,
        use_checkpoint = False,
        num_targets    = 1,
        Tin            = 6,
        Tout           = 6,
        time_embed_d   = 16,
        **kwargs,
    ):
        super().__init__()

        depths         = list(depths)
        num_heads      = list(num_heads)
        num_layers     = len(depths)
        depths_decoder = list(depths_decoder) if depths_decoder is not None \
                         else list(reversed(depths))

        assert len(num_heads)      == num_layers
        assert len(depths_decoder) == num_layers
        assert patch_size >= 1 and (patch_size & (patch_size - 1)) == 0

        self.Tin            = Tin
        self.Tout           = Tout
        self.H_orig         = height
        self.W_orig         = width
        self.num_layers     = num_layers
        self.embed_dim      = embed_dim
        self.use_checkpoint = use_checkpoint
        self.num_targets    = num_targets

        self.time_embed = AbsoluteTimeEmbedding(d=time_embed_d)
        d_time  = self.time_embed.embed_dim          
        eff_in  = Tin * (in_channels + 2 + d_time)   
        out_dim = Tout * num_targets

        stride     = patch_size * (2 ** (num_layers - 1)) * window_size
        self.H_pad = math.ceil(height / stride) * stride
        self.W_pad = math.ceil(width  / stride) * stride
        Ph, Pw     = self.H_pad // patch_size, self.W_pad // patch_size
        self.patches_resolution = (Ph, Pw)

        self.patch_embed = PatchEmbed(eff_in, embed_dim, patch_size)

        dpr     = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        dpr_dec = [x.item() for x in torch.linspace(drop_path_rate, 0, sum(depths_decoder))]

        self.layers = nn.ModuleList()
        for i in range(num_layers):
            dim = int(embed_dim * 2 ** i)
            res = (Ph // (2 ** i), Pw // (2 ** i))
            self.layers.append(BasicLayer(
                dim=dim, input_resolution=res, depth=depths[i],
                num_heads=num_heads[i], window_size=window_size,
                mlp_ratio=mlp_ratio, drop=dropout, attn_drop=attn_drop,
                drop_path=dpr[sum(depths[:i]) : sum(depths[:i + 1])],
                downsample=PatchMerging(res, dim) if i < num_layers - 1 else None,
                use_checkpoint=use_checkpoint))

        self.norm = nn.LayerNorm(int(embed_dim * 2 ** (num_layers - 1)))

        self.layers_up       = nn.ModuleList()
        self.concat_back_dim = nn.ModuleList()
        dec_offset = 0
        for i in range(num_layers):
            enc_stage = num_layers - 1 - i
            dim = int(embed_dim * 2 ** enc_stage)
            res = (Ph // (2 ** enc_stage), Pw // (2 ** enc_stage))

            self.concat_back_dim.append(
                nn.Linear(2 * dim, dim, bias=False) if i > 0 else nn.Identity())

            if i == 0:
                self.layers_up.append(PatchExpand(res, dim))
            else:
                d  = depths_decoder[enc_stage]
                dp = dpr_dec[dec_offset : dec_offset + d]
                dec_offset += d
                self.layers_up.append(BasicLayer_up(
                    dim=dim, input_resolution=res, depth=d,
                    num_heads=num_heads[enc_stage], window_size=window_size,
                    mlp_ratio=mlp_ratio, drop=dropout, attn_drop=attn_drop,
                    drop_path=dp,
                    upsample=PatchExpand(res, dim) if i < num_layers - 1 else None,
                    use_checkpoint=use_checkpoint))

        self.norm_up = nn.LayerNorm(embed_dim)
        self.up      = FinalPatchExpand(self.patches_resolution, embed_dim, patch_size)
        self.head    = nn.Linear(embed_dim, out_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        skips = []
        for layer in self.layers:
            skips.append(x)
            x = layer(x)
        return self.norm(x), skips

    def forward_up_features(self, x, skips):
        for i, layer_up in enumerate(self.layers_up):
            if i == 0:
                x = layer_up(x)
            else:
                skip = skips[self.num_layers - 1 - i]
                x    = self.concat_back_dim[i](torch.cat([x, skip], dim=-1))
                x    = layer_up(x)
        return self.norm_up(x)

    def forward(self, x: torch.Tensor,
                t_in: torch.Tensor = None) -> torch.Tensor:
        B, Tin, H, W, _ = x.shape

        grid = get_grid(B, H, W, x.device)                  
        te   = self.time_embed(t_in) if t_in is not None else None
        x_s  = stack_time_into_channels(x, grid=grid, time_embed=te)

        pad_h = self.H_pad - H
        pad_w = self.W_pad - W
        if pad_h > 0 or pad_w > 0:
            x_s = F.pad(x_s, (0, 0, 0, pad_w, 0, pad_h))

        x_s = self.patch_embed(x_s)
        x_s, skips = self.forward_features(x_s)
        x_s = self.forward_up_features(x_s, skips)
        x_s = self.up(x_s)
        x_s = self.head(x_s)[:, :H, :W, :]   # crop padding

        return unstack_channels_into_time(x_s, self.Tout, self.num_targets)

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {"relative_position_bias_table"}