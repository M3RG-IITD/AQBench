import torch
import torch.nn as nn
import torch.nn.functional as F

from aqbench.registry.model_registry import register_model
from aqbench.models.base import (
    AbsoluteTimeEmbedding,
    get_grid,
    stack_time_into_channels,
    unstack_channels_into_time,
)



class DoubleConv(nn.Module):
    """Two Conv2d → GroupNorm → GELU blocks."""

    def __init__(self, in_channels, out_channels, num_groups=8):
        super().__init__()
        g = self._groups(out_channels, num_groups)
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(g, out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(g, out_channels),
            nn.GELU(),
        )

    @staticmethod
    def _groups(channels, target):
        g = min(target, channels)
        while channels % g != 0:
            g -= 1
        return g

    def forward(self, x):
        return self.net(x)


class ConvStage(nn.Module):
    """
    depth × DoubleConv blocks at the same channel width.
    first block handles any channel change (in_ch → out_ch);
    remaining blocks are out_ch → out_ch.
    """

    def __init__(self, in_channels, out_channels, depth=1, num_groups=8):
        super().__init__()
        assert depth >= 1
        blocks = [DoubleConv(in_channels, out_channels, num_groups)]
        for _ in range(depth - 1):
            blocks.append(DoubleConv(out_channels, out_channels, num_groups))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        return self.blocks(x)


class Down(nn.Module):
    """max pool downsampling"""

    def __init__(self, in_channels, out_channels, depth=1, num_groups=8):
        super().__init__()
        self.pool  = nn.MaxPool2d(2)
        self.stage = ConvStage(in_channels, out_channels, depth, num_groups)

    def forward(self, x):
        return self.stage(self.pool(x))


class Up(nn.Module):
    """bilinear upsample """

    def __init__(self, in_channels, out_channels, depth=1, num_groups=8):
        super().__init__()
        self.up    = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.stage = ConvStage(in_channels, out_channels, depth, num_groups)

    def forward(self, x, skip):
        x  = self.up(x)
        dH = skip.shape[2] - x.shape[2]
        dW = skip.shape[3] - x.shape[3]
        if dH > 0 or dW > 0:
            x = F.pad(x, [dW // 2, dW - dW // 2,
                          dH // 2, dH - dH // 2])
        return self.stage(torch.cat([skip, x], dim=1))


@register_model("unet")
class UNet(nn.Module):
    """
    Input:  (B, Tin, H, W, C_in)
    Output: (B, Tout, H, W, num_targets)
    """

    def __init__(
        self,
        in_channels,
        height,
        width,
        Tin              = 6,
        Tout             = 6,
        base_channels    = 72,
        channel_mult     = (1, 2, 4, 8),
        depths           = (2, 2, 2, 2),
        depths_decoder   = None,        
        bottleneck_depth = 2,
        num_groups       = 8,
        dropout          = 0.0,
        num_targets      = 1,
        time_embed_d     = 16,
        **kwargs,
    ):
        super().__init__()

        self.Tin         = Tin
        self.Tout        = Tout
        self.num_targets = num_targets

        depths         = list(depths)
        depths_decoder = list(depths_decoder) if depths_decoder is not None \
                         else list(reversed(depths))

        self.time_embed = AbsoluteTimeEmbedding(d=time_embed_d)
        d_time  = self.time_embed.embed_dim              
        eff_in  = Tin * (in_channels + 2 + d_time)        
        out_dim = Tout * num_targets

        c = [base_channels * m for m in channel_mult]     

        self.inc   = ConvStage(eff_in, c[0], depths[0], num_groups)
        self.down1 = Down(c[0], c[1], depths[1], num_groups)
        self.down2 = Down(c[1], c[2], depths[2], num_groups)
        self.down3 = Down(c[2], c[3], depths[3], num_groups)

        self.bottleneck = ConvStage(c[3], c[3], bottleneck_depth, num_groups)

        self.up3 = Up(c[3] + c[2], c[2], depths_decoder[0], num_groups)
        self.up2 = Up(c[2] + c[1], c[1], depths_decoder[1], num_groups)
        self.up1 = Up(c[1] + c[0], c[0], depths_decoder[2], num_groups)

        self.drop = nn.Dropout2d(p=dropout)

        self.conv_last = nn.Conv2d(c[0], out_dim, kernel_size=1)

    def forward(self, x: torch.Tensor,
                t_in: torch.Tensor = None) -> torch.Tensor:
 
        B, Tin, H, W, _ = x.shape

        grid = get_grid(B, H, W, x.device)
        te   = self.time_embed(t_in) if t_in is not None else None
        x_s  = stack_time_into_channels(x, grid=grid, time_embed=te)

        x_s = x_s.permute(0, 3, 1, 2)                   

        x1  = self.inc(x_s)                                
        x2  = self.down1(x1)                              
        x3  = self.down2(x2)                              
        x4  = self.down3(x3)                               

        x4  = self.bottleneck(x4)

        x_s = self.up3(x4, x3)                            # (B, c[2], H/4, W/4)
        x_s = self.up2(x_s, x2)                           # (B, c[1], H/2, W/2)
        x_s = self.up1(x_s, x1)                           # (B, c[0], H,   W)

        x_s = self.drop(x_s)
        x_s = self.conv_last(x_s)                         # (B, out_dim, H, W)
        x_s = x_s.permute(0, 2, 3, 1)                     # (B, H, W, out_dim)

        return unstack_channels_into_time(x_s, self.Tout, self.num_targets)