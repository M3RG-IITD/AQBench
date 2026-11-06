import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_

from aqbench.registry.model_registry import register_model
from aqbench.models.base import (
    AbsoluteTimeEmbedding,
    get_grid,
    stack_time_into_channels,
    unstack_channels_into_time,
)


class AFNO2D(nn.Module):
    """
    Global Fourier token mixing over spatial dims.
    Input/output: (B, H, W, C)  — (B, Hp, Wp, embed_dim) in practice.
    """

    def __init__(self, hidden_size, num_blocks=8, sparsity_threshold=0.01,
                 hard_thresholding_fraction=1.0):
        super().__init__()
        assert hidden_size % num_blocks == 0
        self.sparsity_threshold         = sparsity_threshold
        self.num_blocks                 = num_blocks
        self.block_size                 = hidden_size // num_blocks
        self.hard_thresholding_fraction = hard_thresholding_fraction
        self.scale = 0.02
        bs = self.block_size

        self.w1 = nn.Parameter(self.scale * torch.randn(2, num_blocks, bs, bs))
        self.b1 = nn.Parameter(self.scale * torch.randn(2, num_blocks, bs))
        self.w2 = nn.Parameter(self.scale * torch.randn(2, num_blocks, bs, bs))
        self.b2 = nn.Parameter(self.scale * torch.randn(2, num_blocks, bs))

    def forward(self, x):
        # x: (B, H, W, C)
        bias  = x
        dtype = x.dtype
        x     = x.float()
        B, H, W, C = x.shape

        x = torch.fft.rfft2(x, dim=(1, 2), norm="ortho")
        x = x.reshape(B, H, W // 2 + 1, self.num_blocks, self.block_size)

        total_modes = min(H, W) // 2 + 1
        kept_modes  = max(int(total_modes * self.hard_thresholding_fraction), 1)
        h_lo = total_modes - kept_modes
        h_hi = total_modes + kept_modes

        xr = x[:, h_lo:h_hi, :kept_modes].real
        xi = x[:, h_lo:h_hi, :kept_modes].imag

        o1r = F.relu(torch.einsum('...bi,bio->...bo', xr, self.w1[0]) -
                     torch.einsum('...bi,bio->...bo', xi, self.w1[1]) + self.b1[0])
        o1i = F.relu(torch.einsum('...bi,bio->...bo', xi, self.w1[0]) +
                     torch.einsum('...bi,bio->...bo', xr, self.w1[1]) + self.b1[1])
        o2r = (torch.einsum('...bi,bio->...bo', o1r, self.w2[0]) -
               torch.einsum('...bi,bio->...bo', o1i, self.w2[1]) + self.b2[0])
        o2i = (torch.einsum('...bi,bio->...bo', o1i, self.w2[0]) +
               torch.einsum('...bi,bio->...bo', o1r, self.w2[1]) + self.b2[1])

        o2 = F.softshrink(torch.stack([o2r, o2i], dim=-1),
                          lambd=self.sparsity_threshold)
        o2 = torch.view_as_complex(o2)

        out = torch.zeros(B, H, W // 2 + 1, self.num_blocks, self.block_size,
                          dtype=torch.cfloat, device=x.device)
        out[:, h_lo:h_hi, :kept_modes] = o2
        out = out.reshape(B, H, W // 2 + 1, C)

        x = torch.fft.irfft2(out, s=(H, W), dim=(1, 2), norm="ortho")
        return x.type(dtype) + bias


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features, drop=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden_features, in_features),
            nn.Dropout(drop),
        )

    def forward(self, x):
        return self.net(x)


class AFNOBlock(nn.Module):
    def __init__(self, embed_dim, num_blocks=8, mlp_ratio=4.0, drop=0.0,
                 sparsity_threshold=0.01, hard_thresholding_fraction=1.0,
                 double_skip=True):
        super().__init__()
        self.double_skip = double_skip
        self.norm1  = nn.LayerNorm(embed_dim)
        self.filter = AFNO2D(embed_dim, num_blocks, sparsity_threshold,
                             hard_thresholding_fraction)
        self.norm2  = nn.LayerNorm(embed_dim)
        self.mlp    = Mlp(embed_dim, int(embed_dim * mlp_ratio), drop)

    def forward(self, x):
        residual = x
        x = self.filter(self.norm1(x))
        if self.double_skip:
            x        = x + residual
            residual = x
        x = self.mlp(self.norm2(x)) + residual
        return x


class PatchEmbed(nn.Module):
    """
    (B, C_in, H, W) → (B, Hp, Wp, embed_dim)
    Strided Conv2d followed by permute.
    """
    def __init__(self, in_channels, embed_dim, patch_size):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        return self.proj(x).permute(0, 2, 3, 1).contiguous()


@register_model("afno")
class AFNONet(nn.Module):
    def __init__(
        self,
        in_channels,               # C_in per frame (e.g. 21)
        height,                    # H
        width,                     # W
        embed_dim   = 432,         # E  (must be divisible by num_blocks)
        depth       = 8,           # D
        mlp_ratio   = 4.0,
        num_blocks  = 8,           # AFNO internal blocks (E % num_blocks == 0)
        sparsity_threshold          = 0.01,
        hard_thresholding_fraction  = 1.0,
        dropout     = 0.0,
        patch_size  = 2,
        double_skip = True,
        num_targets = 1,
        Tin         = 6,
        Tout        = 6,
        time_embed_d = 16,
        **kwargs,
    ):
        super().__init__()
        assert embed_dim % num_blocks == 0, \
            f"embed_dim {embed_dim} must be divisible by num_blocks {num_blocks}"
        assert height % patch_size == 0 and width % patch_size == 0, \
            f"H={height}, W={width} must be divisible by patch_size={patch_size}"

        self.Tout        = Tout
        self.num_targets = num_targets
        self.patch_size  = patch_size
        self.Hp          = height // patch_size
        self.Wp          = width  // patch_size

        self.time_embed = AbsoluteTimeEmbedding(d=time_embed_d)
        d_time  = self.time_embed.embed_dim        
        eff_in  = Tin * (in_channels + 2 + d_time)  

        self.patch_embed = PatchEmbed(eff_in, embed_dim, patch_size)
        self.pos_embed   = nn.Parameter(
            torch.zeros(1, self.Hp, self.Wp, embed_dim))
        self.pos_drop    = nn.Dropout(p=dropout)

        self.blocks = nn.ModuleList([
            AFNOBlock(embed_dim, num_blocks, mlp_ratio, dropout,
                      sparsity_threshold, hard_thresholding_fraction,
                      double_skip)
            for _ in range(depth)
        ])

        # output head
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(
            embed_dim,
            Tout * num_targets * patch_size * patch_size,
            bias=False,
        )

        trunc_normal_(self.pos_embed, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor,
                t_in: torch.Tensor = None) -> torch.Tensor:
        """
        x    : (B, Tin, H, W, C_in)
        t_in : (B, Tin) int64 or None
        out  : (B, Tout, H, W, num_targets)
        """
        B, Tin, H, W, _ = x.shape
        p = self.patch_size

        # Stack time into channels 
        grid = get_grid(B, H, W, x.device)                   # (B, H, W, 2)
        te   = self.time_embed(t_in) if t_in is not None else None
        x_s  = stack_time_into_channels(x, grid=grid, time_embed=te)
        # (B, H, W, eff_in)

        # Patch embed 
        x_s = x_s.permute(0, 3, 1, 2).contiguous()           # (B, eff_in, H, W)
        x_s = self.patch_embed(x_s)                           # (B, Hp, Wp, E)
        x_s = self.pos_drop(x_s + self.pos_embed)

        # AFNO blocks 
        for blk in self.blocks:
            x_s = blk(x_s)

        x_s = self.norm(x_s)


        x_s = self.head(x_s)

        x_s = x_s.reshape(B, self.Hp, self.Wp,
                           self.Tout, p, p, self.num_targets)

        x_s = x_s.permute(0, 3, 1, 4, 2, 5, 6).contiguous()

        x_s = x_s.reshape(B, self.Tout, H, W, self.num_targets)
        # (B, Tout, H, W, num_targets)

        return x_s