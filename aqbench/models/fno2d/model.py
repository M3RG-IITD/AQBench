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

class ChannelMLP(nn.Module):
    """
    MLP applied channel-wise via 1×1 convolutions.

    dropout : applied after GELU between every pair of hidden layers.
    """
    def __init__(self, in_channels, out_channels, hidden_channels=None,
                 n_layers=2, dropout=0.0):
        super().__init__()
        hidden_channels = hidden_channels or in_channels
        layers = []
        for i in range(n_layers):
            c_in  = in_channels  if i == 0            else hidden_channels
            c_out = out_channels if i == n_layers - 1 else hidden_channels
            layers.append(nn.Conv1d(c_in, c_out, 1))
        self.layers   = nn.ModuleList(layers)
        self.n_layers = n_layers
        self.drop     = nn.Dropout(p=dropout) if dropout > 0.0 else None

    def forward(self, x):
        # x: (B, C, H, W)  or  (B, C, N)
        shape = x.shape
        if x.ndim == 4:
            x = x.reshape(shape[0], shape[1], -1)
        for i, fc in enumerate(self.layers):
            x = fc(x)
            if i < self.n_layers - 1:
                x = F.gelu(x)
                if self.drop is not None:
                    x = self.drop(x)
        if len(shape) == 4:
            x = x.reshape(shape[0], x.shape[1], shape[2], shape[3])
        return x


class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super().__init__()
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        scale = 1 / (in_channels * out_channels)
        self.weights = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes1, modes2,
                                dtype=torch.cfloat)
        )

    def forward(self, x):
        B, C, H, W = x.shape
        x_ft = torch.fft.rfft2(x, dim=(-2, -1))
        out_ft = torch.zeros(B, self.out_channels, H, W // 2 + 1,
                             dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = torch.einsum(
            "bixy,ioxy->boxy",
            x_ft[:, :, :self.modes1, :self.modes2],
            self.weights,
        )
        return torch.fft.irfft2(out_ft, s=(H, W))



class FNOBlock2d(nn.Module):
    def __init__(self, width, modes1, modes2,
                 channel_mlp_expansion=0.5, channel_mlp_dropout=0.0):
        super().__init__()
        self.conv = SpectralConv2d(width, width, modes1, modes2)
        self.w    = nn.Conv2d(width, width, 1)
        self.mlp  = ChannelMLP(
            in_channels=width,
            hidden_channels=int(width * channel_mlp_expansion),
            out_channels=width,
            n_layers=2,
            dropout=channel_mlp_dropout,
        )

    def forward(self, x):
        x = F.gelu(self.conv(x) + self.w(x))
        return x + self.mlp(x)


@register_model("fno2d")
class FNO2d(nn.Module):
    def __init__(
        self,
        in_channels,                   
        modes1,
        modes2,
        hidden_channels         = 128,
        n_layers                = 6,
        lifting_channel_ratio   = 2,
        projection_channel_ratio = 2,
        channel_mlp_expansion   = 0.5,
        channel_mlp_dropout     = 0.0,  
        num_targets  = 1,
        Tin          = 6,
        Tout         = 6,
        time_embed_d = 16,
        **kwargs,
    ):
        super().__init__()
        self.Tin         = Tin
        self.Tout        = Tout
        self.num_targets = num_targets

        self.time_embed = AbsoluteTimeEmbedding(d=time_embed_d)
        d_time = self.time_embed.embed_dim   

        eff_in = Tin * (in_channels + 2 + d_time)

        W  = hidden_channels
        lc = int(W * lifting_channel_ratio)
        pc = int(W * projection_channel_ratio)

        self.lifting = ChannelMLP(
            in_channels=eff_in, hidden_channels=lc, out_channels=W,
            n_layers=2, dropout=channel_mlp_dropout)

        self.blocks = nn.ModuleList([
            FNOBlock2d(W, modes1, modes2,
                       channel_mlp_expansion=channel_mlp_expansion,
                       channel_mlp_dropout=channel_mlp_dropout)
            for _ in range(n_layers)
        ])

        self.proj1 = nn.Conv2d(W,  pc,                 1)
        self.proj2 = nn.Conv2d(pc, Tout * num_targets, 1)

    def forward(self, x: torch.Tensor,
                t_in: torch.Tensor = None) -> torch.Tensor:
        """
        x    : (B, Tin, H, W, C_in)
        t_in : (B, Tin) int64 or None
        out  : (B, Tout, H, W, num_targets)
        """
        B, Tin, H, W, _ = x.shape

        grid = get_grid(B, H, W, x.device)                  
        te   = self.time_embed(t_in) if t_in is not None else None
        x_s  = stack_time_into_channels(x, grid=grid, time_embed=te)

        x_s = x_s.permute(0, 3, 1, 2).contiguous()          
        x_s = self.lifting(x_s)                               

        for block in self.blocks:
            x_s = block(x_s)

        x_s = F.gelu(self.proj1(x_s))
        x_s = self.proj2(x_s)                                 # (B, Tout*num_tgt, H, W)

        y = x_s.permute(0, 2, 3, 1).contiguous()              # (B, H, W, Tout*num_tgt)
        return unstack_channels_into_time(y, self.Tout, self.num_targets)