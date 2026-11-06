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
    """Channel-wise MLP via 1×1 convolutions."""

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
        self.modes1       = modes1
        self.modes2       = modes2
        scale = 1 / (in_channels * out_channels)
        self.weights = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes1, modes2,
                                dtype=torch.cfloat)
        )

    def forward(self, x):
        B, C, H, W = x.shape
        x_ft   = torch.fft.rfft2(x, dim=(-2, -1))
        out_ft = torch.zeros(B, self.out_channels, H, W // 2 + 1,
                             dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = torch.einsum(
            "bixy,ioxy->boxy",
            x_ft[:, :, :self.modes1, :self.modes2],
            self.weights,
        )
        return torch.fft.irfft2(out_ft, s=(H, W))


class UNOBlock2d(nn.Module):
    """
    Single UNO block: spectral conv + pointwise residual,
    optional bilinear spatial resampling, then channel MLP residual.

    Resampling is applied after the GELU so the MLP always
    operates at the output resolution.
    """

    def __init__(self, in_channels, out_channels, modes1, modes2,
                 scaling_factor=1.0,
                 channel_mlp_expansion=0.5,
                 channel_mlp_dropout=0.0):
        super().__init__()
        self.scaling_factor = scaling_factor

        self.conv = SpectralConv2d(in_channels, out_channels, modes1, modes2)
        self.w    = nn.Conv2d(in_channels, out_channels, 1)
        self.mlp  = ChannelMLP(
            in_channels     = out_channels,
            hidden_channels = max(1, int(out_channels * channel_mlp_expansion)),
            out_channels    = out_channels,
            n_layers        = 2,
            dropout         = channel_mlp_dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.gelu(self.conv(x) + self.w(x))

        if self.scaling_factor != 1.0:
            new_H = max(1, int(round(x.shape[-2] * self.scaling_factor)))
            new_W = max(1, int(round(x.shape[-1] * self.scaling_factor)))
            x = F.interpolate(x, size=(new_H, new_W),
                              mode='bilinear', align_corners=False)

        return x + self.mlp(x)


@register_model("uno2d")
class UNO2d(nn.Module):
    def __init__(
        self,
        in_channels,                           
        height,                                
        width,                                  
        hidden_channels          = 96,
        n_layers                 = 4,
        uno_out_channels         = None,        
        uno_n_modes              = None,        
        uno_scalings             = None,       
        lifting_channel_ratio    = 2,
        projection_channel_ratio = 2,
        channel_mlp_expansion    = 0.5,
        channel_mlp_dropout      = 0.0,
        num_targets              = 1,
        Tin                      = 6,
        Tout                     = 6,
        time_embed_d             = 16,
        **kwargs,
    ):
        super().__init__()
        self.Tin         = Tin
        self.Tout        = Tout
        self.num_targets = num_targets

        self.time_embed = AbsoluteTimeEmbedding(d=time_embed_d)
        d_time = self.time_embed.embed_dim   # 16
        eff_in = Tin * (in_channels + 2 + d_time)

        W  = hidden_channels
        lc = int(W * lifting_channel_ratio)
        pc = int(W * projection_channel_ratio)

        if uno_out_channels is None:
            uno_out_channels = [W] * n_layers
        if uno_n_modes is None:
            uno_n_modes = [[16, 16]] * n_layers
        if uno_scalings is None:
            uno_scalings = [[1, 1], [0.5, 0.5], [2, 2], [1, 1]]

        assert len(uno_out_channels) == n_layers, "uno_out_channels length mismatch"
        assert len(uno_n_modes)      == n_layers, "uno_n_modes length mismatch"
        assert len(uno_scalings)     == n_layers, "uno_scalings length mismatch"


        self.horizontal_skips_map = {
            n_layers - i - 1: i for i in range(n_layers // 2)
        }

        self.lifting = ChannelMLP(
            in_channels     = eff_in,
            hidden_channels = lc,
            out_channels    = W,
            n_layers        = 2,
            dropout         = channel_mlp_dropout,
        )

        self.blocks     = nn.ModuleList()
        self.skip_convs = nn.ModuleDict()

        prev_out = W
        for i in range(n_layers):
            if i in self.horizontal_skips_map:
                skip_from = self.horizontal_skips_map[i]
                prev_out  = prev_out + uno_out_channels[skip_from]

            self.blocks.append(UNOBlock2d(
                in_channels           = prev_out,
                out_channels          = uno_out_channels[i],
                modes1                = uno_n_modes[i][0],
                modes2                = uno_n_modes[i][1],
                scaling_factor        = uno_scalings[i][0],   # isotropic
                channel_mlp_expansion = channel_mlp_expansion,
                channel_mlp_dropout   = channel_mlp_dropout,
            ))

            if i in self.horizontal_skips_map.values():
                self.skip_convs[str(i)] = nn.Conv2d(
                    uno_out_channels[i], uno_out_channels[i], 1)

            prev_out = uno_out_channels[i]

        self.proj1 = nn.Conv2d(prev_out, pc,                 1)
        self.proj2 = nn.Conv2d(pc,       Tout * num_targets, 1)

    def forward(self, x: torch.Tensor,
                t_in: torch.Tensor = None, **kwargs) -> torch.Tensor:
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

        skip_outputs = {}

        for i, block in enumerate(self.blocks):
            if i in self.horizontal_skips_map:
                skip_from = self.horizontal_skips_map[i]
                skip      = skip_outputs[skip_from]
                if skip.shape[-2:] != x_s.shape[-2:]:
                    skip = F.interpolate(skip, size=x_s.shape[-2:],
                                         mode='bilinear', align_corners=False)
                x_s = torch.cat([x_s, skip], dim=1)

            x_s = block(x_s)

            if str(i) in self.skip_convs:
                skip_outputs[i] = self.skip_convs[str(i)](x_s)

        x_s = F.gelu(self.proj1(x_s))
        x_s = self.proj2(x_s)                          # (B, Tout*num_tgt, H, W)

        y = x_s.permute(0, 2, 3, 1).contiguous()       # (B, H, W, Tout*num_tgt)
        return unstack_channels_into_time(y, self.Tout, self.num_targets)