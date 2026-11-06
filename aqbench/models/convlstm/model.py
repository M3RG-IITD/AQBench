import torch
import torch.nn as nn
from aqbench.registry.model_registry import register_model
from aqbench.models.base import AbsoluteTimeEmbedding, get_grid


class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, height, width, bias=True, layer_norm=False):
        super().__init__()
        self.hidden_dim = hidden_dim
        padding = kernel_size[0] // 2, kernel_size[1] // 2

        self.conv = nn.Conv2d(
            in_channels  = input_dim + hidden_dim,
            out_channels = 4 * hidden_dim,
            kernel_size  = kernel_size,
            padding      = padding,
            bias         = bias,
        )
        self.norm = nn.GroupNorm(1, 4 * hidden_dim) if layer_norm else nn.Identity()

    def forward(self, x, h, c):
        combined = self.norm(self.conv(torch.cat([x, h], dim=1)))
        i, f, o, g = torch.split(combined, self.hidden_dim, dim=1)
        c_next = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
        h_next = torch.sigmoid(o) * torch.tanh(c_next)
        return h_next, c_next


@register_model("convlstm")
class ConvLSTM(nn.Module):

    def __init__(
        self,
        in_dim,            
        out_channels,      
        in_channels,        
        height,            
        width,             
        num_layers   = 4,
        num_hidden   = 64,
        kernel_size  = (5, 5),
        bias         = True,
        layer_norm   = False,
        time_embed_d = 16,
        num_targets  = 1,
        **kwargs,
    ):
        super().__init__()

        self.T_in        = in_dim
        self.T_out       = out_channels
        self.num_targets = num_targets

        self.time_embed = AbsoluteTimeEmbedding(d=time_embed_d)
        d_time          = self.time_embed.embed_dim  # 16

        eff_in = in_channels + 2 + d_time

        if isinstance(num_hidden, int):
            num_hidden = [num_hidden] * num_layers
        if isinstance(kernel_size, (tuple, list)) and not isinstance(kernel_size[0], (tuple, list)):
            kernel_size = [tuple(kernel_size)] * num_layers 

        assert len(num_hidden) == num_layers, "num_hidden length must equal num_layers"
        assert len(kernel_size) == num_layers, "kernel_size length must equal num_layers"

        self.num_layers = num_layers
        self.num_hidden = num_hidden

        cell_list = []
        for i in range(num_layers):
            in_ch = eff_in if i == 0 else num_hidden[i - 1]
            cell_list.append(
                ConvLSTMCell(
                    input_dim  = in_ch,
                    hidden_dim = num_hidden[i],
                    kernel_size = kernel_size[i],
                    height     = height,
                    width      = width,
                    bias       = bias,
                    layer_norm = layer_norm,
                )
            )
        self.cell_list = nn.ModuleList(cell_list)

        self.conv_last = nn.Conv2d(
            num_hidden[-1],
            out_channels * num_targets,
            kernel_size = 1,
            bias        = bias,
        )

    def forward(self, x: torch.Tensor, t_in: torch.Tensor = None, **kwargs) -> torch.Tensor:
        B, T, H, W, C = x.shape
        device = x.device

        # (B, T, C, H, W)
        x = x.permute(0, 1, 4, 2, 3).contiguous()

        # grid: (B, 2, H, W)
        grid = get_grid(B, H, W, device).permute(0, 3, 1, 2).contiguous()

        # time embedding: (B, T, d_time)
        if t_in is not None:
            te = self.time_embed(t_in)
        else:
            te = torch.zeros(B, T, self.time_embed.embed_dim, device=device)

        grid_exp = grid.unsqueeze(1).expand(-1, T, -1, -1, -1)        # (B, T, 2, H, W)
        te_exp   = te[:, :, :, None, None].expand(-1, -1, -1, H, W)   # (B, T, d_time, H, W)

        x_aug = torch.cat([x, grid_exp, te_exp], dim=2)

        h_t = [torch.zeros(B, self.num_hidden[i], H, W, device=device) for i in range(self.num_layers)]
        c_t = [torch.zeros(B, self.num_hidden[i], H, W, device=device) for i in range(self.num_layers)]

        for t in range(self.T_in):
            h_t[0], c_t[0] = self.cell_list[0](x_aug[:, t], h_t[0], c_t[0])
            for i in range(1, self.num_layers):
                h_t[i], c_t[i] = self.cell_list[i](h_t[i - 1], h_t[i], c_t[i])

        # decode: project final hidden state
        # (B, Tout*num_targets, H, W) → (B, Tout, H, W, num_targets)
        out = self.conv_last(h_t[-1])
        out = out.permute(0, 2, 3, 1).contiguous()
        out = out.reshape(B, H, W, self.T_out, self.num_targets)
        out = out.permute(0, 3, 1, 2, 4).contiguous()

        return out