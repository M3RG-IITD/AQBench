import torch
import torch.nn as nn
import torch.nn.functional as F
import torchcomplex.nn as cn
from torch_frft.dfrft_module import dfrft

from aqbench.registry.model_registry import register_model
from aqbench.models.base import (
    AbsoluteTimeEmbedding,
    get_grid,
    stack_time_into_channels,
    unstack_channels_into_time,
)



class ComplexGELU(nn.Module):
    def forward(self, x):
        return torch.complex(F.gelu(x.real), F.gelu(x.imag))



class SpectralConv2d_CONO(nn.Module):
    def __init__(self, width, modes):
        super().__init__()
        self.modes = modes
        scale = 1 / (width * width)
        self.w1 = nn.Parameter(scale * torch.rand(
            width, width, modes, modes, dtype=torch.cfloat))
        self.w2 = nn.Parameter(scale * torch.rand(
            width, width, modes, modes, dtype=torch.cfloat))

    @staticmethod
    def compl_mul2d(inp, w):
        # (B, in, H, W), (in, out, H, W) → (B, out, H, W)
        return torch.einsum("bixy,ioxy->boxy", inp, w)

    def forward(self, x):
        # x: (B, W, H, Ws) complex
        x_ft   = torch.fft.fft2(x, dim=[-2, -1])
        out_ft = torch.zeros_like(x_ft)
        out_ft[:, :,  :self.modes, :self.modes] = self.compl_mul2d(
            x_ft[:, :,  :self.modes, :self.modes], self.w1)
        out_ft[:, :, -self.modes:, :self.modes] = self.compl_mul2d(
            x_ft[:, :, -self.modes:, :self.modes], self.w2)
        return torch.fft.ifft2(out_ft, dim=[-2, -1])



class SpectralFractionalConv2d(nn.Module):
    def __init__(self, width, alpha=0.5):
        super().__init__()
        self.alpha1 = nn.Parameter(torch.tensor(alpha, dtype=torch.float32))
        self.alpha2 = nn.Parameter(torch.tensor(alpha, dtype=torch.float32))
        self.layer  = cn.Conv2d(width, width, 1)

    def forward(self, x):
        x_ft   = dfrft(dfrft(x, self.alpha1, dim=-1), self.alpha2, dim=-2)
        out_ft = self.layer(x_ft)
        return dfrft(dfrft(out_ft, -self.alpha2, dim=-2), -self.alpha1, dim=-1)



class ComplexBlock(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm2d(width), nn.GELU(),
            nn.Conv2d(width, width, 3, padding=1),
            nn.BatchNorm2d(width), nn.GELU(),
            nn.Conv2d(width, width, 3, padding=1),
        )

    def forward(self, x):
        return self.block(x)



class ConvBNActivation(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.conv = cn.Conv2d(width, width, 1)
        self.bn   = cn.BatchNorm2d(width)
        self.act  = ComplexGELU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))



class ComplexChannelMixing(nn.Module):
    """
    Applies a complex linear transformation across the channel dimension for mixing
    """
    def __init__(self, width):
        super().__init__()
        self.linear = cn.Linear(width, width)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.linear(x)
        return x.permute(0, 3, 1, 2).contiguous()



@register_model("cono")
class CONO2D(nn.Module):
    def __init__(
        self,
        in_channels,         
        height,              
        width_spatial,      
        num_basis   = 10,    
        d_model     = 134,   
        n_layers    = 4,
        mlp_hidden  = 128,
        num_targets = 1,
        Tin         = 6,
        Tout        = 6,
        time_embed_d = 16,
        **kwargs,
    ):
        super().__init__()
        W                = d_model
        self.Tout        = Tout
        self.num_targets = num_targets
        self.act         = ComplexGELU()

        self.time_embed = AbsoluteTimeEmbedding(d=time_embed_d)
        d_time  = self.time_embed.embed_dim
        eff_in  = Tin * (in_channels + 2 + d_time)
        out_dim = Tout * num_targets

        self.fc0 = nn.Linear(eff_in, W)

        self.complexblock = ComplexBlock(W)
        self.residual     = ConvBNActivation(W)

        self.convs   = nn.ModuleList([
            SpectralConv2d_CONO(W, num_basis)        for _ in range(n_layers)])
        self.fconvs  = nn.ModuleList([
            SpectralFractionalConv2d(W)              for _ in range(n_layers)])
        self.ws      = nn.ModuleList([
            cn.Conv2d(W, W, 1)                       for _ in range(n_layers)])
        
        self.mixings = nn.ModuleList([
            ComplexChannelMixing(W)                  for _ in range(n_layers)])

        self.fc1        = nn.Linear(W,          mlp_hidden)
        self.fc2        = nn.Linear(mlp_hidden, mlp_hidden)
        self.final_proj = nn.Linear(mlp_hidden, out_dim)

    def forward(self, x: torch.Tensor,
                t_in: torch.Tensor = None) -> torch.Tensor:
        """
        x    : (B, Tin, H, W_spatial, C_in)
        t_in : (B, Tin)
        out  : (B, Tout, H, W_spatial, num_targets)
        """
        B, Tin, H, Ws, _ = x.shape

        grid = get_grid(B, H, Ws, x.device)
        te   = self.time_embed(t_in) if t_in is not None else None
        x_s  = stack_time_into_channels(x, grid=grid, time_embed=te)

        # Lift
        x_s = self.fc0(x_s)
        x_s = x_s.permute(0, 3, 1, 2).contiguous()

        # build complex 
        x_i = self.complexblock(x_s)
        x_s = torch.complex(x_s, x_i)
        x_s = self.residual(x_s)

        # CONO blocks 
        for i in range(len(self.convs)):
            # 1. Summation of standard conv, fractional conv, and 1x1 bypass
            x_s = self.convs[i](x_s) + self.ws[i](x_s) + self.fconvs[i](x_s)
            
            # 2. Complex Mixing step
            x_s = self.mixings[i](x_s)
            
            # 3. Activation
            if i < len(self.convs) - 1:
                x_s = self.act(x_s)

        x_s = x_s.real

        # output head
        x_s = x_s.permute(0, 2, 3, 1).contiguous()                           # (B, H, Ws, W)
        x_s = F.gelu(self.fc1(x_s))                                          # (B, H, Ws, mlp_hidden)
        x_s = F.gelu(self.fc2(x_s))                                          # (B, H, Ws, mlp_hidden)
        x_s = self.final_proj(x_s)                                           # (B, H, Ws, Tout*num_tgt)

        return unstack_channels_into_time(x_s, self.Tout, self.num_targets)