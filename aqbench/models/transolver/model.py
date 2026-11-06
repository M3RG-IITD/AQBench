import torch
import torch.nn as nn
from einops import rearrange

from aqbench.registry.model_registry import register_model
from aqbench.models.base import (
    AbsoluteTimeEmbedding,
    get_grid,
    stack_time_into_channels,
    unstack_channels_into_time,
)

ACTIVATION = {"gelu": nn.GELU, "relu": nn.ReLU, "silu": nn.SiLU}



class Physics_Attention_Structured_Mesh_2D(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.,
                 slice_num=32, shapelist=None, kernel=3):
        super().__init__()
        inner_dim        = dim_head * heads
        self.dim_head    = dim_head
        self.heads       = heads
        self.scale       = dim_head ** -0.5
        self.softmax     = nn.Softmax(dim=-1)
        self.dropout     = nn.Dropout(dropout)
        self.temperature = nn.Parameter(torch.ones([1, heads, 1, 1]) * 0.5)
        self.H           = shapelist[0]
        self.W           = shapelist[1]

        self.in_project_x     = nn.Conv2d(dim, inner_dim, kernel, 1, kernel // 2)
        self.in_project_fx    = nn.Conv2d(dim, inner_dim, kernel, 1, kernel // 2)
        self.in_project_slice = nn.Linear(dim_head, slice_num)
        self.to_q  = nn.Linear(dim_head, dim_head, bias=False)
        self.to_k  = nn.Linear(dim_head, dim_head, bias=False)
        self.to_v  = nn.Linear(dim_head, dim_head, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim), nn.Dropout(dropout))

    def forward(self, x):
        B, N, C = x.shape
        x = x.reshape(B, self.H, self.W, C).permute(0, 3, 1, 2)  

        fx_mid = self.in_project_fx(x).permute(0, 2, 3, 1).reshape(
            B, N, self.heads, self.dim_head).permute(0, 2, 1, 3)
        x_mid  = self.in_project_x(x).permute(0, 2, 3, 1).reshape(
            B, N, self.heads, self.dim_head).permute(0, 2, 1, 3)

        slice_weights = self.softmax(
            self.in_project_slice(x_mid) /
            torch.clamp(self.temperature, 0.1, 5))
        slice_norm  = slice_weights.sum(2)
        slice_token = torch.einsum("bhnc,bhng->bhgc", fx_mid, slice_weights)
        slice_token = slice_token / (slice_norm[..., None] + 1e-5)

        q, k, v = self.to_q(slice_token), self.to_k(slice_token), self.to_v(slice_token)
        attn = self.softmax(torch.matmul(q, k.transpose(-1, -2)) * self.scale)
        out  = torch.matmul(self.dropout(attn), v)
        out  = torch.einsum("bhgc,bhng->bhnc", out, slice_weights)
        return self.to_out(rearrange(out, "b h n d -> b n (h d)"))



class MLP(nn.Module):
    def __init__(self, n_input, n_hidden, n_output,
                 n_layers=1, act="gelu", res=True):
        super().__init__()
        act = ACTIVATION[act]
        self.linear_pre  = nn.Sequential(nn.Linear(n_input, n_hidden), act())
        self.linear_post = nn.Linear(n_hidden, n_output)
        self.n_layers    = n_layers
        self.res         = res
        self.linears     = nn.ModuleList([
            nn.Sequential(nn.Linear(n_hidden, n_hidden), act())
            for _ in range(n_layers)
        ])

    def forward(self, x):
        x = self.linear_pre(x)
        for layer in self.linears:
            x = x + layer(x) if self.res else layer(x)
        return self.linear_post(x)


class TransolverBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout, mlp_ratio,
                 slice_num, last_layer=False, out_dim=1, shapelist=None):
        super().__init__()
        self.last_layer = last_layer
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.attn = Physics_Attention_Structured_Mesh_2D(
            hidden_dim, heads=num_heads, dim_head=hidden_dim // num_heads,
            dropout=dropout, slice_num=slice_num, shapelist=shapelist)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.mlp = MLP(hidden_dim, hidden_dim * mlp_ratio, hidden_dim,
                       n_layers=0, res=False)
        if last_layer:
            self.ln3 = nn.LayerNorm(hidden_dim)
            self.out = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        if self.last_layer:
            x = self.out(self.ln3(x))
        return x



@register_model("transolver")
class Transolver(nn.Module):
    def __init__(
        self,
        in_channels,                 
        height,
        width,
        n_hidden   = 256,
        n_layers   = 8,
        n_heads    = 8,
        slice_num  = 64,
        mlp_ratio  = 4,
        dropout    = 0.0,
        num_targets = 1,
        Tin        = 6,
        Tout       = 6,
        time_embed_d = 16,
        **kwargs,
    ):
        super().__init__()
        self.Tin         = Tin
        self.Tout        = Tout
        self.H           = height
        self.W           = width
        self.num_targets = num_targets

        self.time_embed = AbsoluteTimeEmbedding(d=time_embed_d)
        d_time  = self.time_embed.embed_dim          
        eff_in  = Tin * (in_channels + 2 + d_time)   
        out_dim = Tout * num_targets

        self.preprocess  = MLP(eff_in, n_hidden * 2, n_hidden, n_layers=0, res=False)
        self.placeholder = nn.Parameter((1 / n_hidden) * torch.rand(n_hidden))

        self.blocks = nn.ModuleList([
            TransolverBlock(
                hidden_dim=n_hidden, num_heads=n_heads, dropout=dropout,
                mlp_ratio=mlp_ratio, slice_num=slice_num,
                last_layer=(i == n_layers - 1),
                out_dim=out_dim, shapelist=[height, width])
            for i in range(n_layers)
        ])

    def forward(self, x: torch.Tensor,
                t_in: torch.Tensor = None) -> torch.Tensor:

        B, Tin, H, W, _ = x.shape

        grid = get_grid(B, H, W, x.device)                   
        te   = self.time_embed(t_in) if t_in is not None else None
        x_s  = stack_time_into_channels(x, grid=grid, time_embed=te)

        x_tok = x_s.reshape(B, H * W, -1)
        x_tok = self.preprocess(x_tok) + self.placeholder[None, None, :]

        for blk in self.blocks:
            x_tok = blk(x_tok)

        y = x_tok.reshape(B, H, W, self.Tout * self.num_targets)
        return unstack_channels_into_time(y, self.Tout, self.num_targets)