"""
SimVP adapted for bundled spatiotemporal I/O — aqbench conventions.

Reference
---------
    SimVP: Simpler yet Better Video Prediction
    https://arxiv.org/abs/2206.05099
"""

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



class ConvSC(nn.Module):
    """
    Single Conv2d → GroupNorm → SiLU block.
    stride > 1  → downsampling (encoder).
    transpose=True → ConvTranspose2d upsampling (decoder).
    """

    def __init__(self, C_in, C_out, stride=1, transpose=False, num_groups=8):
        super().__init__()
        if transpose:
            self.conv = nn.ConvTranspose2d(
                C_in, C_out, kernel_size=3,
                stride=stride, padding=1,
                output_padding=stride - 1, bias=False,
            )
        else:
            self.conv = nn.Conv2d(
                C_in, C_out, kernel_size=3,
                stride=stride, padding=1, bias=False,
            )
        g = self._groups(C_out, num_groups)
        self.norm = nn.GroupNorm(g, C_out)
        self.act  = nn.SiLU()

    @staticmethod
    def _groups(channels, target):
        g = min(target, channels)
        while channels % g != 0:
            g -= 1
        return g

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class Inception(nn.Module):

    def __init__(self, in_ch, hid_ch, out_ch,
                 incep_ker=(3, 5, 7, 11), groups=8):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, hid_ch, 1, groups=1, bias=False)
        self.branches = nn.ModuleList([
            nn.Conv2d(
                hid_ch, hid_ch,
                kernel_size=k, padding=k // 2,
                groups=groups, bias=False,
            )
            for k in incep_ker
        ])
        self.conv2 = nn.Conv2d(
            hid_ch * len(incep_ker), out_ch, 1, groups=1, bias=False
        )
        g_out = self._groups(out_ch, groups)
        self.norm = nn.GroupNorm(g_out, out_ch)
        self.act  = nn.SiLU()

    @staticmethod
    def _groups(channels, target):
        g = min(target, channels)
        while channels % g != 0:
            g -= 1
        return g

    def forward(self, x):
        x = self.conv1(x)
        x = torch.cat([b(x) for b in self.branches], dim=1)
        return self.act(self.norm(self.conv2(x)))


def stride_generator(N, reverse=False):
    strides = ([1, 2] * 10)[:N]
    return list(reversed(strides)) if reverse else strides



class SpatialEncoder(nn.Module):
    """
    Encodes a single spatial frame: (B, C_in, H, W) to latent, enc1.

    latent : (B, hid_S, H', W')   H' = H / 2^(#stride-2 steps)
    enc1   : (B, hid_S, H,  W)   output of first ConvSC (for skip)
    """

    def __init__(self, C_in, hid_S, N_S, num_groups=8):
        super().__init__()
        strides = stride_generator(N_S)
        self.enc = nn.Sequential(
            ConvSC(C_in,  hid_S, stride=strides[0], num_groups=num_groups),
            *[ConvSC(hid_S, hid_S, stride=s, num_groups=num_groups)
              for s in strides[1:]],
        )

    def forward(self, x):
        enc1   = self.enc[0](x)
        latent = enc1
        for layer in self.enc[1:]:
            latent = layer(latent)
        return latent, enc1          


class SpatialDecoder(nn.Module):
    """
    Decodes a single spatial latent back to frame space.

    (B, hid_S, H', W') + enc1 (B, hid_S, H, W) → (B, C_out, H, W)

    Last ConvSC receives [hid | skip] → 2*hid_S channels.
    """

    def __init__(self, hid_S, C_out, N_S, num_groups=8):
        super().__init__()
        strides = stride_generator(N_S, reverse=True)
        self.dec = nn.Sequential(
            *[ConvSC(hid_S,    hid_S, stride=s, transpose=True,
                     num_groups=num_groups)
              for s in strides[:-1]],
            ConvSC(2 * hid_S, hid_S, stride=strides[-1], transpose=True,
                   num_groups=num_groups),
        )
        self.readout = nn.Conv2d(hid_S, C_out, kernel_size=1)

    def forward(self, hid, enc1):
        for layer in self.dec[:-1]:
            hid = layer(hid)
        if hid.shape != enc1.shape:
            hid = F.interpolate(hid, size=enc1.shape[2:],
                                mode='bilinear', align_corners=True)
        hid = self.dec[-1](torch.cat([hid, enc1], dim=1))
        return self.readout(hid)




class Mid_Xnet(nn.Module):
    """
    Temporal U-Net operating in latent spatial space.

    Input  : (B, T*hid_S, H', W')   — time stacked into channels
    Output : (B, T*hid_S, H', W')   — same shape

    N_T encoder Inception blocks, N_T decoder Inception blocks,
    with skip connections between symmetric layers.
    """

    def __init__(self, channel_in, channel_hid, N_T,
                 incep_ker=(3, 5, 7, 11), groups=8):
        super().__init__()
        assert N_T >= 2, "N_T must be at least 2"
        self.N_T = N_T

        # ── Encoder ───────────────────────────────────────────────────────
        enc_layers = [Inception(channel_in,  channel_hid // 2,
                                channel_hid, incep_ker, groups)]
        for _ in range(1, N_T):
            enc_layers.append(Inception(channel_hid, channel_hid // 2,
                                        channel_hid, incep_ker, groups))
        self.enc = nn.Sequential(*enc_layers)

        # ── Decoder ───────────────────────────────────────────────────────
        dec_layers = [Inception(channel_hid,     channel_hid // 2,
                                channel_hid,     incep_ker, groups)]
        for _ in range(1, N_T - 1):
            dec_layers.append(Inception(2 * channel_hid, channel_hid // 2,
                                        channel_hid,     incep_ker, groups))
        dec_layers.append(Inception(2 * channel_hid, channel_hid // 2,
                                    channel_in,       incep_ker, groups))
        self.dec = nn.Sequential(*dec_layers)

    def forward(self, x):
        """x : (B, T*hid_S, H', W')"""
        # ── Encoder with skip collection ──────────────────────────────────
        skips = []
        z = x
        for i, layer in enumerate(self.enc):
            z = layer(z)
            if i < self.N_T - 1:
                skips.append(z)

        # ── Decoder with skip injection ───────────────────────────────────
        z = self.dec[0](z)
        for i in range(1, self.N_T):
            z = self.dec[i](torch.cat([z, skips[-(i)]], dim=1))

        return z


@register_model("simvp")
class SimVP(nn.Module):
    """
    Input:  (B, Tin, H, W, C_in)  + optional t_in (B, Tin) int64
    Output: (B, Tout, H, W, num_targets)
    """

    def __init__(
        self,
        in_channels,
        height,
        width,
        Tin           = 6,
        Tout          = 6,
        hid_S         = 32,
        hid_T         = 256,
        N_S           = 4,
        N_T           = 8,
        incep_ker     = (3, 5, 7, 11),
        groups        = 8,
        num_targets   = 1,
        time_embed_d  = 16,
        num_groups    = 8,
        **kwargs,
    ):
        super().__init__()

        self.Tin         = Tin
        self.Tout        = Tout
        self.num_targets = num_targets
        self.hid_S       = hid_S

        self.time_embed = AbsoluteTimeEmbedding(d=time_embed_d)
        d_time  = self.time_embed.embed_dim                 
        C_eff   = in_channels + 2 + d_time                 

        self.spatial_enc = SpatialEncoder(C_eff,  hid_S, N_S, num_groups)
        self.spatial_dec = SpatialDecoder(hid_S, C_eff,  N_S, num_groups)

        self.mid_xnet = Mid_Xnet(
            channel_in  = Tin  * hid_S,
            channel_hid = hid_T,
            N_T         = N_T,
            incep_ker   = incep_ker,
            groups      = groups,
        )
        # output head
        self.head = nn.Linear(C_eff, num_targets)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor,
                t_in: torch.Tensor = None) -> torch.Tensor:
        """
        x    : (B, Tin, H, W, C_in)
        t_in : (B, Tin) 
        out  : (B, Tout, H, W, num_targets)
        """
        B, Tin, H, W, _ = x.shape

        grid = get_grid(B, H, W, x.device)                  
        te   = self.time_embed(t_in) if t_in is not None else None


        frames = []
        for t in range(Tin):
            xt   = x[:, t]                                    
            te_t = te[:, t:t+1] if te is not None else None   
            if te_t is not None:
                te_t = te_t.unsqueeze(1).expand(B, H, W, -1)
                frame = torch.cat([xt, grid, te_t], dim=-1)   
            else:
                frame = torch.cat([xt, grid], dim=-1)        
            frames.append(frame)
        x_emb = torch.stack(frames, dim=1)                   

        x_flat = x_emb.view(B * Tin, H, W, x_emb.shape[-1])
        x_flat = x_flat.permute(0, 3, 1, 2)                 

        latent, enc1 = self.spatial_enc(x_flat)             
        _, hid_S, Hp, Wp = latent.shape                      

        z = latent.view(B, Tin * hid_S, Hp, Wp)
        z = self.mid_xnet(z)                                  
 
        z = z.view(B, Tin, hid_S, Hp, Wp)
        if self.Tout != Tin:
            z = F.interpolate(
                z.view(B, Tin, -1).permute(0, 2, 1),         
                size=self.Tout, mode='linear', align_corners=False,
            ).permute(0, 2, 1).view(B, self.Tout, hid_S, Hp, Wp)

        z_flat   = z.view(B * self.Tout, hid_S, Hp, Wp)
        enc1_mean = enc1.view(B, Tin, hid_S, H, W).mean(dim=1)  
        enc1_out  = enc1_mean.unsqueeze(1).expand(
            B, self.Tout, hid_S, H, W
        ).reshape(B * self.Tout, hid_S, H, W)

        out = self.spatial_dec(z_flat, enc1_out)               

        out = out.permute(0, 2, 3, 1)                         
        out = self.head(out)                                   
        out = out.view(B, self.Tout, H, W, self.num_targets)

        return out