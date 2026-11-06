"""
Shared channel-stacking utilities for all models.

Per-frame channel layout :
    [ C_in  |  2 (grid x,y)  |  d_time (abs time)  ]
    C_eff   = C_in + 2 + d_time   (d_time=16 for default AbsoluteTimeEmbedding)
    eff_in  = Tin × C_eff

get_grid   : normalised (B,H,W,2) linspace spatial coords, same for all frames.
             Grid is uniform 25×25 km² so no scale encoding needed.

stack_time_into_channels  : (B,Tin,H,W,C) → (B,H,W,Tin×C_eff)
unstack_channels_into_time: (B,H,W,Tout×C_tgt) → (B,Tout,H,W,C_tgt)
"""

import torch


def get_grid(B: int, H: int, W: int, device: torch.device) -> torch.Tensor:
    """
    Normalised spatial coordinate grid
    Returns
    -------
    (B, H, W, 2) float32
        [..., 0] : x ∈ [0,1] along H
        [..., 1] : y ∈ [0,1] along W
    """
    gridx = torch.linspace(0, 1, H, dtype=torch.float32, device=device)
    gridx = gridx.view(1, H, 1, 1).expand(B, H, W, 1)

    gridy = torch.linspace(0, 1, W, dtype=torch.float32, device=device)
    gridy = gridy.view(1, 1, W, 1).expand(B, H, W, 1)

    return torch.cat([gridx, gridy], dim=-1)   # (B, H, W, 2)


def stack_time_into_channels(
    x:          torch.Tensor,
    grid:       torch.Tensor = None,
    time_embed: torch.Tensor = None,
) -> torch.Tensor:
    """
    Flatten the time axis into the channel dimension.

    Parameters
    ----------
    x          : (B, Tin, H, W, C_in)
    grid       : (B, H, W, 2)       or None
    time_embed : (B, Tin, d_time)   or None

    Returns
    -------
    (B, H, W, Tin × C_eff)
    where C_eff = C_in + (2 if grid) + (d_time if time_embed)
    """
    B, Tin, H, W, C = x.shape
    parts = [x]

    if grid is not None:
        g = grid.unsqueeze(1).expand(B, Tin, H, W, 2)
        parts.append(g)

    if time_embed is not None:
        d = time_embed.shape[-1]
        t = time_embed.view(B, Tin, 1, 1, d).expand(B, Tin, H, W, d)
        parts.append(t)

    x_aug = torch.cat(parts, dim=-1)             

    return x_aug.permute(0, 2, 3, 1, 4).contiguous().reshape(B, H, W, -1)


def unstack_channels_into_time(
    y:           torch.Tensor,
    Tout:        int,
    num_targets: int,
) -> torch.Tensor:
    """
    Split the last channel axis back into (Tout, num_targets).

    Parameters
    ----------
    y           : (B, H, W, Tout × num_targets)
    Tout        : output time steps
    num_targets : channels per time step

    Returns
    -------
    (B, Tout, H, W, num_targets)
    """
    B, H, W, C_out = y.shape
    expected = Tout * num_targets
    if C_out != expected:
        raise ValueError(
            f"unstack: expected {expected}, got {C_out}")
    return y.view(B, H, W, Tout, num_targets).permute(0, 3, 1, 2, 4).contiguous()