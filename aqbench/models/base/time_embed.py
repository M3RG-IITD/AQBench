import math
import numpy as np
import torch
import torch.nn as nn

__all__ = ["FourierExpansion", "AbsoluteTimeEmbedding"]

_ABS_TIME_LOWER: float = 1.0
_ABS_TIME_UPPER: float = 24.0 * 365.25   


class FourierExpansion(nn.Module):

    def __init__(self, lower: float, upper: float,
                 assert_range: bool = True) -> None:
        super().__init__()
        self.lower        = lower
        self.upper        = upper
        self.assert_range = assert_range

    def forward(self, x: torch.Tensor, d: int) -> torch.Tensor:
        """
        x : (...) float  — values to encode
        d : int          — output dims, must be even
        returns (..., d) float32
        """
        if d % 2 != 0:
            raise ValueError(f"d must be even, got {d}")

        if self.assert_range:
            in_range = torch.logical_and(
                self.lower <= x.abs(), x.abs() <= self.upper)
            if not torch.all(torch.logical_or(in_range, x == 0)):
                raise AssertionError(
                    f"Input not within [{self.lower}, {self.upper}]")

        x = x.double()   

        wavelengths = torch.logspace(
            math.log10(self.lower),
            math.log10(self.upper),
            d // 2,
            base=10,
            device=x.device,
            dtype=x.dtype,
        )
        prod     = torch.einsum("...i,j->...ij", x, 2 * np.pi / wavelengths)
        encoding = torch.cat([torch.sin(prod), torch.cos(prod)], dim=-1)
        return encoding.float()

    def extra_repr(self) -> str:
        return (f"lower={self.lower}, upper={self.upper}, "
                f"assert_range={self.assert_range}")


class AbsoluteTimeEmbedding(nn.Module):
    """
    Aurora-style absolute time embedding.

    Parameters
    ----------
    d : int   output dims per timestep, must be even (default 16)

    forward(t_in)
        t_in : (B, Tin) int64  hours since 2000-01-01 00:00 UTC
        returns (B, Tin, d)    float32
    """

    def __init__(self, d: int = 16) -> None:
        super().__init__()
        if d % 2 != 0:
            raise ValueError(f"d must be even, got {d}")
        self.embed_dim  = d
        self._expansion = FourierExpansion(
            lower=_ABS_TIME_LOWER,
            upper=_ABS_TIME_UPPER,
            assert_range=False,
        )

    def forward(self, t_in: torch.Tensor) -> torch.Tensor:
        return self._expansion(t_in.float(), self.embed_dim)

    def extra_repr(self) -> str:
        return (f"d={self.embed_dim}, "
                f"lower={_ABS_TIME_LOWER}h, upper={_ABS_TIME_UPPER:.0f}h")