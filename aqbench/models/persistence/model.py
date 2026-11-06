import torch
import torch.nn as nn
from aqbench.registry.model_registry import register_model


@register_model("persistence")
class PersistenceModel(nn.Module):
    """
    Persistence baseline:
    Predict target pollutants using the current observed snapshot.

    Output: (B, H, W, num_targets)
    """

    def __init__(
        self,
        in_channels,
        target_indices,
        num_targets=1,
        **kwargs,
    ):
        super().__init__()

        self.target_indices = target_indices
        self.num_targets = num_targets

    def forward(self, x):
        """
        x: (B, H, W, C)
        returns: (B, H, W, num_targets)
        """
        return x[..., self.target_indices]