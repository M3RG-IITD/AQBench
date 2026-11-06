import torch
from torch import nn


def regular_grid_2d(spatial_dims, grid_boundaries=[[0, 1], [0, 1]]):

    height, width = spatial_dims

    xt = torch.linspace(grid_boundaries[0][0], grid_boundaries[0][1], height)
    yt = torch.linspace(grid_boundaries[1][0], grid_boundaries[1][1], width)

    grid_x, grid_y = torch.meshgrid(xt, yt, indexing="ij")

    return grid_x, grid_y


class GridEmbedding2D(nn.Module):

    def __init__(self, in_channels, grid_boundaries=[[0,1],[0,1]]):
        super().__init__()
        self.in_channels = in_channels
        self.grid_boundaries = grid_boundaries
        self._grid = None
        self._res = None

    @property
    def out_channels(self):
        return self.in_channels + 2

    def grid(self, spatial_dims, device, dtype):

        if self._grid is None or self._res != spatial_dims:

            grid_x, grid_y = regular_grid_2d(
                spatial_dims,
                grid_boundaries=self.grid_boundaries
            )

            grid_x = grid_x.to(device).to(dtype).unsqueeze(0).unsqueeze(0)
            grid_y = grid_y.to(device).to(dtype).unsqueeze(0).unsqueeze(0)

            self._grid = grid_x, grid_y
            self._res = spatial_dims

        return self._grid

    def forward(self, x):

        batch = x.shape[0]

        gx, gy = self.grid(x.shape[-2:], x.device, x.dtype)

        x = torch.cat(
            (
                x,
                gx.expand(batch,-1,-1,-1),
                gy.expand(batch,-1,-1,-1)
            ),
            dim=1
        )

        return x