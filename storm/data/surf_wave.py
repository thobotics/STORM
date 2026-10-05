"""SURF-WAVE: a 2D wave on an L-shaped water surface, driven by three moving balls.

Nodes are the water cells of the grid plus one node per ball; each ball is connected to the
six corners of the L-shaped grid. Channels per node:
    0 wave height   1 x   2 y   3 d wave height   4 d x   5 d y   (d = one-step difference)
    6 is_normal   7 is_ball   8 is_boundary
Grid nodes are scored on channels {0, 3}, ball nodes on {1, 2, 4, 5}. Channels 0-5 are
min-max scaled with statistics of the training split; the node-type channels are not.

The HDF5 file is produced by scripts/generate_surf_wave.py.
"""
import h5py
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset
from torch_sparse import SparseTensor
from tsl.data import StaticBatch
from tsl.data.preprocessing import scalers
from tsl.metrics import torch_metrics
from tsl.utils.casting import parse_index

from .datamodule import SequentialSplitDataModule

GRID_CHANNELS = [0, 3]
BALL_CHANNELS = [1, 2, 4, 5]
N_PHYSICAL = 6
N_BALLS = 3


class PhysicalChannelScaler:
    """Applies a scaler to the physical channels only."""

    def __init__(self, scaler):
        self.scaler = scaler

    def _apply(self, fn, x: Tensor) -> Tensor:
        phys = fn(x[..., :N_PHYSICAL].cpu()).to(device=x.device, dtype=x.dtype)
        return torch.cat([phys, x[..., N_PHYSICAL:]], dim=-1)

    def transform(self, x: Tensor) -> Tensor:
        return self._apply(self.scaler.transform, x)

    def inverse_transform(self, x: Tensor) -> Tensor:
        return self._apply(self.scaler.inverse_transform, x)


class SurfWave(Dataset):

    n_channels = 9

    def __init__(self, path: str, window: int = 25, horizon: int = 25):
        super().__init__()
        with h5py.File(path, "r") as f:
            wave = torch.tensor(f["wave"][:], dtype=torch.float32)    # [S, T, G, G]
            balls = torch.tensor(f["balls"][:], dtype=torch.float32)  # [S, T, 3, 2]
            self.obstacle = f["mask"][:]                               # [G, G], 1 = wall
            self.grid_size = int(f.attrs["grid_size"])
        if window + horizon != wave.size(1):
            raise ValueError("window + horizon must equal the trajectory length.")
        self.window, self.horizon = window, horizon

        self.cells = [tuple(p) for p in np.argwhere(self.obstacle == 0)]
        self.cell_index = {p: i for i, p in enumerate(self.cells)}
        self.n_grid = len(self.cells)
        self.n_nodes = self.n_grid + N_BALLS
        self.n_samples = wave.size(0)

        self.edge_index = self.build_graph()
        self.node_types = self.build_node_types()
        features = self.build_features(wave, balls)                   # [S, T, N, 6]
        self.x = features[:, :window]
        self.y = features[:, window:]
        self.mask = torch.zeros(self.n_nodes, self.n_channels, dtype=torch.bool)
        self.mask[:self.n_grid, GRID_CHANNELS] = True
        self.mask[self.n_grid:, BALL_CHANNELS] = True
        self.transform = {}

    def is_free(self, i: int, j: int) -> bool:
        return 0 <= i < self.grid_size and 0 <= j < self.grid_size and self.obstacle[i, j] == 0

    def build_graph(self) -> Tensor:
        g = self.grid_size
        edges = []
        for node, (i, j) in enumerate(self.cells):
            for di, dj in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                if self.is_free(i + di, j + dj):
                    edges.append([node, self.cell_index[(i + di, j + dj)]])
        corners = [(0, 0), (0, g - 1), (g - 1, 0), (g - 1, (g - 1) // 2),
                   ((g - 1) // 2, (g - 1) // 2), ((g - 1) // 2, g - 1)]
        corners = [self.cell_index[c] for c in corners if self.is_free(*c)]
        for b in range(N_BALLS):
            for c in corners:
                edges += [[c, self.n_grid + b], [self.n_grid + b, c]]
        adj = SparseTensor.from_edge_index(torch.tensor(edges, dtype=torch.long).t(),
                                           sparse_sizes=(self.n_nodes, self.n_nodes))
        row, col, _ = adj.coo()
        return torch.stack([col, row])

    def build_node_types(self) -> Tensor:
        """One-hot [is_normal, is_ball, is_boundary].

        Grid node n gets is_boundary when the cell with row-major index n (i * grid_size + j)
        touches a wall or the domain edge. The paper's models were trained with this encoding.
        """
        g = self.grid_size
        boundary = set()
        for i, j in self.cells:
            on_border = i in (0, g - 1) or j in (0, g - 1)
            near_wall = any(0 <= i + di < g and 0 <= j + dj < g
                            and self.obstacle[i + di, j + dj] == 1
                            for di, dj in [(0, 1), (1, 0), (0, -1), (-1, 0)])
            if on_border or near_wall:
                boundary.add(i * g + j)
        types = torch.zeros(self.n_nodes, 3)
        for node in range(self.n_grid):
            types[node, 2 if node in boundary else 0] = 1.
        types[self.n_grid:, 1] = 1.
        return types

    def build_features(self, wave: Tensor, balls: Tensor) -> Tensor:
        """Positions and one-step differences; the first step has zero difference."""
        rows, cols = zip(*self.cells)
        f = torch.zeros(self.n_samples, wave.size(1), self.n_nodes, N_PHYSICAL)
        f[:, :, :self.n_grid, 0] = wave[:, :, list(rows), list(cols)]
        f[:, :, :self.n_grid, 1] = torch.tensor(cols, dtype=torch.float32)
        f[:, :, :self.n_grid, 2] = torch.tensor(rows, dtype=torch.float32)
        f[:, :, self.n_grid:, 1:3] = balls
        f[:, 1:, :self.n_grid, 3] = f[:, 1:, :self.n_grid, 0] - f[:, :-1, :self.n_grid, 0]
        f[:, 1:, self.n_grid:, 4:6] = f[:, 1:, self.n_grid:, 1:3] - f[:, :-1, self.n_grid:, 1:3]
        return f

    def fit_scaler(self, train_idx: list[int]):
        scaler = scalers.MinMaxScaler(axis=(0, 1, 2))
        scaler.fit(torch.cat([self.x[train_idx], self.y[train_idx]], dim=1))
        self.transform = {"x": PhysicalChannelScaler(scaler),
                          "y": PhysicalChannelScaler(scaler)}

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, item):
        item = parse_index(item, length=self.n_samples, layout="index")
        b = len(item)
        x = torch.cat([self.x[item], self.node_types.expand(b, self.window, -1, -1)], -1)
        y = torch.cat([self.y[item], self.node_types.expand(b, self.horizon, -1, -1)], -1)
        if "x" in self.transform:
            x = self.transform["x"].transform(x)
        batch = StaticBatch(input={"x": x},
                            target={"y": y},
                            mask=self.mask.expand(b, self.horizon, -1, -1),
                            transform=self.transform,
                            pattern={"x": "b t n f", "y": "b t n f", "edge_index": "n n"},
                            size=b)
        batch.input.update(edge_index=self.edge_index)
        return batch


class SurfWaveDataModule(SequentialSplitDataModule):
    """Sequential split; the scaler is fit on the training trajectories (input + target)."""

    def setup(self, stage=None):
        super().setup(stage)
        self.dataset.fit_scaler(self.train_set.indices)


class WeightedMaskedMSE(torch_metrics.MaskedMSE):
    """grid_weight * MSE(grid channels) + ball_weight * MSE(ball channels)."""

    def __init__(self, grid_weight: float = 1.0, ball_weight: float = 10.0, **kwargs):
        super().__init__(**kwargs)
        self.grid_weight = grid_weight
        self.ball_weight = ball_weight

    def forward(self, y_hat: Tensor, y: Tensor, mask: Tensor) -> Tensor:
        grid = super().forward(y_hat[..., GRID_CHANNELS], y[..., GRID_CHANNELS],
                               mask[..., GRID_CHANNELS])
        ball = super().forward(y_hat[..., BALL_CHANNELS], y[..., BALL_CHANNELS],
                               mask[..., BALL_CHANNELS])
        return self.grid_weight * grid + self.ball_weight * ball
