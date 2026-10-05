"""RocketMan benchmark (Marisca et al., "Over-squashing in Spatiotemporal Graph Neural
Networks", NeurIPS 2025, https://github.com/marshka/spatiotemporal-oversquashing).

Every node receives an i.i.d. uniform signal over a window of length W. The target is the
mean of the signal at nodes exactly k hops from the target node, taken i steps before the
end of the window. k is the spatial distance, i the temporal distance. The loss is computed
on the target node only (via the mask).
"""
import numpy as np
import torch
from scipy.sparse.csgraph import floyd_warshall
from torch.utils.data import Dataset
from torch_sparse import SparseTensor
from tsl.data import StaticBatch
from tsl.ops.graph_generators import build_circle_graph
from tsl.utils.casting import parse_index


def build_lollipop_graph(n_nodes: int) -> np.ndarray:
    """A clique on the first n/2 nodes, joined by one edge to a path on the other n/2."""
    half = n_nodes // 2
    rows, cols = np.meshgrid(np.arange(half), np.arange(half), indexing="ij")
    clique = np.stack([rows.flatten(), cols.flatten()])
    clique = clique[:, clique[0] != clique[1]]
    path = np.stack([np.arange(half, n_nodes - 1), np.arange(half + 1, n_nodes)])
    return np.concatenate([clique, [[half - 1], [half]], path], axis=1)


class RocketMan(Dataset):
    """
    Args:
        n_nodes: number of nodes (even, >= 6).
        n_samples: number of sequences.
        window: input length W.
        spatial_order: spatial distance k of the source nodes from the target node.
        temporal_distance: temporal distance i of the source step from the window end.
        graph_generator: 'ring' or 'lollipop'.
        target_node: index of the target node.
        horizon: output length.
        seed: seed of the data generator (the paper uses seed = n_nodes).
    """

    n_channels = 1

    def __init__(self, n_nodes: int, n_samples: int, window: int, spatial_order: int,
                 temporal_distance: int, graph_generator: str = "ring",
                 target_node: int = 0, horizon: int = 1, seed: int = 16):
        super().__init__()
        self.n_nodes = n_nodes
        self.n_samples = n_samples
        self.window = window
        self.horizon = horizon
        self.spatial_order = spatial_order
        self.temporal_distance = temporal_distance
        self.target_node = target_node
        rng = torch.Generator().manual_seed(seed)

        adj = self.build_graph(graph_generator)
        row, col, _ = adj.coo()
        self.edge_index = torch.stack([col, row])

        self.x = torch.rand(n_samples, window, n_nodes, self.n_channels, generator=rng)
        self.y, self.mask = self.build_target(adj)

    def build_graph(self, graph_generator: str) -> SparseTensor:
        if graph_generator == "ring":
            _, edge_index, _ = build_circle_graph(self.n_nodes, undirected=True)
        elif graph_generator == "lollipop":
            edge_index = build_lollipop_graph(self.n_nodes)
        else:
            raise ValueError(f"Unknown graph_generator '{graph_generator}'.")
        edge_index = torch.tensor(edge_index[[1, 0]], dtype=torch.long)
        adj = SparseTensor.from_edge_index(edge_index)
        adj.set_value_(torch.ones(adj.nnz()))
        return adj.to_symmetric(reduce="max")

    def build_target(self, adj: SparseTensor) -> tuple[torch.Tensor, torch.Tensor]:
        dist = floyd_warshall(adj.to_scipy(layout="csr"), unweighted=True)
        adj_k = SparseTensor.from_edge_index(
            torch.tensor(np.stack(np.nonzero(dist == self.spatial_order))),
            sparse_sizes=adj.sizes())

        t = self.window - self.temporal_distance - 1
        y = self.x[:, t:t + self.horizon]
        if self.spatial_order > 0:
            y = adj_k.matmul(y, reduce="mean")

        target = torch.zeros_like(y)
        target[:, :, self.target_node] = y[:, :, self.target_node]
        mask = torch.zeros_like(y, dtype=torch.bool)
        mask[:, :, self.target_node] = True
        return target, mask

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, item):
        item = parse_index(item, length=self.n_samples, layout="index")
        batch = StaticBatch(input={"x": self.x[item]},
                            target={"y": self.y[item]},
                            mask=self.mask[item],
                            pattern={"x": "b t n f", "y": "b t n f", "edge_index": "n n"},
                            size=len(item))
        batch.input.update(edge_index=self.edge_index)
        return batch
