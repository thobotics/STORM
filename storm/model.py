"""STORM: Spatio-Temporal Oscillatory dynamics via graph Resistance Model.

Node states H and velocities V follow one explicit Euler step of size eps (Eq. 3 of the paper):

    V <- V - eps * ( L_hat H W + D(H) * V - F(H, X_t) )
    H <- H + eps * V

    L_hat H W   graph Laplacian with learned, non-negative edge resistances (ResistiveLaplacian)
    D(H)        dissipation, ReLU(H W_D)
    F(H, X_t)   forcing: a GCN on [H, X_t] (GCNForcing) plus an MLP on [H, X_t] (input_net)

The last block runs window + horizon steps; X_t is only available for the first `window`
steps. The last `horizon` states are decoded by an MLP readout.
"""
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from tsl.nn.layers.base import NodeEmbedding
from tsl.nn.models import BaseModel
from tsl.nn.utils import get_layer_activation, maybe_cat_exog


def mlp(in_size: int, hidden_size: int, out_size: int, activation=nn.GELU) -> nn.Sequential:
    return nn.Sequential(nn.Linear(in_size, hidden_size), activation(),
                         nn.Linear(hidden_size, out_size))


class ResistiveLaplacian(MessagePassing):
    """Laplacian term L_hat H W: sum_j a_ij * (W h_i - W h_j), with edge resistances a_ij."""

    def __init__(self, hidden_size: int):
        super().__init__(aggr="add")
        self.lin = nn.Linear(hidden_size, hidden_size)

    def forward(self, x: Tensor, edge_index: Tensor, edge_resistance: Tensor) -> Tensor:
        return self.propagate(edge_index, x=self.lin(x), edge_resistance=edge_resistance)

    def message(self, x_i: Tensor, x_j: Tensor, edge_resistance: Tensor) -> Tensor:
        return edge_resistance.view(x_j.size(0), -1, 1) * (x_i - x_j)


class GCNForcing(MessagePassing):
    """GCN part of the forcing: GCN-normalised aggregation and a non-linearity."""

    def __init__(self, hidden_size: int, output_style: str = "linear"):
        super().__init__(aggr="add")
        self.lin = nn.Linear(hidden_size, hidden_size, bias=False)
        if output_style == "linear":
            self.output_net = nn.Sequential(nn.ReLU(), nn.Linear(hidden_size, hidden_size))
        elif output_style == "tanh":
            self.output_net = F.tanh
        elif output_style == "relu":
            self.output_net = F.relu
        else:
            raise ValueError(f"Unknown forcing_style '{output_style}'.")

    def forward(self, x: Tensor, edge_index: Tensor, edge_weight: Tensor) -> Tensor:
        return self.output_net(self.propagate(edge_index, x=self.lin(x),
                                              edge_weight=edge_weight))

    def message(self, x_j: Tensor, edge_weight: Tensor) -> Tensor:
        return edge_weight.view(1, -1, 1) * x_j


class STORMBlock(MessagePassing):
    """Integrates the STORM ODE for a given number of steps."""

    def __init__(self,
                 hidden_size: int,
                 step_size: float,
                 fix_resistance: bool = True,
                 use_dissipation: bool = True,
                 use_forcing: bool = True,
                 forcing_style: str = "linear",
                 input_style: str = "combine"):
        super().__init__(aggr="add")
        if input_style not in ("combine", "none"):
            raise ValueError(f"Unknown input_style '{input_style}'.")
        self.step_size = step_size
        self.fix_resistance = fix_resistance
        self.input_style = input_style

        self.bnorm = nn.BatchNorm1d(hidden_size)  # unused, kept for checkpoint compatibility
        self.conv = ResistiveLaplacian(hidden_size)
        self.dissipative_net = (nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.ReLU())
                                if use_dissipation else None)
        self.external_forcing_net = (GCNForcing(hidden_size, forcing_style)
                                     if use_forcing else None)
        self.edge_resistance_net = nn.Sequential(nn.Linear(2 * hidden_size, hidden_size),
                                                 nn.ReLU(),
                                                 nn.Linear(hidden_size, 1),
                                                 nn.Tanh())
        self.velocity_net = nn.Sequential(nn.Linear(hidden_size, hidden_size))
        self.forcing_input_net = mlp(2 * hidden_size, hidden_size, hidden_size)
        self.input_net = (mlp(2 * hidden_size, hidden_size, hidden_size)
                          if input_style == "combine" else None)

    def edge_resistance(self, x: Tensor, edge_index: Tensor) -> Tensor:
        pairs = torch.cat([x[:, edge_index[0]], x[:, edge_index[1]]], dim=-1)
        return self.edge_resistance_net(pairs).squeeze(-1).abs()

    def forward(self,
                x: Tensor,
                edge_index: Tensor,
                gcn_edges: tuple[Tensor, Tensor],
                num_steps: int,
                inputs: Tensor) -> Tensor:
        """
        Args:
            x: initial node states, [batch, nodes, hidden].
            edge_index: graph connectivity, [2, edges].
            gcn_edges: (edge_index, edge_weight) with self-loops and GCN normalisation.
            num_steps: number of Euler steps.
            inputs: encoded inputs, [batch, steps, nodes, hidden]; step t reads inputs[:, t]
                while t < inputs.size(1).

        Returns:
            Node states at steps 0..num_steps, [batch, num_steps + 1, nodes, hidden].
        """
        v = self.velocity_net(x)
        resistance = self.edge_resistance(x, edge_index)
        states = [x]
        for t in range(num_steps):
            u = inputs[:, t] if t < inputs.size(1) else None
            if not self.fix_resistance:
                resistance = self.edge_resistance(x, edge_index)

            laplacian = self.conv(x, edge_index, resistance)
            dissipation = self.dissipative_net(x) if self.dissipative_net is not None else 0.

            forcing = 0.
            if self.external_forcing_net is not None:
                x_mess = x if u is None else self.forcing_input_net(torch.cat([x, u], dim=-1))
                forcing = self.external_forcing_net(x_mess, *gcn_edges)
            if u is not None and self.input_net is not None:
                forcing = forcing + self.input_net(torch.cat([x, u], dim=-1))

            v = v - self.step_size * (laplacian + dissipation * v - forcing)
            x = x + self.step_size * v
            states.append(x)
        return torch.stack(states, dim=1)


class STORM(BaseModel):
    """STORM forecasting model.

    Args:
        input_size: input channels per node.
        n_nodes: number of nodes.
        window: number of input time steps.
        horizon: number of predicted time steps.
        hidden_size: node state size.
        step_size: Euler step size (eps).
        num_blocks: number of stacked STORM blocks.
        num_iters: Euler steps of the last block (default: window + horizon).
        output_size: output channels per node (default: input_size).
        exog_size: exogenous channels concatenated to the input.
        activation: activation of the input encoders and readout.
        fix_resistance: compute edge resistances once per block (True) or at every step.
        use_dissipation: use the dissipation term D(x).
        use_forcing: use the GCN forcing term.
        forcing_style: output non-linearity of the GCN forcing ('linear', 'tanh', 'relu').
        input_style: 'combine' adds a direct input term to the forcing, 'none' does not.
    """

    def __init__(self,
                 input_size: int,
                 n_nodes: int,
                 window: int,
                 horizon: int,
                 hidden_size: int = 128,
                 step_size: float = 0.1,
                 num_blocks: int = 1,
                 num_iters: Optional[int] = None,
                 output_size: Optional[int] = None,
                 exog_size: int = 0,
                 activation: str = "tanh",
                 fix_resistance: bool = True,
                 use_dissipation: bool = True,
                 use_forcing: bool = True,
                 forcing_style: str = "linear",
                 input_style: str = "combine"):
        super().__init__()
        self.window = window
        self.horizon = horizon
        self.num_blocks = num_blocks
        self.num_iters = window + horizon if num_iters is None else num_iters
        if self.num_iters < horizon:
            raise ValueError("num_iters must be at least horizon.")
        act = get_layer_activation(activation)
        in_size = input_size + exog_size

        # Do not reorder: the creation order fixes the initial weights of a given seed and
        # the checkpoint layout of the paper's models.
        self.emb = NodeEmbedding(n_nodes=n_nodes, emb_size=hidden_size)
        self.input_encoder = mlp(in_size, hidden_size, hidden_size, act)
        self.init_encoder = mlp(in_size, hidden_size, hidden_size, act)
        # unused, kept for checkpoint compatibility
        self.norm_layers = nn.ModuleList([nn.BatchNorm1d(hidden_size) for _ in range(num_blocks)])
        self.conv_layers = nn.ModuleList([
            STORMBlock(hidden_size, step_size,
                       fix_resistance=fix_resistance,
                       use_dissipation=use_dissipation,
                       use_forcing=use_forcing,
                       forcing_style=forcing_style,
                       input_style=input_style)
            for _ in range(num_blocks)])
        self.mlp_layers = nn.ModuleList([mlp(hidden_size, hidden_size, hidden_size)
                                         for _ in range(num_blocks)])
        self.readout = mlp(hidden_size, hidden_size, output_size or input_size, act)

    def forward(self, x: Tensor, edge_index: Tensor, u: Optional[Tensor] = None) -> Tensor:
        """
        Args:
            x: input window, [batch, window, nodes, input_size].
            edge_index: graph connectivity, [2, edges].
            u: optional exogenous variables, [batch, window, (nodes,) exog_size].

        Returns:
            Forecast, [batch, horizon, nodes, output_size].
        """
        x = maybe_cat_exog(x, u)
        gcn_edges = gcn_norm(edge_index=edge_index, edge_weight=None)

        h = self.init_encoder(x[:, :self.window]) + self.emb()
        inputs = self.input_encoder(x)[:, :self.window]
        for i, block in enumerate(self.conv_layers):
            last = i == self.num_blocks - 1
            h = block(h[:, 0], edge_index, gcn_edges,
                      num_steps=self.num_iters if last else self.window,
                      inputs=inputs)
            if not last:
                h = self.mlp_layers[i](h)
                inputs = h[:, :self.window]
        return self.readout(h[:, -self.horizon:])
