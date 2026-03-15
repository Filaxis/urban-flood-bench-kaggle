# src/ufb/models/gnn_py.py
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import SAGEConv
except ImportError as e:
    raise ImportError(
        "torch_geometric is required. On Kaggle: "
        "pip install torch-geometric -f https://data.pyg.org/whl/torch-<TORCH>+cu<CUDA>.html"
    ) from e


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 128, out_dim: int = 1, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SAGEStack(nn.Module):
    """GraphSAGE stack with residual + LayerNorm."""
    def __init__(self, in_dim: int, hidden_dim: int = 128, num_layers: int = 3, dropout: float = 0.1):
        super().__init__()
        assert num_layers >= 1
        self.in_proj = nn.Linear(in_dim, hidden_dim)
        self.convs = nn.ModuleList([SAGEConv(hidden_dim, hidden_dim) for _ in range(num_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = self.in_proj(x)
        for conv, norm in zip(self.convs, self.norms):
            h_in = h
            h = conv(h, edge_index)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            h = norm(h + h_in)
        return h


class Model1Net(nn.Module):
    """
    Four-output model for Model_1 and Model_2.

    Inputs
    ------
    x2d          : (N2, in_dim_2d)  — 2D node features
    edge_index_2d: (2, E2)          — 2D graph connectivity
    x1d          : (N1, in_dim_1d)  — 1D node features (includes inlet_flow lags)
    edge_index_1d: (2, E1)          — 1D pipe graph connectivity
    x_edge       : (E1, in_dim_edge)— 1D edge features (flow lags + static)

    Outputs (all raw, unnormalised)
    -------
    d2      : (N2,)  delta_wl for 2D nodes
    d1      : (N1,)  delta_wl for 1D nodes
    inlet   : (N1,)  predicted inlet_flow at t+1  (absolute, not delta)
    eflow   : (E1,)  predicted edge_flow at t+1   (absolute, not delta)
    """

    def __init__(
        self,
        in_dim_2d: int,
        in_dim_1d: int,
        in_dim_edge: int,
        hidden_dim: int = 128,
        gnn_layers_2d: int = 3,
        gnn_layers_1d: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        # 2D subsystem — GraphSAGE (unchanged)
        self.gnn2d  = SAGEStack(in_dim=in_dim_2d, hidden_dim=hidden_dim,
                                num_layers=gnn_layers_2d, dropout=dropout)
        self.head2d = MLP(in_dim=hidden_dim, hidden_dim=hidden_dim, out_dim=1, dropout=dropout)

        # 1D subsystem — now also GraphSAGE (was MLP)
        self.gnn1d  = SAGEStack(in_dim=in_dim_1d, hidden_dim=hidden_dim,
                                num_layers=gnn_layers_1d, dropout=dropout)
        self.head1d       = MLP(in_dim=hidden_dim, hidden_dim=hidden_dim, out_dim=1, dropout=dropout)
        self.head_inlet   = MLP(in_dim=hidden_dim, hidden_dim=hidden_dim, out_dim=1, dropout=dropout)

        # Edge flow head — takes concatenated (src_emb, dst_emb, edge_feats)
        # src and dst embeddings come from the 1D node embeddings after GNN
        self.edge_proj = nn.Linear(in_dim_edge, hidden_dim)
        self.head_edge = MLP(in_dim=hidden_dim * 3, hidden_dim=hidden_dim, out_dim=1, dropout=dropout)

    def forward(
        self,
        x2d: torch.Tensor,           # (N2, in_dim_2d)
        edge_index_2d: torch.Tensor, # (2, E2)
        x1d: torch.Tensor,           # (N1, in_dim_1d)
        edge_index_1d: torch.Tensor, # (2, E1)
        x_edge: torch.Tensor,        # (E1, in_dim_edge)
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        # ---- 2D ----
        h2  = self.gnn2d(x2d, edge_index_2d)   # (N2, hidden)
        d2  = self.head2d(h2).squeeze(-1)        # (N2,)

        # ---- 1D nodes ----
        h1      = self.gnn1d(x1d, edge_index_1d)     # (N1, hidden)
        d1      = self.head1d(h1).squeeze(-1)          # (N1,)
        inlet   = self.head_inlet(h1).squeeze(-1)      # (N1,)

        # ---- 1D edges ----
        src, dst = edge_index_1d[0], edge_index_1d[1]  # (E1,) each
        h_src  = h1[src]                               # (E1, hidden)
        h_dst  = h1[dst]                               # (E1, hidden)
        h_e    = F.relu(self.edge_proj(x_edge))        # (E1, hidden)
        h_edge = torch.cat([h_src, h_dst, h_e], dim=-1)  # (E1, 3*hidden)
        eflow  = self.head_edge(h_edge).squeeze(-1)    # (E1,)

        return d2, d1, inlet, eflow
