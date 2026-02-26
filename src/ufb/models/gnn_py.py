# src/ufb/models/gnn.py
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
    """Simple, stable GraphSAGE stack with residual + LayerNorm."""
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
            h = norm(h + h_in)  # residual
        return h


class Model1Net(nn.Module):
    """
    Two-head model for Model_1:
      - 2D: GraphSAGE on 2D graph
      - 1D: MLP (uses your engineered coupling features like conn2d_wl_t)
    Outputs DELTAS: delta_wl (t+1 - t) for both 2D and 1D.
    """
    def __init__(
        self,
        in_dim_2d: int,
        in_dim_1d: int,
        hidden_dim: int = 128,
        gnn_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.gnn2d = SAGEStack(in_dim=in_dim_2d, hidden_dim=hidden_dim, num_layers=gnn_layers, dropout=dropout)
        self.head2d = MLP(in_dim=hidden_dim, hidden_dim=hidden_dim, out_dim=1, dropout=dropout)

        self.mlp1d = MLP(in_dim=in_dim_1d, hidden_dim=hidden_dim, out_dim=1, dropout=dropout)

    def forward(
        self,
        x2d: torch.Tensor,
        edge_index_2d: torch.Tensor,
        x1d: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # 2D deltas
        h2 = self.gnn2d(x2d, edge_index_2d)
        d2 = self.head2d(h2).squeeze(-1)

        # 1D deltas
        d1 = self.mlp1d(x1d).squeeze(-1)

        return d2, d1