"""
models.py
Model architectures for the comparison (Chapter 4, Section D).
The backbone is the edge-aware Graph Attention Network from the Multi-GNN
reference implementation (Egressy et al., 2024; GAT of Velickovic et al., 2018),
reproduced here with two deliberate additions required by this dissertation:

1. ``return_embeddings`` -- the forward pass can optionally return the per-layer
   node embeddings, which the over-smoothing diagnostics (Dirichlet energy, MAD)
   consume. The reference implementation does not expose these.

2. A ``MLPBaseline`` -- a graph-agnostic classifier over edge features only. This
   is the supervisor's requested "Logistic Regression / XGBoost on transaction
   features" baseline (point 5), included to test whether graph structure adds
   value at all. Implemented as an MLP so it shares the training/evaluation loop
   with the GNNs and uses the identical features and splits.

The GATe architecture is kept faithful to the reference so results remain
comparable with published Multi-GNN numbers: node/edge embedding layers, a stack
of ``GATConv`` layers with edge features, a residual "(x + relu(bn(conv)))/2"
update, optional edge-update MLPs, and a 3-part edge readout
(source embedding || target embedding || edge embedding) into an MLP head.

The Multi-GAT enhancements (ports, ego IDs, reverse message passing) are applied
at the *data* level (see data module), exactly as in the reference repo, so the
same model class serves both the GAT and Multi-GAT arms.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import BatchNorm, GATConv, Linear

logger = logging.getLogger(__name__)


class GATe(torch.nn.Module):
    """Edge-aware GAT backbone (faithful to Multi-GNN, with embedding hooks)."""

    def __init__(
        self,
        num_features: int,
        num_gnn_layers: int,
        n_classes: int = 2,
        n_hidden: int = 64,
        n_heads: int = 4,
        edge_updates: bool = False,
        edge_dim: int | None = None,
        dropout: float = 0.0,
        final_dropout: float = 0.5,
    ):
        super().__init__()
        # GAT concatenates heads, so hidden size must be divisible by n_heads.
        tmp_out = n_hidden // n_heads
        n_hidden = tmp_out * n_heads

        self.n_hidden = n_hidden
        self.n_heads = n_heads
        self.num_gnn_layers = num_gnn_layers
        self.edge_updates = edge_updates
        self.final_dropout = final_dropout

        self.node_emb = nn.Linear(num_features, n_hidden)
        self.edge_emb = nn.Linear(edge_dim, n_hidden)

        self.convs = nn.ModuleList()
        self.emlps = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        for _ in range(self.num_gnn_layers):
            self.convs.append(
                GATConv(self.n_hidden, tmp_out, self.n_heads, concat=True,
                        dropout=dropout, add_self_loops=True, edge_dim=self.n_hidden)
            )
            if self.edge_updates:
                self.emlps.append(nn.Sequential(
                    nn.Linear(3 * self.n_hidden, self.n_hidden), nn.ReLU(),
                    nn.Linear(self.n_hidden, self.n_hidden),
                ))
            self.batch_norms.append(BatchNorm(n_hidden))

        # Edge readout head: [x_src || x_dst || edge_attr] -> class logits.
        self.mlp = nn.Sequential(
            Linear(n_hidden * 3, 50), nn.ReLU(), nn.Dropout(self.final_dropout),
            Linear(50, 25), nn.ReLU(), nn.Dropout(self.final_dropout),
            Linear(25, n_classes),
        )

    def forward(self, x, edge_index, edge_attr, n_classify: int | None = None,
                sup_edge_index=None, sup_edge_attr=None,
                return_embeddings: bool = False):
        """Message-pass over ``edge_index``; classify ``sup_edge_index``.

        The two edge sets are deliberately distinct (Chapter 3, Sections 3.3.2
        and 3.7):

          E_message    -- edges that carry messages, i.e. that shape the node
                          representations. DropEdge samples from THIS set.
          E_supervision-- edges that receive a label and enter the loss. This set
                          is NOT touched by DropEdge, so contraction of the
                          topology is never confounded with random undersampling
                          of the supervised examples.

        When ``sup_edge_index`` is None the two sets coincide (the no-intervention
        case), and behaviour is identical to the original implementation.

        A supervision edge may be absent from the message-passing graph, so its
        edge state cannot come from the convolutions. Instead it is embedded by
        the same ``edge_emb`` and, when edge updates are enabled, refreshed by the
        same per-layer MLPs using the current node states -- so it follows the
        same transformation path as a message-passing edge without contributing
        messages itself.
        """
        src, dst = edge_index

        x = self.node_emb(x)
        edge_attr = self.edge_emb(edge_attr)

        # Supervision edges: embedded in parallel, never used for message passing.
        separate_sup = sup_edge_index is not None
        if separate_sup:
            s_src, s_dst = sup_edge_index
            sup_state = self.edge_emb(sup_edge_attr)

        layer_embeddings = [x] if return_embeddings else None

        for i in range(self.num_gnn_layers):
            x = (x + F.relu(self.batch_norms[i](self.convs[i](x, edge_index, edge_attr)))) / 2
            if self.edge_updates:
                edge_attr = edge_attr + self.emlps[i](
                    torch.cat([x[src], x[dst], edge_attr], dim=-1)) / 2
                if separate_sup:
                    sup_state = sup_state + self.emlps[i](
                        torch.cat([x[s_src], x[s_dst], sup_state], dim=-1)) / 2
            if return_embeddings:
                layer_embeddings.append(x)

        # classification head over the supervision edges 
        if separate_sup:
            ce, cea = sup_edge_index, sup_state
        else:
            m = edge_index.shape[1] if n_classify is None else n_classify
            ce, cea = edge_index[:, :m], edge_attr[:m]

        e = x[ce.T].reshape(-1, 2 * self.n_hidden).relu()
        e = torch.cat((e, cea.view(-1, cea.shape[1])), 1)
        logits = self.mlp(e)

        if return_embeddings:
            return logits, layer_embeddings
        return logits


class MLPBaseline(torch.nn.Module):
    """Graph-agnostic edge classifier over edge features only.

    Supervisor point 5: a non-graph baseline is needed to show whether graph
    learning adds value. This MLP sees only the edge feature vector (no message
    passing), so if the GNNs cannot beat it, the graph structure is not helping.

    It ignores ``x`` and ``edge_index`` in ``forward`` so that it is a drop-in
    replacement inside the same training loop.
    """

    def __init__(self, edge_dim: int, n_classes: int = 2, n_hidden: int = 64,
                 final_dropout: float = 0.5, **_ignored):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(edge_dim, n_hidden), nn.ReLU(), nn.Dropout(final_dropout),
            nn.Linear(n_hidden, n_hidden // 2), nn.ReLU(), nn.Dropout(final_dropout),
            nn.Linear(n_hidden // 2, n_classes),
        )

    def forward(self, x, edge_index, edge_attr, n_classify: int | None = None,
                return_embeddings: bool = False):
        # The MLP never sees reverse edges (it is only ever the E0 baseline), but
        # it accepts n_classify for a uniform interface with GATe.
        attr = edge_attr if n_classify is None else edge_attr[:n_classify]
        logits = self.net(attr)
        if return_embeddings:
            return logits, [attr]
        return logits


def build_model(model_name: str, num_features: int, edge_dim: int, config: dict):
    """Factory that builds a model from a config dict (used by the runner)."""
    if model_name == "mlp":
        return MLPBaseline(
            edge_dim=edge_dim, n_hidden=config.get("n_hidden", 64),
            final_dropout=config.get("final_dropout", 0.5),
        )
    if model_name in ("gat", "multi_gat"):
        return GATe(
            num_features=num_features, num_gnn_layers=config["n_gnn_layers"],
            n_classes=2, n_hidden=config.get("n_hidden", 64),
            n_heads=config.get("n_heads", 4), edge_updates=config.get("edge_updates", False),
            edge_dim=edge_dim, dropout=config.get("dropout", 0.0),
            final_dropout=config.get("final_dropout", 0.5),
        )
    raise ValueError(f"Unknown model '{model_name}'")
