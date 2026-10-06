"""
multigat.py
===========

The Multi-GAT structural enhancements of Egressy et al. (2024), applied at the
DATA level so the same GATe model class serves both the GAT and Multi-GAT arms
(keeping the comparison controlled).

Egressy et al. propose three enhancements. Two of them translate to full-batch
training and are implemented here; the third does not, and is documented
rather than fake it:
  1. REVERSE MESSAGE PASSING  (implemented)
     For every directed transaction (u -> v) a reverse edge (v -> u) is added for
     message passing only. Without it, an account's representation reflects only
     the funds it SENDS (incoming messages travel along edge direction); with it,
     the representation also reflects the funds it RECEIVES. Laundering is a
     two-sided phenomenon (a mule both receives and forwards), so this is the
     single most important enhancement for this task.
  2. PORT NUMBERING  (implemented)
     A plain message-passing scheme cannot tell two parallel transactions between
     the same pair of accounts apart -- yet "many small transfers between the same
     pair" is exactly the structuring/layering signature. Port numbering gives
     each edge an index recording its position within the set of parallel edges
     between its endpoints, restoring that distinction.

NOte ON WHAT GETS CLASSIFIED
Reverse edges exist ONLY to carry messages; they are not transactions and must
not be scored. All real transactions are therefore kept FIRST in the edge list
and append the reverse edges AFTER them, and return ``n_real`` (the count of
real edges). The model message-passes over everything but applies its
classification head only to the first ``n_real`` edges (see models.py,
``n_classify``). This keeps the set of scored edges identical to the GAT arms,
so the only thing that changes is the richer message passing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch


def _appearance_ports(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    """Port index = position of each edge within its (src, dst) group.

    Edges are numbered 0, 1, 2, ... in the order they appear in the edge list.
    Because the graph is built in chronological order, appearance order is
    (very nearly) time order, so this needs no separate timestamp array and
    stays correct after DropEdge removes edges or Edge-SMOTE appends them.

    A log1p transform keeps the value bounded: the vast majority of account
    pairs transact once (port 0 -> log1p(0) = 0), and the rare high-multiplicity
    pairs are compressed rather than dominating the feature.
    """
    n = src.shape[0]
    # Unique integer key per ordered (src, dst) pair. src, dst < N accounts, so
    # src*N + dst is unique; N up to ~1e6 keeps this well within int64 range.
    N = int(max(int(src.max()) if n else 0, int(dst.max()) if n else 0)) + 1
    key = (src.long() * N + dst.long()).cpu().numpy()
    # cumcount() gives, for each row, how many earlier rows share its key -> the
    # within-group appearance rank, in original order (no reindexing needed).
    port = pd.Series(np.zeros(n)).groupby(key).cumcount().to_numpy()
    return torch.log1p(torch.tensor(port, dtype=torch.float, device=src.device)).unsqueeze(1)


def add_reverse_and_ports(edge_index: torch.Tensor, edge_attr: torch.Tensor):
    """Augment a graph with reverse edges and port features for Multi-GAT.

    Adds exactly TWO edge-feature columns to every edge, so a Multi-GAT model is
    built with ``edge_dim = base_edge_features + 2``:
        column -2 : port index (log1p of within-pair appearance rank)
        column -1 : direction flag (0 = real transaction, 1 = reverse edge)

    Returns
    -------
    mp_edge_index : (2, 2*E) message-passing edges = [real | reverse]
    mp_edge_attr  : (2*E, F+2) matching features
    n_real        : int, number of real edges (the first n_real columns), i.e.
                    the edges the classifier will score.
    """
    src, dst = edge_index
    n = edge_index.shape[1]
    dev = edge_index.device

    # --- real edges: port at (src -> dst), direction flag 0 ------------------
    port = _appearance_ports(src, dst)
    zeros = torch.zeros(n, 1, device=dev)
    real_attr = torch.cat([edge_attr, port, zeros], dim=1)

    # --- reverse edges: flip direction, port at (dst -> src), flag 1 ---------
    rev_index = torch.stack([dst, src], dim=0)
    rport = _appearance_ports(dst, src)
    ones = torch.ones(n, 1, device=dev)
    rev_attr = torch.cat([edge_attr, rport, ones], dim=1)

    mp_edge_index = torch.cat([edge_index, rev_index], dim=1)
    mp_edge_attr = torch.cat([real_attr, rev_attr], dim=0)
    return mp_edge_index, mp_edge_attr, n
