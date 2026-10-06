"""
graph_cache.py
Prepares each graph ONCE, on the target device, instead of rebuilding it on every
forward pass.
The earlier implementation called ``add_reverse_and_ports`` inside every forward
pass -- roughly three times per epoch (training, validation scoring, and test
scoring). Each call ran a pandas groupby on the CPU, built an edge tensor of
twice the original size, and copied it to the GPU. Over a hundred epochs that is
several hundred rebuilds of the largest tensor in the program.
The consequence was not merely slowness. PyTorch's caching allocator holds freed
blocks for reuse, and repeatedly allocating large tensors of slightly varying
size fragments that pool, so a run can exhaust memory while reporting a large
"reserved but unallocated" figure. That is the out-of-memory pattern this module
removes.

Two changes:
  1. Every graph is moved to the device once, at the start of training.
  2. The Multi-GAT augmentation (reverse edges and port numbers) is computed once
     per graph and kept. Validation and test graphs never change, so recomputing
     them was pure waste.

DROPEDGE UNDER PRECOMPUTATION
DropEdge still resamples every epoch, but now by masking the precomputed tensors
rather than rebuilding them. The mask is generated over the real transactions and
applied to the reverse edges as well, so a dropped transaction disappears in both
directions. That is also more coherent than dropping the two directions
independently: a transaction either participates in message passing or it does
not.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch

from multigat import add_reverse_and_ports

logger = logging.getLogger(__name__)


@dataclass
class PreparedGraph:
    """A graph already on the device, with any Multi-GAT augmentation applied.

    Attributes
    ----------
    x, edge_index, edge_attr : the message-passing tensors, on the device.
    y                        : labels for the REAL edges only.
    n_real                   : number of real transactions. For Multi-GAT the
                               reverse edges occupy positions n_real: onwards and
                               are never scored.
    """

    x: torch.Tensor
    edge_index: torch.Tensor
    edge_attr: torch.Tensor
    y: torch.Tensor
    n_real: int
    multi_gat: bool

    def dropped(self, drop_rate: float, generator: torch.Generator | None = None):
        """Return (edge_index, edge_attr) with a fresh random subset removed.

        Masks the precomputed tensors rather than rebuilding them. The same mask
        is applied to the reverse block so that a dropped transaction is absent in
        both directions.
        """
        keep = torch.rand(self.n_real, generator=generator) >= drop_rate
        if self.multi_gat:
            keep_full = torch.cat([keep, keep])          # mirror onto reverse edges
        else:
            keep_full = keep
        keep_full = keep_full.to(self.edge_index.device)
        return self.edge_index[:, keep_full], self.edge_attr[keep_full]


def prepare(x, edge_index, edge_attr, y, device, multi_gat: bool) -> PreparedGraph:
    """Move a graph to the device, applying the Multi-GAT augmentation once."""
    if multi_gat:
        edge_index, edge_attr, n_real = add_reverse_and_ports(edge_index, edge_attr)
    else:
        n_real = edge_index.shape[1]
    return PreparedGraph(
        x=x.to(device),
        edge_index=edge_index.to(device),
        edge_attr=edge_attr.to(device),
        y=y.to(device) if y is not None else None,
        n_real=n_real,
        multi_gat=multi_gat,
    )


def free(*graphs):
    """Drop references to prepared graphs and release cached GPU memory."""
    import gc
    for g in graphs:
        if g is None:
            continue
        for attr in ("x", "edge_index", "edge_attr", "y"):
            setattr(g, attr, None)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
