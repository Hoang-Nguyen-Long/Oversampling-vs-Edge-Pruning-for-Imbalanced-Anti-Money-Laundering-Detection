"""
diagnostics.py
Over-smoothing diagnostics (Chapter 4 / Chapter 5 link).
Two complementary measures, computed layer-wise on node embeddings, both of
which trend to zero as embeddings collapse to a common value:
* Dirichlet energy: mean squared distance between connected nodes'
  representations (Rusch et al., 2023). Reported alongside the embedding norm,
  because it is sensitive to embedding magnitude.
* Mean Average Distance (MAD): mean pairwise cosine distance between node
  representations (Chen, D. et al., 2020). Angular, hence scale-invariant.
MEASUREMENT PROTOCOL (fixed in advance so arms are comparable):
- computed in EVALUATION mode on the restored (un-dropped) graph;
- on a fixed random sample of nodes, held constant across arms and seeds;
- reported per layer, for every depth in the sweep.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


@torch.no_grad()
def dirichlet_energy(embeddings: torch.Tensor, edge_index: torch.Tensor) -> float:
    """Mean squared distance between the representations of connected nodes.
    """
    src, dst = edge_index
    diff = embeddings[src] - embeddings[dst]
    return float((diff * diff).sum(dim=1).mean().item())


@torch.no_grad()
def mean_average_distance(embeddings: torch.Tensor, sample_size: int = 1000,
                          generator: torch.Generator | None = None) -> float:
    """Average pairwise cosine distance over a random node sample.
    A random sample keeps the O(n^2) computation affordable on large graphs and,
    crucially, the SAME sample indices are reused across arms (the caller passes
    a fixed generator) so differences cannot come from sampling.
    """
    n = embeddings.shape[0]
    if n > sample_size:
        idx = torch.randperm(n, generator=generator)[:sample_size]
        embeddings = embeddings[idx]
    normed = F.normalize(embeddings, p=2, dim=1)
    cosine = normed @ normed.T                    # (m, m) cosine similarities
    m = cosine.shape[0]
    off_diag = ~torch.eye(m, dtype=torch.bool, device=cosine.device)
    return float((1.0 - cosine[off_diag]).mean().item())


@torch.no_grad()
def profile_layers(model, x, edge_index, edge_attr, mad_sample: int = 1000,
                   generator: torch.Generator | None = None) -> dict:
    """Run one forward pass and return per-layer Dirichlet energy, MAD and norm.
    Requires a model whose forward accepts ``return_embeddings=True`` (GATe does).
    """
    model.eval()
    _, layer_embeddings = model(x, edge_index, edge_attr, return_embeddings=True)
    profile = {"dirichlet": [], "mad": [], "norm": []}
    for h in layer_embeddings:
        profile["dirichlet"].append(dirichlet_energy(h, edge_index))
        profile["mad"].append(mean_average_distance(h, mad_sample, generator))
        profile["norm"].append(float(h.norm(dim=1).mean().item()))
    return profile


@torch.no_grad()
def profile_layers_full(model, x, edge_index, edge_attr, y=None,
                        classify_index=None, mad_sample: int = 1000,
                        generator: torch.Generator | None = None) -> dict:
    """Layer-wise diagnostics reported BOTH globally and for minority-incident nodes.
    Chapter 3, Section 3.10.5 requires each diagnostic to be reported globally and
    restricted to accounts incident to at least one illicit transaction, because
    representation collapse specifically within the minority neighbourhood is the
    mechanism of interest. It also requires the mean representation norm alongside
    Dirichlet energy, since that energy is magnitude-sensitive.

    Parameters
    y : optional labels for the *classified* edges (used to find illicit edges).
    classify_index : the edge_index columns corresponding to y (for Multi-GAT the
        message-passing graph contains appended reverse edges, so the first
        ``len(y)`` columns are the real, labelled transactions).
    """
    model.eval()
    out = model(x, edge_index, edge_attr, return_embeddings=True)
    _, layer_embeddings = out if isinstance(out, tuple) else (None, [])

    # Identify accounts touched by at least one illicit transaction.
    minority_nodes = None
    if y is not None:
        ci = classify_index if classify_index is not None else edge_index[:, :len(y)]
        ill = (y == 1).nonzero(as_tuple=True)[0]
        if ill.numel() > 0:
            minority_nodes = torch.unique(ci[:, ill].reshape(-1))

    prof = {"dirichlet": [], "mad": [], "norm": [],
            "dirichlet_minority": [], "mad_minority": [], "norm_minority": []}
    for h in layer_embeddings:
        prof["dirichlet"].append(dirichlet_energy(h, edge_index))
        prof["mad"].append(mean_average_distance(h, mad_sample, generator))
        prof["norm"].append(float(h.norm(dim=1).mean().item()))
        if minority_nodes is not None and minority_nodes.numel() > 1:
            # Restrict to the sub-graph induced on minority-incident accounts.
            mask = torch.zeros(h.shape[0], dtype=torch.bool, device=h.device)
            mask[minority_nodes] = True
            src, dst = edge_index
            keep = mask[src] & mask[dst]
            sub_ei = edge_index[:, keep]
            prof["dirichlet_minority"].append(
                dirichlet_energy(h, sub_ei) if sub_ei.shape[1] > 0 else float("nan"))
            prof["mad_minority"].append(
                mean_average_distance(h[minority_nodes], mad_sample, generator))
            prof["norm_minority"].append(float(h[minority_nodes].norm(dim=1).mean().item()))
        else:
            prof["dirichlet_minority"].append(float("nan"))
            prof["mad_minority"].append(float("nan"))
            prof["norm_minority"].append(float("nan"))
    return prof
