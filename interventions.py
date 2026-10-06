"""
interventions.py

"""

from __future__ import annotations

import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ===========================================================================
# GRAPH CONTRACTION: DropEdge
# ===========================================================================
def drop_edge(edge_index: torch.Tensor, edge_attr: torch.Tensor,
              drop_rate: float, generator: torch.Generator | None = None):
    """Randomly remove a proportion of MESSAGE-PASSING edges (Rong et al., 2020).

    IMPORTANT (Chapter 3, Section 3.7). This function returns a contracted
    *message-passing* graph only. The supervised edge set is NOT passed in and is
    NOT modified, so every training transaction still receives a label and enters
    the loss. This separation matters: if dropped edges were also removed from the
    loss, the intervention would simultaneously contract the topology AND randomly
    undersample the supervised examples, and a fall in recall could not be
    attributed to contraction rather than to having discarded positive examples.

    Sampling is uniform and class-agnostic; illicit edges are not protected. The
    reason is NOT that using training labels would leak -- training labels are
    legitimately available at training time -- but that protecting positives would
    no longer be canonical, label-agnostic DropEdge and would therefore constitute
    a different intervention.

    Returns the sparsified (edge_index, edge_attr). Resampled every epoch.
    """
    n = edge_index.shape[1]
    keep = torch.rand(n, generator=generator) >= drop_rate
    return edge_index[:, keep], edge_attr[keep]


# SIGNAL EXPANSION: Edge-SMOTE (edge-level GraphSMOTE adaptation)
def edge_smote(
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    y: torch.Tensor,
    oversample_ratio: float = 1.0,
    k_neighbours: int = 5,
    generator: torch.Generator | None = None,
    continuous_cols=None,
    categorical_cols=None,
    cyclical_spec=None,
):
    """GraphSMOTE-Edge: oversample minority transactions (Chapter 3, Section 3.6).

    Three points of the specification are enforced here, each following directly
    from a methodological objection:
    1. CONTINUOUS-ONLY INTERPOLATION (Section 3.6.1). Only continuous columns are
       interpolated. Categorical codes (currency, payment format) are INHERITED
       from the anchor, because a value halfway between two arbitrary payment-type
       codes denotes no real payment type. Interpolating them would manufacture
       transactions that could not occur.
    2. TEMPORAL CONSISTENCY (Section 3.6.2). The synthetic transaction inherits the
       anchor's timestamp, and its cyclical hour/day features are RECOMPUTED from
       that timestamp rather than interpolated. Interpolating the cyclical features
       independently while copying the timestamp would place contradictory temporal
       information in the same vector.
    3. TOPOLOGY (Section 3.6.3). Endpoints are inherited from the anchor, so no
       synthetic account is created and the node set is unchanged.

    Parameters
    ----------
    continuous_cols : indices interpolated (default: all but col 0).
    categorical_cols : indices inherited wholesale from the anchor.
    cyclical_spec : list of (col_index, period_seconds, kind) with kind in
        {"sin","cos"}, recomputed from the inherited timestamp (column 0).
    """
    minority_idx = (y == 1).nonzero(as_tuple=True)[0]
    n_minority = minority_idx.numel()
    if n_minority < 2:
        logger.warning("GraphSMOTE-Edge: only %d minority edges; skipping.", n_minority)
        return edge_index, edge_attr, y
    n_synth = int(round(oversample_ratio * n_minority))
    if n_synth <= 0:
        return edge_index, edge_attr, y

    F_dim = edge_attr.shape[1]
    if continuous_cols is None:
        continuous_cols = [c for c in range(1, F_dim)]
    if categorical_cols is None:
        categorical_cols = []
    continuous_cols = [c for c in continuous_cols if c not in categorical_cols]

    min_attr = edge_attr[minority_idx]
    # Distances computed on the interpolated (continuous) subspace only, so that
    # neighbour choice is not driven by meaningless categorical-code distances.
    feat = min_attr[:, continuous_cols]
    with torch.no_grad():
        dist = torch.cdist(feat, feat)
        dist.fill_diagonal_(float("inf"))
        k = min(k_neighbours, n_minority - 1)
        knn = dist.topk(k, largest=False).indices

    anchor = torch.randint(0, n_minority, (n_synth,), generator=generator)
    nbr = knn[anchor, torch.randint(0, k, (n_synth,), generator=generator)]
    lam = torch.rand(n_synth, 1, generator=generator)

    # Start from a copy of the anchor: this inherits timestamp, categoricals and
    # anything not explicitly interpolated.
    synth_attr = min_attr[anchor].clone()
    synth_attr[:, continuous_cols] = (
        feat[anchor] + lam * (feat[nbr] - feat[anchor]))

    # Recompute cyclical features from the inherited timestamp (column 0).
    if cyclical_spec:
        t = synth_attr[:, 0]
        for col, period, kind in cyclical_spec:
            ang = 2 * math.pi * (t % period) / period
            synth_attr[:, col] = torch.sin(ang) if kind == "sin" else torch.cos(ang)

    synth_edge_index = edge_index[:, minority_idx[anchor]]
    synth_y = torch.ones(n_synth, dtype=y.dtype)
    logger.info("GraphSMOTE-Edge: +%d synthetic illicit edges (%d -> %d minority).",
                n_synth, n_minority, n_minority + n_synth)
    return (torch.cat([edge_index, synth_edge_index], dim=1),
            torch.cat([edge_attr, synth_attr], dim=0),
            torch.cat([y, synth_y], dim=0))


def minority_duplicate(edge_index: torch.Tensor, edge_attr: torch.Tensor,
                       y: torch.Tensor, oversample_ratio: float = 1.0,
                       generator: torch.Generator | None = None):
    """Random minority oversampling WITHOUT interpolation (control baseline).

    Chapter 3, Section 3.6.8. GraphSMOTE-Edge both (a) interpolates new feature
    vectors and (b) increases the multiplicity of existing minority relationships.
    Any gain it shows could therefore come from repeated exposure to the same
    minority relationships rather than from interpolation being useful.

    This baseline isolates that: it duplicates observed illicit transactions
    verbatim, adding the same number of edges between the same endpoint pairs, but
    generating no new feature values. Comparing it against GraphSMOTE-Edge
    attributes any difference to interpolation specifically.
    """
    minority_idx = (y == 1).nonzero(as_tuple=True)[0]
    n_minority = minority_idx.numel()
    if n_minority < 1:
        return edge_index, edge_attr, y
    n_synth = int(round(oversample_ratio * n_minority))
    if n_synth <= 0:
        return edge_index, edge_attr, y
    pick = minority_idx[torch.randint(0, n_minority, (n_synth,), generator=generator)]
    logger.info("Minority duplication: +%d copied illicit edges.", n_synth)
    return (torch.cat([edge_index, edge_index[:, pick]], dim=1),
            torch.cat([edge_attr, edge_attr[pick]], dim=0),
            torch.cat([y, torch.ones(n_synth, dtype=y.dtype)], dim=0))


# IMBALANCE-AWARE LOSSES (bridging baselines)
def class_weighted_ce(class_counts: torch.Tensor, beta: float = 0.9999,
                      device=None) -> nn.Module:
    """Class-balanced cross-entropy using the effective number of samples.

    Weights follow Cui et al. (2019): w_c = (1 - beta) / (1 - beta^{n_c}). Under
    extreme skew this is more stable than raw inverse frequency. ``beta`` close
    to 1 approaches inverse-frequency weighting; ``beta = 0`` gives no weighting.
    """
    effective_num = 1.0 - torch.pow(beta, class_counts.float())
    weights = (1.0 - beta) / torch.clamp(effective_num, min=1e-12)
    weights = weights / weights.sum() * len(class_counts)  # normalise to mean 1
    if device is not None:
        weights = weights.to(device)
    logger.info("Class-balanced CE weights (beta=%.4f): %s", beta, weights.tolist())
    return nn.CrossEntropyLoss(weight=weights)


class FocalLoss(nn.Module):
    """Focal loss for extreme imbalance (Lin, T. Y. et al., 2017).

    FL(p_t) = -alpha_t (1 - p_t)^gamma log(p_t). The modulating factor
    (1 - p_t)^gamma down-weights easy, well-classified examples so training
    focuses on hard (typically minority) cases. Originally proposed for dense
    object detection; used here as a strong imbalance baseline whose transfer to
    graph-based AML is being tested, not assumed.
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, target, reduction="none")
        p_t = torch.exp(-ce)                       # prob of the true class
        # alpha weighting: alpha for the positive class, (1-alpha) for negative.
        alpha_t = torch.where(target == 1, self.alpha, 1 - self.alpha)
        loss = alpha_t * (1 - p_t) ** self.gamma * ce
        return loss.mean()


def build_loss(loss_name: str, class_counts: torch.Tensor, device=None,
               focal_alpha: float = 0.25, focal_gamma: float = 2.0,
               cb_beta: float = 0.9999) -> nn.Module:
    """Factory for the three loss options used across the ablation arms."""
    if loss_name == "ce":
        return nn.CrossEntropyLoss()
    if loss_name == "weighted_ce":
        return class_weighted_ce(class_counts, beta=cb_beta, device=device)
    if loss_name == "focal":
        return FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
    raise ValueError(f"Unknown loss '{loss_name}'")
