"""
run_experiments.py
Experiment runner implementing the formal ablation matrix (Chapter 4, Section D).

This encodes the E1-E9 ablation table from the supervisor's feedback (point 9),
extended with the non-graph MLP baseline (point 5) as E0. Each experiment is a
(model, intervention, loss) triple, run over multiple seeds (point 7), with
mean +/- std reported for every metric.

    ID  Model      Intervention  Loss         Purpose
    E0  MLP        none          weighted_ce  Non-graph baseline (does graph help?)
    E1  GAT        none          ce           Basic baseline
    E2  GAT        none          weighted_ce  Effect of class weighting
    E3  GAT        none          focal        Effect of imbalance-aware loss
    E4  GAT        dropedge      weighted_ce  Simple pruning baseline
    E5  GAT        edge_smote    weighted_ce  Oversampling (expansion)
    E6  Multi-GAT  none          weighted_ce  Effect of stronger architecture
    E7  Multi-GAT  dropedge      weighted_ce  Best contraction configuration
    E8  Multi-GAT  edge_smote    weighted_ce  Best expansion configuration

Usage:
    python run_experiments.py --csv /path/formatted_transactions.csv \
        --seeds 0 1 2 3 4 --epochs 50 --experiments E1 E2 E5 E8
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import tempfile

import torch

from data_understanding import load_transactions, analyse
from data_pipeline import build_graphs
from models import build_model
from trainer import train_and_evaluate
from diagnostics import profile_layers, profile_layers_full
from evaluation import aggregate_seeds

logger = logging.getLogger(__name__)

# The ablation matrix as data, so adding an experiment is a one-line change.
ABLATION = {
    "E0": dict(model="mlp",       intervention="none",       loss="weighted_ce"),
    "E1": dict(model="gat",       intervention="none",       loss="ce"),
    "E2": dict(model="gat",       intervention="none",       loss="weighted_ce"),
    "E3": dict(model="gat",       intervention="none",       loss="focal"),
    "E4": dict(model="gat",       intervention="dropedge",   loss="weighted_ce"),
    "E5": dict(model="gat",       intervention="edge_smote", loss="weighted_ce"),
    "E6": dict(model="multi_gat", intervention="none",       loss="weighted_ce"),
    "E7": dict(model="multi_gat", intervention="dropedge",   loss="weighted_ce"),
    "E8": dict(model="multi_gat", intervention="edge_smote", loss="weighted_ce"),
    # E9 isolates INTERPOLATION from mere repeated minority exposure: it copies
    # observed illicit transactions verbatim instead of interpolating new ones.
    # Any advantage E5 holds over E9 is attributable to interpolation itself.
    "E9": dict(model="gat",       intervention="duplicate",  loss="weighted_ce"),
}

# Conditions carried through the depth sweep. Running all ten at four depths is
# not affordable; these five are the ones the depth argument (H2/H4) needs -- the
# weighted-CE baseline plus contraction and expansion on each backbone.
# Conditions carried through the depth sweep.
#
# E6 (Multi-GAT with no intervention) belongs here even though it was omitted
# from the original economy. Without it, no comparison at depth isolates the
# contribution of contraction from that of the backbone: E7 differs from the
# baseline E2 in TWO respects, and the nearest available stand-in, E8, carries an
# expansion intervention of its own, so E7 vs E8 compares contraction against
# expansion rather than against the bare architecture. E6 supplies the missing
# arm, making E7 vs E6 the clean isolating comparison.
DEPTH_SWEEP_CONDITIONS = ["E2", "E4", "E5", "E6", "E7", "E8"]


def default_config(depth: int, edge_updates: bool = False) -> dict:
    """Base hyperparameters, matched across arms (only depth/intervention vary)."""
    return dict(
        lr=0.006, n_hidden=64, n_heads=4, n_gnn_layers=depth,
        dropout=0.1, final_dropout=0.1, edge_updates=edge_updates, epochs=50,
        drop_rate=0.2, smote_ratio=1.0, smote_k=5,
        focal_alpha=0.25, focal_gamma=2.0, cb_beta=0.9999,
    )


def _safe_write(obj, path):
    """Write JSON, falling back to the temp dir then cwd if the path is unwritable."""
    for cand in [path, os.path.join(tempfile.gettempdir(), os.path.basename(path)),
                 os.path.basename(path)]:
        try:
            parent = os.path.dirname(cand)
            if parent:
                os.makedirs(parent, exist_ok=True)
            with open(cand, "w") as f:
                json.dump(obj, f, indent=2)
            return cand
        except OSError:
            continue
    return None


def _strip_scores(per_seed):
    """Separate the raw score vectors from the scalar metrics.

    They are kept out of the main results file because of size: one vector per
    seed per condition would inflate it by orders of magnitude and make it slow
    to read while writing up. They go to a companion ``*_scores.json`` instead,
    loaded only by the analyses that need them.
    """
    scores = [d.pop("_scores", None) for d in per_seed]
    return per_seed, scores


def run(csv_path, experiments, seeds, depth, epochs, device, out_path=None, resume=True,
        tuned=None, overrides=None, edge_updates=False, allow_fingerprint_drift=False):
    df = load_transactions(csv_path)
    analyse(df)  # Chapter 4 data-understanding stage.

    # Multi-GAT arms apply reverse message passing and port numbering at the data
    # level (see multigat.py); the model class is shared with GAT so the backbone
    # comparison stays controlled.
    graphs = build_graphs(df)
    n_node_f = graphs["n_node_features"]
    n_edge_f = graphs["n_edge_features"]
    logger.info("Graph built: %d node feats, %d edge feats | edge_updates=%s",
                n_node_f, n_edge_f, edge_updates)

    # ---- Resume support, guarded by a configuration fingerprint ------------
    device_name = (torch.cuda.get_device_name(0)
                   if torch.cuda.is_available() else "cpu")
    run_config = {
        "seeds": list(seeds),
        "epochs": int(epochs),
        "edge_updates": bool(edge_updates),
        "depth": int(depth),
        "device": device_name,
        # Whether this run persisted prediction scores. Without it a file
        # produced before score persistence is indistinguishable from one
        # produced after -- the configuration is identical -- so resume skips it
        # and the score-based analyses stay blocked with no indication why.
        "saves_scores": True,
    }
    results = {}
    if out_path and resume and os.path.exists(out_path):
        try:
            saved = json.load(open(out_path))
            saved_config = saved.pop("_config", None)
            if saved_config is not None and saved_config != run_config \
                    and allow_fingerprint_drift:
                # Appending an arm to a completed file. The existing conditions
                # were trained under settings that differ only in bookkeeping
                # (for instance score persistence, added later), not in anything
                # that affects what a model sees, so discarding them would mean
                # retraining every arm to add one. The drift is reported rather
                # than silently accepted.
                drift = [k for k in run_config if saved_config.get(k) != run_config[k]]
                logger.warning("Appending despite fingerprint drift in: %s",
                               ", ".join(drift))
                logger.warning("The existing conditions are kept; only the named "
                               "ones are trained.")
                results = saved
                done = [e for e in experiments if e in results]
                if done:
                    logger.info("Already present: %s", ", ".join(done))
            elif saved_config is not None and saved_config != run_config:
                changed = [k for k in run_config
                           if saved_config.get(k) != run_config[k]]
                logger.warning(
                    "Configuration changed since %s was written (%s). "
                    "Discarding the cached results and re-running: reusing them "
                    "would mix settings within one comparison.",
                    os.path.basename(out_path), ", ".join(changed))
                for k in changed:
                    logger.warning("    %-13s saved=%s  now=%s",
                                   k, saved_config.get(k), run_config[k])
            else:
                if saved_config is None:
                    logger.warning(
                        "%s predates configuration fingerprinting; its settings "
                        "cannot be verified. Discarding it and re-running.",
                        os.path.basename(out_path))
                else:
                    results = saved
                    done = [e for e in experiments if e in results]
                    if done:
                        logger.info("Resuming: %d experiment(s) already complete (%s)",
                                    len(done), ", ".join(done))
        except (OSError, ValueError):
            logger.warning("Could not read %s; starting fresh.", out_path)

    for eid in experiments:
        if eid in results:
            continue  # already done in a previous session
        spec = ABLATION[eid]
        logger.info("=" * 60)
        logger.info("Experiment %s: %s", eid, spec)
        per_seed = []
        smoothing = None
        is_multi = spec["model"] == "multi_gat"
        for s in seeds:
            cfg = default_config(depth, edge_updates=edge_updates)
            cfg["epochs"] = epochs

            if tuned and eid in tuned:
                tuned_eu = (tuned[eid].get("decision", {}) or {}).get("edge_updates")
                if tuned_eu is None:
                    tuned_eu = tuned[eid].get("edge_updates")
                if tuned_eu is not None and bool(tuned_eu) != bool(edge_updates):
                    logger.warning(
                        "%s: tuned with edge_updates=%s but scoring with %s. "
                        "Re-run exp2_tuning.py so the two agree.",
                        eid, tuned_eu, edge_updates)
            # Apply per-condition tuned hyperparameters from Experiment 2, if given.
            if tuned and eid in tuned and tuned[eid].get("best"):
                b = tuned[eid]["best"]
                cfg["lr"] = b["lr"]
                cfg["dropout"] = b["dropout"]
                cfg["final_dropout"] = b["dropout"]
            # Explicit overrides (used by the intervention-strength sweeps).
            if overrides:
                cfg.update(overrides)
            # Multi-GAT adds 2 edge-feature columns (port + direction flag), so
            # its model is built with a wider edge embedding input.
            model_edge_dim = n_edge_f + (2 if is_multi else 0)
            model = build_model(spec["model"], n_node_f, model_edge_dim, cfg)
            metrics, trained = train_and_evaluate(
                model, graphs, cfg, device, seed=s,
                intervention=spec["intervention"], loss_name=spec["loss"],
                multi_gat=is_multi)
            per_seed.append(metrics)
            logger.info("  seed %d -> AUPRC %.4f | recall %.4f | F1 %.4f",
                        s, metrics["auprc"], metrics["recall"], metrics["f1"])
            # Over-smoothing profile from the last seed's trained model. For
            # Multi-GAT the diagnostic graph must carry the same reverse+port
            # augmentation the model was trained with, or the shapes mismatch.
            if s == seeds[-1] and spec["model"] != "mlp":
                gen = torch.Generator().manual_seed(12345)  # fixed sample across arms
                te = graphs["test"]
                # Prepare the diagnostic graph once, on the device, carrying the
                # same augmentation the model was trained with (otherwise the
                # edge-feature width would not match the model's embedding).
                import graph_cache
                dg = graph_cache.prepare(te.x, te.edge_index, te.edge_attr,
                                         te.y, device, is_multi)
                smoothing = profile_layers_full(
                    trained, dg.x, dg.edge_index, dg.edge_attr,
                    y=dg.y, classify_index=dg.edge_index[:, :dg.n_real],
                    generator=gen)
                graph_cache.free(dg)

            # Release this seed's model before the next one is built. Otherwise
            # every seed and every condition accumulates on the GPU, and the
            # deepest/heaviest configuration runs on top of all its predecessors.
            del model, trained
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        per_seed, seed_scores = _strip_scores(per_seed)
        if out_path and any(s is not None for s in seed_scores):
            score_path = out_path.replace(".json", "_scores.json")
            existing = {}
            if os.path.exists(score_path):
                try:
                    existing = json.load(open(score_path))
                except (OSError, ValueError):
                    existing = {}
            existing[eid] = {"seeds": list(seeds), "per_seed": seed_scores}
            _safe_write(existing, score_path)

        results[eid] = {"spec": dict(spec, edge_updates=edge_updates),
                        "aggregate": aggregate_seeds(per_seed),
                        "per_seed": per_seed, "smoothing": smoothing}
        # Checkpoint immediately: this experiment's result is now safe on disk.
        # Stamp the configuration so a later invocation can tell whether these
        # results were produced under the same settings (see the resume block).
        results["_config"] = run_config
        _safe_write(results, out_path or os.path.join(tempfile.gettempdir(), "results.json"))
        results.pop("_config")   # keep the in-memory object clean for callers

    logger.info("Sweep complete: %d experiment(s) in %s", len(results), out_path)
    return results


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--experiments", nargs="+", default=list(ABLATION.keys()))
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    p.add_argument("--depth", type=int, default=2, help="GNN layers (depth sweep uses several)")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--out", default="results.json")
    a = p.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)
    run(a.csv, a.experiments, a.seeds, a.depth, a.epochs, device, a.out)


if __name__ == "__main__":
    main()
