"""
EXPERIMENT 5 -- Intervention Strength Sweeps  (Chapter 3, Section 3.9.3; RQ5)
Section 3.9.3 commits that "no value is adopted from the literature without
being varied". This experiment delivers that, and with it the
intervention-strength component of RQ5 (robustness):

    DropEdge          drop rate p in {0.1, 0.2, 0.3, 0.5}
    GraphSMOTE-Edge   ratio r in {0.5, 1.0, 2.0}  x  neighbourhood K in {5, 10}

Each strength is a separate run of the relevant condition, at the reference
depth, over all seeds. Reporting performance ACROSS the range -- rather than at
one tuned value -- is what makes the robustness claim, so the sweep results are
reported as curves, not collapsed to a best value.

Interpretation to expect: DropEdge recall should fall as p rises (H3), because
uniform removal deletes scarce minority edges; GraphSMOTE-Edge should improve
then plateau or degrade as r grows, since interpolation eventually adds
redundant or implausible samples.

    python exp5_intervention_strength.py                # both sweeps
    python exp5_intervention_strength.py --which dropedge
"""
import argparse
import common as C
from run_experiments import run

# DOWNSIZED grid (Section 3.9.4). The purpose is to show the SHAPE of the
# sensitivity curve, not to locate an optimum, so three well-separated points per
# intervention suffice. K is fixed at 5 rather than swept: it changes only which
# minority neighbours are eligible for interpolation and moved results far less
# than the oversampling ratio. Sweeps run on the GAT backbone only, since the
# question is how each intervention responds to its own strength, not how that
# response differs by backbone.
DROP_RATES = [0.1, 0.3, 0.5]
SMOTE_RATIOS = [0.5, 1.0, 2.0]
SMOTE_KS = [5]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", choices=["dropedge", "smote", "both"], default="both")
    ap.add_argument("--depth", type=int, default=C.REFERENCE_DEPTH)
    a = ap.parse_args()

    tuned, csv = C.load("exp2_tuning"), C.subset_path()

    # This experiment keeps its own cache keyed by setting, so the fingerprint
    # guard inside run() never sees it: run() is called with resume=False and a
    # throwaway path, and the decision to skip is made here. Without a matching
    # guard, the cache would survive a change of seeds or epochs and results from
    # different configurations could be combined.
    import torch
    run_config = {"seeds": list(C.SEEDS), "epochs": int(C.MAX_EPOCHS),
                  "edge_updates": bool(C.EDGE_UPDATES), "depth": int(a.depth),
                  "device": (torch.cuda.get_device_name(0)
                             if torch.cuda.is_available() else "cpu")}
    results = C.load("exp5_intervention_strength") or {}
    saved_config = results.pop("_config", None)
    if results and saved_config != run_config:
        if saved_config is None:
            C.logger.warning("Cached strength sweep predates configuration "
                             "fingerprinting; discarding it and re-running.")
        else:
            changed = [k for k in run_config if saved_config.get(k) != run_config[k]]
            C.logger.warning("Strength-sweep settings changed (%s); discarding the "
                             "cache and re-running.", ", ".join(changed))
            for k in changed:
                C.logger.warning("    %-13s saved=%s  now=%s",
                                 k, saved_config.get(k), run_config[k])
        results = {}

    if a.which in ("dropedge", "both"):
        for p in DROP_RATES:
            key = f"dropedge_p{p}"
            if key in results:
                continue
            C.logger.info("=== DropEdge rate p=%.2f (E4, E7) ===", p)
            r = run(csv_path=csv, experiments=["E4"], seeds=C.SEEDS,
                    depth=a.depth, epochs=C.MAX_EPOCHS, device=C.DEVICE,
                    out_path=f"{C.RESULTS_DIR}/_tmp_{key}.json", resume=False,
                    tuned=tuned, overrides={"drop_rate": p})
            results[key] = {"drop_rate": p, "results": r}
            C.save({**results, "_config": run_config}, "exp5_intervention_strength")

    if a.which in ("smote", "both"):
        for ratio in SMOTE_RATIOS:
            for k in SMOTE_KS:
                key = f"smote_r{ratio}_k{k}"
                if key in results:
                    continue
                C.logger.info("=== GraphSMOTE-Edge ratio=%.1f K=%d (E5, E8) ===", ratio, k)
                r = run(csv_path=csv, experiments=["E5"], seeds=C.SEEDS,
                        depth=a.depth, epochs=C.MAX_EPOCHS, device=C.DEVICE,
                        out_path=f"{C.RESULTS_DIR}/_tmp_{key}.json", resume=False,
                        tuned=tuned, overrides={"smote_ratio": ratio, "smote_k": k})
                results[key] = {"smote_ratio": ratio, "smote_k": k, "results": r}
                C.save({**results, "_config": run_config}, "exp5_intervention_strength")

    C.logger.info("Strength sweep complete: %d settings.", len(results))

if __name__ == "__main__":
    C.ensure_dirs(); C.require_gpu(); main()
