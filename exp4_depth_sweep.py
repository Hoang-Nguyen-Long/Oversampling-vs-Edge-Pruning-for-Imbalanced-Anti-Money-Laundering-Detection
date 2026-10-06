"""
EXPERIMENT 4 -- Depth Sweep  (Chapter 3, Section 3.9.2)
Every condition at every pre-specified depth L in {2, 4, 8, 16}. This is the
over-smoothing evidence: not a search for the best depth, but a measurement of
how each method's performance CHANGES with depth, alongside the layer-wise
Dirichlet-energy and MAD profiles that explain why.

What to look for (H2/H4): the plain baseline should degrade as depth grows;
DropEdge should degrade more slowly; GraphSMOTE-Edge's benefit should be roughly
depth-independent.

CRITICAL: the subset size in common.py must be IDENTICAL for all depths, or the
depth effect is confounded with dataset size. Deeper models use more memory, so
choose a subset that survives depth 16 and keep it.

Run ONE DEPTH PER SESSION on Colab (restart between them) so a crash at depth 16
does not cost the earlier depths. Results are checkpointed per condition.

    python exp4_depth_sweep.py --depths 2
    python exp4_depth_sweep.py --depths 4
    ...
"""
import argparse
import common as C
from run_experiments import ABLATION, DEPTH_SWEEP_CONDITIONS, run

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--depths", nargs="+", type=int, default=C.DEPTHS)
    ap.add_argument("--only", nargs="+", default=DEPTH_SWEEP_CONDITIONS)
    ap.add_argument("--probe", action="store_true",
                    help="memory probe: write to a separate file so the real "
                         "sweep results are never overwritten or invalidated")
    ap.add_argument("--edge-updates", dest="edge_updates", action="store_true",
                    help="enable the per-layer edge-update MLP for ALL conditions "
                         "(roughly triples activation memory; caps the feasible depth)")
    ap.add_argument("--no-edge-updates", dest="edge_updates", action="store_false",
                    help="disable edge updates for ALL conditions (default; allows deeper sweeps)")
    ap.set_defaults(edge_updates=C.EDGE_UPDATES)
    a = ap.parse_args()

    # The suffix records the setting in the filename so runs with and without
    # edge updates cannot silently overwrite one another.
    eu_tag = "_eu" if a.edge_updates else ""

    tuned = C.load("exp2_tuning")
    csv = C.subset_path()
    for depth in a.depths:
        C.logger.info("#" * 64)
        C.logger.info("DEPTH %d", depth)
        run(csv_path=csv, experiments=a.only, seeds=C.SEEDS, depth=depth,
            epochs=C.MAX_EPOCHS, device=C.DEVICE,
            out_path=(f"{C.RESULTS_DIR}/_probe_depth{depth}{eu_tag}.json" if a.probe
                      else f"{C.RESULTS_DIR}/exp4_depth{depth}{eu_tag}.json"),
            tuned=tuned, edge_updates=a.edge_updates)

if __name__ == "__main__":
    C.ensure_dirs(); C.require_gpu(); main()
