"""
EXPERIMENT 8 -- Regenerate prediction scores for specific conditions
The maximum-F1, bootstrap and inductive analyses all read raw prediction scores.
Earlier builds computed those scores, used them for metrics, and discarded them,
so results produced before score persistence cannot support those analyses.

This script re-runs ONLY what is needed, rather than the whole programme.

    python exp8_add_scores.py --depth 8                     # the depth-8 sweep
    python exp8_add_scores.py --depth 2 --file main         # the main matrix
    python exp8_add_scores.py --depth 8 --only E7 E2        # one comparison

"""
import argparse
import datetime
import os
import shutil

import common as C
from run_experiments import ABLATION, DEPTH_SWEEP_CONDITIONS, run


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--depth", type=int, required=True)
    ap.add_argument("--file", choices=["sweep", "main"], default="sweep",
                    help="'sweep' rewrites exp4_depth{D}.json, "
                         "'main' rewrites exp3_main_depth{D}.json")
    ap.add_argument("--only", nargs="+", default=None,
                    help="conditions to regenerate; default is all in that file")
    ap.add_argument("--yes", action="store_true",
                    help="skip the confirmation prompt")
    ap.add_argument("--append", action="store_true",
                    help="keep existing conditions and add only the named ones; "
                         "use this to add E6 to a completed depth sweep")
    a = ap.parse_args()

    C.ensure_dirs()
    C.require_gpu(hard=False)   # a warning suffices here: this step is opt-in

    if a.file == "main":
        out_path = f"{C.RESULTS_DIR}/exp3_main_depth{a.depth}.json"
        conditions = a.only or list(ABLATION.keys())
    else:
        out_path = f"{C.RESULTS_DIR}/exp4_depth{a.depth}.json"
        conditions = a.only or DEPTH_SWEEP_CONDITIONS

    score_path = out_path.replace(".json", "_scores.json")

    print("=" * 72)
    print("REGENERATING WITH SCORE PERSISTENCE")
    print("=" * 72)
    print(f"  file       : {os.path.basename(out_path)}")
    print(f"  depth      : {a.depth}")
    print(f"  conditions : {', '.join(conditions)}")
    print(f"  seeds      : {C.SEEDS}")
    print(f"  runs       : {len(conditions) * len(C.SEEDS)}")
    print()
    if os.path.exists(score_path):
        print("  Scores already exist for this file. If they were produced under the")
        print("  current settings, exp7_statistics.py can use them without any re-run.")
        print()
    print("  The metrics in this file WILL be replaced. A re-run does not reproduce")
    print("  the previous numbers exactly, and mixing old metrics with new scores")
    print("  would leave the file internally inconsistent.")
    print("=" * 72)

    if not a.yes:
        reply = input("\nProceed? [y/N] ").strip().lower()
        if reply != "y":
            print("Cancelled. Nothing was changed.")
            return

    # --append adds conditions to a completed file instead of replacing it.
    # Legitimate here because the added condition is trained under the same
    # settings as the existing ones -- the fingerprint is unchanged -- so the
    # file stays internally consistent. Without this, adding one arm would mean
    # retraining every arm.
    if a.append:
        print("\nAppending to the existing file; conditions already present are kept.")
        tuned = C.load("exp2_tuning")
        run(C.subset_path(), conditions, C.SEEDS, a.depth, C.MAX_EPOCHS, C.DEVICE,
            out_path=out_path, resume=True, tuned=tuned,
            edge_updates=C.EDGE_UPDATES, allow_fingerprint_drift=True)
        print(f"\nDONE. metrics -> {out_path}")
        return

    # Archive whatever is there now, so the old numbers remain available.
    if os.path.exists(out_path):
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
        archive = os.path.join(C.RESULTS_DIR, f"_archive_{stamp}")
        os.makedirs(archive, exist_ok=True)
        shutil.copy(out_path, os.path.join(archive, os.path.basename(out_path)))
        print(f"\narchived the previous file -> {archive}")
        os.remove(out_path)          # resume must not skip the conditions

    tuned = C.load("exp2_tuning")
    if tuned:
        C.logger.info("Using tuned hyperparameters from Experiment 2")

    run(C.subset_path(), conditions, C.SEEDS, a.depth, C.MAX_EPOCHS, C.DEVICE,
        out_path=out_path, resume=False, tuned=tuned,
        edge_updates=C.EDGE_UPDATES)

    print("\n" + "=" * 72)
    print("DONE")
    print(f"  metrics -> {out_path}")
    print(f"  scores  -> {score_path}")
    print()
    print("Now run the full analysis, which will no longer skip anything:")
    print(f"  python exp7_statistics.py --depth {a.depth}")
    print("=" * 72)


if __name__ == "__main__":
    main()
