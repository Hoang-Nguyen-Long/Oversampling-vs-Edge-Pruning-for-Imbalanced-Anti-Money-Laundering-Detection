"""
run_all.py -- run the entire experimental programme with one command
"""
import argparse
import subprocess
import sys
import time

import common as C

STEPS = [
    (1, "Data understanding",   ["exp1_data_understanding.py"],                 "~5 min"),
    (2, "Tuning, depth 2",      ["exp2_tuning.py", "--depth", "2"],             "~15 min"),
    (3, "Tuning, deep",         ["exp2_tuning.py", "--depth", "DEEP"],          "~15 min"),
    (4, "Tuning, reconcile",    ["exp2_tuning.py", "--combine"],                "seconds"),
    (5, "Main matrix E0-E9",    ["exp3_main_matrix.py"],                        "~1 hour"),
    (6, "Depth sweep",          ["exp4_depth_sweep.py", "--depths", "ALL"],     "~1 h per depth"),
    (7, "Intervention strength",["exp5_intervention_strength.py"],              "~30 min"),
    (8, "Figures and exports",  ["exp6_figures.py"],                            "seconds"),
    (9, "Statistical analysis", ["exp7_statistics.py"],                         "~2 min"),
]


def _resolve(cmd):
    """Substitute the placeholders that depend on the configured depths."""
    out = []
    for token in cmd:
        if token == "DEEP":
            out.append(str(max(C.DEPTHS)))
        elif token == "ALL":
            out.extend(str(d) for d in C.DEPTHS)
        else:
            out.append(token)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from", dest="start", type=int, default=1, help="first step to run")
    ap.add_argument("--only", type=int, default=None, help="run a single step")
    ap.add_argument("--dry-run", action="store_true", help="print the plan and exit")
    a = ap.parse_args()

    steps = [s for s in STEPS if (s[0] == a.only if a.only else s[0] >= a.start)]

    print("=" * 70)
    print("EXPERIMENTAL PROGRAMME")
    print("=" * 70)
    print(f"  depths        : {C.DEPTHS}")
    print(f"  edge updates  : {C.EDGE_UPDATES}")
    print(f"  subset        : {C.SUBSET_FRACTION}")
    print(f"  seeds         : {C.SEEDS}")
    print(f"  max epochs    : {C.MAX_EPOCHS}")
    print(f"  results       : {C.RESULTS_DIR}")
    print("-" * 70)
    for n, name, cmd, est in steps:
        print(f"  step {n}  {name:<24} {est:>16}   {' '.join(_resolve(cmd))}")
    print("=" * 70)
    if a.dry_run:
        return

    t_start = time.perf_counter()
    for n, name, cmd, est in steps:
        print(f"\n{'=' * 70}\nSTEP {n}: {name}\n{'=' * 70}", flush=True)
        t0 = time.perf_counter()
        # A fresh interpreter per step: the CUDA context is destroyed on exit,
        # which fully releases GPU memory before the next step begins.
        r = subprocess.run([sys.executable] + _resolve(cmd))
        mins = (time.perf_counter() - t0) / 60
        if r.returncode != 0:
            print(f"\nSTEP {n} FAILED after {mins:.1f} min (exit {r.returncode}).")
            print(f"Fix the problem, then resume with:  python run_all.py --from {n}")
            sys.exit(r.returncode)
        print(f"\nstep {n} finished in {mins:.1f} min", flush=True)

    print(f"\n{'=' * 70}")
    print(f"ALL STEPS COMPLETE in {(time.perf_counter() - t_start) / 60:.0f} min")
    print(f"Figures and exports: {C.FIGURES_DIR}")
    print("Keep results_long.csv -- every future figure can be rebuilt from it.")


if __name__ == "__main__":
    main()
