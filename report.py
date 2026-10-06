"""
report.py — printing results in a form that can go straight into the write-up
Every function here reads saved JSON and prints. Nothing trains, nothing is
recomputed, and none of it touches the GPU, so these can be run inline in a
notebook and re-run as often as needed.

Values held only in JSON must otherwise be transcribed by hand into the written
report, which risks a figure being copied from one run and left unchanged when
the results are regenerated. Printing formatted blocks directly from the result
files removes that step.
"""
from __future__ import annotations

import json
import os
import platform
import sys

# §5.4.2 — the paired comparison
def paired_block(results_dir, depth=8, comparison="E7_vs_E2"):
    """Print the paired statistics for one comparison, ready to paste."""
    path = os.path.join(results_dir, f"exp7_statistics_depth{depth}.json")
    if not os.path.exists(path):
        print(f"Not found: {path}")
        print("Run exp7_statistics.py first.")
        return None
    rep = json.load(open(path))
    c = rep.get("paired", {}).get(comparison)
    if not c:
        avail = ", ".join(rep.get("paired", {}))
        print(f"No comparison '{comparison}' in the report. Available: {avail}")
        return None

    a, b = c["condition_a"], c["condition_b"]
    print("=" * 74)
    print(f"PAIRED COMPARISON  {a} vs {b}  at depth {depth}")
    print("=" * 74)
    print(f"\n  seed :   " + "  ".join(f"{i:>8}" for i in range(c["n_pairs"])))
    print(f"  {a:<5}:   " + "  ".join(f"{v:>8.4f}" for v in c["a_values"]))
    print(f"  {b:<5}:   " + "  ".join(f"{v:>8.4f}" for v in c["b_values"]))
    print(f"  diff :   " + "  ".join(f"{v:>+8.4f}" for v in c["differences"]))
    print()
    print(f"  mean of {a}          : {c['a_mean']:.4f}")
    print(f"  mean of {b}          : {c['b_mean']:.4f}")
    print(f"  ratio of means       : {c['ratio_of_means']:.2f}x")
    print(f"  mean difference      : {c['mean_difference']:+.4f} "
          f"(SD {c['sd_difference']:.4f})")
    if c.get("ci95_low") is not None:
        print(f"  95% CI of difference : [{c['ci95_low']:+.4f}, {c['ci95_high']:+.4f}]")
        print(f"  paired t({c['n_pairs'] - 1})           : {c['paired_t']:.3f}, "
              f"p = {c['paired_p']:.4f}")
        print(f"  Cohen's d (paired)   : {c['cohens_d']:.2f}")
    print(f"  seeds favouring {a}   : {c['n_favouring_a']} of {c['n_pairs']}")
    print(f"  sign test p          : {c['sign_test_p']:.4f} "
          f"(minimum attainable for n={c['n_pairs']}: {c['sign_test_floor']:.4f})")

    print("\n" + "-" * 74)
    print("PASTE-READY SENTENCE FOR §5.4.2")
    print("-" * 74)
    diffs = ", ".join(f"{v:+.4f}" for v in c["differences"])
    sig = ("does not include zero" if c.get("ci95_low", 0) > 0
           or (c.get("ci95_high") or 0) < 0 else "includes zero")
    print(f"""
Across the five matched seeds the paired differences in AUPRC ({a} minus {b})
were {diffs}, giving a mean difference of {c['mean_difference']:+.4f}
(SD {c['sd_difference']:.4f}) and a 95 per cent confidence interval of
[{c['ci95_low']:+.4f}, {c['ci95_high']:+.4f}], which {sig}. {c['n_favouring_a']} of the
{c['n_pairs']} seeds favoured {a}. A paired t-test gives t({c['n_pairs'] - 1}) =
{c['paired_t']:.3f}, p = {c['paired_p']:.4f}, with a paired Cohen's d of
{c['cohens_d']:.2f}. The conditions share seeds, so the observations are matched
and a paired test is the appropriate one; a two-sample test would discard that
pairing. The sign test returns p = {c['sign_test_p']:.4f}, but with five pairs its
smallest attainable two-sided value is {c['sign_test_floor']:.4f}, so it cannot reach
conventional significance however large the effect. These figures describe
run-to-run variation of the training procedure on this subset and this split;
they are not evidence that the effect generalises to other graphs or to the
complete benchmark.
""".strip())
    return c


# §4.3.2 — split prevalences
def split_prevalence_block(results_dir):
    """Print the exact per-split prevalences from the data-understanding run."""
    path = os.path.join(results_dir, "exp1_data_understanding.json")
    if not os.path.exists(path):
        print(f"Not found: {path}\nRun exp1_data_understanding.py first.")
        return None
    d = json.load(open(path))

    def _dig(obj, *names):
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k in names:
                    return v
                r = _dig(v, *names)
                if r is not None:
                    return r
        return None

    splits = _dig(d, "splits", "split_prevalence", "split_stats")
    print("=" * 74)
    print("SPLIT PREVALENCES  (from exp1_data_understanding.json)")
    print("=" * 74)
    if not splits:
        print("\nNo split section found. Top-level keys:")
        print("  " + ", ".join(d))
        return d

    print(f"\n{'split':>8} {'transactions':>14} {'illicit':>10} {'prevalence':>12}")
    rows = []
    for name in ("train", "val", "test"):
        s = splits.get(name) if isinstance(splits, dict) else None
        if not isinstance(s, dict):
            continue
        n = s.get("n_edges") or s.get("n") or s.get("edges")
        pos = s.get("n_illicit") or s.get("illicit") or s.get("n_positive")
        prev = s.get("illicit_rate") or s.get("prevalence")
        if prev is not None and prev > 1:
            prev = prev / 100.0
        rows.append((name, n, pos, prev))
        print(f"{name:>8} {n:>14,} {pos:>10,} {100 * prev:>11.4f}%")

    if len(rows) == 3 and all(r[3] for r in rows):
        tr, va, te = (r[3] for r in rows)
        print(f"\n  test / train prevalence ratio: {te / tr:.2f}x")
        print("\n" + "-" * 74)
        print("PASTE-READY SENTENCE FOR §4.3.2")
        print("-" * 74)
        print(f"""
Illicit prevalence differs across the three periods: {100 * tr:.4f} per cent in
training, {100 * va:.4f} per cent in validation and {100 * te:.4f} per cent in test,
so the test period is approximately {te / tr:.1f} times more concentrated than the
training period. The splits are therefore not equivalent in difficulty, and this
is the reason AUPRC is reported alongside its prevalence baseline throughout:
a raw AUPRC is not comparable across periods with different positive rates,
whereas lift relative to prevalence is.
""".strip())
    return splits


# §5.x — maximum attainable F1
def max_f1_block(results_dir, depth=8):
    """Print reported F1 against the best attainable at any threshold."""
    path = os.path.join(results_dir, f"exp7_statistics_depth{depth}.json")
    if not os.path.exists(path):
        print(f"Not found: {path}\nRun exp7_statistics.py first.")
        return None
    rep = json.load(open(path))
    mf = rep.get("max_f1")
    print("=" * 74)
    print(f"MAXIMUM ATTAINABLE F1  at depth {depth}")
    print("=" * 74)
    if not mf:
        print("\nNo max-F1 section: the run had no saved prediction scores.")
        print("Regenerate with:  python exp8_add_scores.py --depth %d" % depth)
        return None

    print(f"\n{'cond':>6} {'reported F1':>13} {'max F1':>9} {'ratio':>8} "
          f"{'prec@max':>10} {'rec@max':>9} {'thr@max':>9}")
    for eid, m in sorted(mf.items()):
        rep_f1 = m.get("reported_f1_mean", float("nan"))
        ratio = m["max_f1"] / rep_f1 if rep_f1 else float("nan")
        print(f"{eid:>6} {rep_f1:>13.4f} {m['max_f1']:>9.4f} {ratio:>8.2f}x "
              f"{m['precision_at_max']:>10.4f} {m['recall_at_max']:>9.4f} "
              f"{m['threshold_at_max']:>9.4f}")

    best = max(mf.items(), key=lambda kv: kv[1]["max_f1"])
    print("\n" + "-" * 74)
    print("PASTE-READY SENTENCE")
    print("-" * 74)
    print(f"""
The F1 scores reported in Table 4.3 are computed at an operating threshold chosen
on validation to reach a target precision of 0.10, so a low value may reflect
either a weak ranking or a threshold selected for a different objective. Sweeping
the threshold separates the two. The highest F1 attainable by any threshold was
{best[1]['max_f1']:.4f} for {best[0]}, at precision {best[1]['precision_at_max']:.4f} and
recall {best[1]['recall_at_max']:.4f}. The gap between reported and maximum F1
indicates how much of the low reported value is attributable to threshold choice
rather than to ranking quality, and it does not alter the ranking of conditions
by AUPRC, which is threshold-independent.
""".strip())
    return mf


# §5.x — performance by account history
def inductive_block(results_dir, depth=8):
    """Print detection performance split by whether accounts were seen in training."""
    path = os.path.join(results_dir, f"exp7_statistics_depth{depth}.json")
    if not os.path.exists(path):
        print(f"Not found: {path}\nRun exp7_statistics.py first.")
        return None
    rep = json.load(open(path))
    ind = rep.get("inductive")
    print("=" * 74)
    print(f"PERFORMANCE BY ACCOUNT HISTORY  at depth {depth}")
    print("=" * 74)
    if not ind:
        print("\nNo inductive section: the run had no saved prediction scores.")
        print("Regenerate with:  python exp8_add_scores.py --depth %d" % depth)
        return None

    print(f"\n{'cond':>6} {'partition':>14} {'n':>9} {'pos':>6} {'prev':>9} "
          f"{'AUPRC':>9} {'lift':>8}")
    any_inductive = False
    for eid, part in sorted(ind.items()):
        for name in ("transductive", "inductive"):
            p = part.get(name)
            if not isinstance(p, dict) or p.get("auprc") is None:
                continue
            if name == "inductive":
                any_inductive = True
            print(f"{eid:>6} {name:>14} {p['n']:>9,} {p['n_positive']:>6} "
                  f"{100 * p['prevalence']:>8.4f}% {p['auprc']:>9.4f} "
                  f"{p['auprc_lift']:>7.2f}x")

    if not any_inductive:
        print("\nNo test transaction touched an account absent from training, so the")
        print("inductive partition is empty. That is itself reportable: under this")
        print("split every test account had a training history, and the frozen-feature")
        print("limitation described in §3.4.3 did not bind on this data.")
        return ind

    print("\n" + "-" * 74)
    print("PASTE-READY SENTENCE")
    print("-" * 74)
    ratios = [p.get("lift_ratio_transductive_over_inductive")
              for p in ind.values() if p.get("lift_ratio_transductive_over_inductive")]
    mean_ratio = sum(ratios) / len(ratios) if ratios else None
    tail = (f" Averaged across conditions, lift on transactions with a full account "
            f"history was {mean_ratio:.1f} times that on transactions touching an "
            f"account first seen in the test period." if mean_ratio else "")
    print(f"""
Section 3.4.3 records that node features are frozen from the training period, so
an account first appearing in the test period carries all-zero aggregates. Those
transactions are therefore harder by construction, and averaging over both groups
conceals the difference. Partitioning the test set by whether either endpoint had
a training history shows the effect directly.{tail} Both partitions are scored at
the same threshold, since a deployed system applies one threshold to all traffic;
using a partition-specific threshold would flatter the harder group and describe
a system nobody would build. This matters operationally because newly created
accounts are exactly where mule activity is placed.
""".strip())
    return ind


# §4.4.6 — environment
def environment_block():
    """Print the library and hardware versions for the reproducibility section."""
    print("=" * 74)
    print("ENVIRONMENT  (for §4.4.6)")
    print("=" * 74)
    rows = [("Python", platform.python_version()),
            ("Platform", platform.platform())]
    try:
        import torch
        rows.append(("PyTorch", torch.__version__))
        rows.append(("CUDA (torch build)", torch.version.cuda or "n/a"))
        rows.append(("cuDNN", str(torch.backends.cudnn.version())))
        if torch.cuda.is_available():
            rows.append(("GPU", torch.cuda.get_device_name(0)))
            rows.append(("GPU memory (GiB)",
                         f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}"))
            cap = torch.cuda.get_device_capability(0)
            rows.append(("Compute capability", f"{cap[0]}.{cap[1]}"))
    except Exception:
        pass
    for name, mod in (("PyTorch Geometric", "torch_geometric"), ("NumPy", "numpy"),
                      ("pandas", "pandas"), ("scikit-learn", "sklearn"),
                      ("SciPy", "scipy"), ("Matplotlib", "matplotlib")):
        try:
            rows.append((name, __import__(mod).__version__))
        except Exception:
            rows.append((name, "not installed"))
    try:
        import subprocess
        drv = subprocess.run(["nvidia-smi",
                              "--query-gpu=driver_version", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=15)
        if drv.returncode == 0 and drv.stdout.strip():
            rows.append(("NVIDIA driver", drv.stdout.strip().splitlines()[0]))
    except Exception:
        pass

    for k, v in rows:
        print(f"  {k:<22} {v}")

    print("\n" + "-" * 74)
    print("PASTE-READY SENTENCE FOR §4.4.6")
    print("-" * 74)
    g = dict(rows)
    print(f"""
All experiments were executed on a single {g.get('GPU', 'GPU')} with
{g.get('GPU memory (GiB)', 'n/a')} GiB of memory, under Python
{g.get('Python', 'n/a')}, PyTorch {g.get('PyTorch', 'n/a')} built against CUDA
{g.get('CUDA (torch build)', 'n/a')}, with PyTorch Geometric
{g.get('PyTorch Geometric', 'n/a')}, NumPy {g.get('NumPy', 'n/a')}, pandas
{g.get('pandas', 'n/a')}, scikit-learn {g.get('scikit-learn', 'n/a')} and SciPy
{g.get('SciPy', 'n/a')}. Random seeds were fixed for parameter initialisation and
for intervention sampling, and the GPU model is recorded with every result file so
that runs from different hardware cannot be combined. Bitwise reproducibility is
not claimed: floating-point reduction order on the GPU is not deterministic, so
repeated execution of an identical configuration produces small variation in the
third or fourth decimal place.
""".strip())
    return dict(rows)


def full_report(results_dir, depth=8, comparisons=("E7_vs_E2",)):
    """Print every block in the order the write-up needs them."""
    split_prevalence_block(results_dir); print("\n")
    for comp in comparisons:
        paired_block(results_dir, depth, comp); print("\n")
    max_f1_block(results_dir, depth); print("\n")
    inductive_block(results_dir, depth); print("\n")
    environment_block()
