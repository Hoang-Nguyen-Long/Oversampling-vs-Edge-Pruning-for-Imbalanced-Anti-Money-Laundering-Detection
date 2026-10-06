"""
EXPERIMENT 7 -- Statistical analysis and threshold diagnostics

"""
import argparse
import json

import numpy as np

import common as C
import figures as F
from inductive import account_partition, evaluate_by_partition
from stats_analysis import (load_scores, max_f1, paired_bootstrap, paired_comparison,
                        seed_variability)

DEFAULT_COMPARISONS = [("E7", "E2"), ("E4", "E2"), ("E5", "E2"), ("E9", "E5")]


def _print_paired(c):
    if not c:
        return
    a, b = c["condition_a"], c["condition_b"]
    print(f"\n--- {a} vs {b} ({c['metric']}) ---")
    print(f"  {a:>4}: " + "  ".join(f"{v:.4f}" for v in c["a_values"]))
    print(f"  {b:>4}: " + "  ".join(f"{v:.4f}" for v in c["b_values"]))
    print(f"  diff: " + "  ".join(f"{v:+.4f}" for v in c["differences"]))
    print(f"  mean difference {c['mean_difference']:+.4f} "
          f"(SD {c['sd_difference']:.4f}), "
          f"{c['n_favouring_a']}/{c['n_pairs']} favour {a}")
    print(f"  ratio of means  {c['ratio_of_means']:.2f}x")
    if c.get("paired_t") is not None:
        print(f"  paired t({c['n_pairs'] - 1}) = {c['paired_t']:.3f}, "
              f"p = {c['paired_p']:.4f}")
        print(f"  95% CI of the mean difference "
              f"[{c['ci95_low']:+.4f}, {c['ci95_high']:+.4f}]")
        print(f"  Cohen's d (paired) = {c['cohens_d']:.2f}")
    print(f"  sign test p = {c['sign_test_p']:.4f} "
          f"(floor for n={c['n_pairs']} is {c['sign_test_floor']:.4f})")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--depth", type=int, default=None,
                    help="depth to analyse; default is the deepest available")
    ap.add_argument("--compare", nargs=2, action="append", default=None,
                    metavar=("A", "B"))
    ap.add_argument("--n-boot", type=int, default=2000)
    a = ap.parse_args()

    C.ensure_dirs()
    depth = a.depth if a.depth is not None else max(C.DEPTHS)
    comparisons = [tuple(x) for x in a.compare] if a.compare else DEFAULT_COMPARISONS

    src = f"exp4_depth{depth}"
    results = C.load(src)
    if results is None:
        src = f"exp3_main_depth{depth}"
        results = C.load(src)
    if results is None:
        raise SystemExit(f"No results found for depth {depth} in {C.RESULTS_DIR}")
    results.pop("_config", None)

    report = {"depth": depth, "source": src}

    #  1. seed variability 
    print("=" * 72)
    print(f"SEED VARIABILITY at depth {depth}")
    print("=" * 72)
    conds = sorted(results)
    var = seed_variability(results, conds)
    print(f"{'cond':>5} {'mean':>9} {'SD':>9} {'min':>9} {'max':>9} {'SD/mean':>9} {'max/min':>9}")
    for eid, d in var.items():
        print(f"{eid:>5} {d['mean']:>9.4f} {d['sd']:>9.4f} {d['min']:>9.4f} "
              f"{d['max']:>9.4f} {d['cv']:>9.2f} {d['range_ratio']:>9.2f}")
    report["seed_variability"] = var
    print("\nA SD/mean near or above 1 means run-to-run noise is comparable to the")
    print("quantity being compared, which bounds what any test can establish.")

    #  2. paired comparisons -
    print("\n" + "=" * 72)
    print("PAIRED COMPARISONS ACROSS SEEDS")
    print("=" * 72)
    print("Conditions share seeds, so observations are matched. The differences")
    print("are given in full because with five pairs they can simply be read.")
    report["paired"] = {}
    for ea, eb in comparisons:
        if ea not in results or eb not in results:
            continue
        c = paired_comparison(results, ea, eb)
        _print_paired(c)
        if c:
            report["paired"][f"{ea}_vs_{eb}"] = c
            F.paired_difference_figure(c, C.FIGURES_DIR)

    print("\nNOTE. Five seeds are five training repetitions on one subset, one")
    print("split and one test period. A p-value across them describes run-to-run")
    print("variation of the procedure on this graph. It is not evidence that the")
    print("effect generalises to other graphs or to the full benchmark.")

    #  3. score-based analyses ---
    scores = load_scores(C.RESULTS_DIR)
    if not scores:
        print("\n" + "=" * 72)
        print("No saved prediction scores found; skipping the bootstrap, the")
        print("maximum-F1 analysis and the inductive split. These require a run")
        print("with score persistence enabled.")
        print("=" * 72)
    else:
        print("\n" + "=" * 72)
        print("BOOTSTRAP OVER TEST TRANSACTIONS")
        print("=" * 72)
        report["bootstrap"] = {}
        for ea, eb in comparisons:
            ka, kb = (src, ea), (src, eb)
            if ka not in scores or kb not in scores:
                continue
            sa = scores[ka]["per_seed"][0]; sb = scores[kb]["per_seed"][0]
            if not sa or not sb:
                continue
            boot = paired_bootstrap(sa["test"], sa["test_labels"], sb["test"],
                                    n_boot=a.n_boot)
            if boot:
                print(f"\n--- {ea} vs {eb} (seed 0) ---")
                print(f"  mean difference {boot['mean_difference']:+.4f}")
                print(f"  95% percentile interval "
                      f"[{boot['ci95_low']:+.4f}, {boot['ci95_high']:+.4f}]")
                print(f"  resamples favouring {ea}: "
                      f"{100 * boot['fraction_favouring_a']:.1f}%")
                report["bootstrap"][f"{ea}_vs_{eb}"] = {
                    k: v for k, v in boot.items() if k != "differences"}
                F.bootstrap_figure(boot, C.FIGURES_DIR, label=f"{ea} vs {eb}")

        print("\n" + "=" * 72)
        print("MAXIMUM ATTAINABLE F1")
        print("=" * 72)
        print("The reported F1 uses a threshold chosen on validation for a target")
        print("precision of 0.10. Sweeping the threshold separates ranking quality")
        print("from threshold choice.")
        print(f"\n{'cond':>5} {'reported F1':>13} {'max F1':>9} {'prec@max':>10} "
              f"{'rec@max':>9}")
        report["max_f1"] = {}
        for eid in conds:
            k = (src, eid)
            if k not in scores or not scores[k]["per_seed"][0]:
                continue
            sd = scores[k]["per_seed"][0]
            m = max_f1(sd["test"], sd["test_labels"])
            if not m:
                continue
            rep_f1 = results[eid]["aggregate"].get("f1_mean", float("nan"))
            print(f"{eid:>5} {rep_f1:>13.4f} {m['max_f1']:>9.4f} "
                  f"{m['precision_at_max']:>10.4f} {m['recall_at_max']:>9.4f}")
            report["max_f1"][eid] = {**m, "reported_f1_mean": rep_f1}
            F.threshold_figure(sd["test"], sd["test_labels"], C.FIGURES_DIR,
                               label=f"{eid} depth {depth}")

        #  4. inductive split 
        print("\n" + "=" * 72)
        print("PERFORMANCE BY ACCOUNT HISTORY")
        print("=" * 72)
        print("Accounts first appearing in the test period carry all-zero frozen")
        print("features, so they are harder by construction. Both partitions are")
        print("scored at the same threshold, as a deployed system would.")
        try:
            from data_pipeline import build_graphs
            from data_understanding import load_transactions
            graphs = build_graphs(load_transactions(C.subset_path()))
            unseen = account_partition(graphs)
            print(f"\ntest transactions: {len(unseen):,} | "
                  f"touching an unseen account: {unseen.sum():,} "
                  f"({100 * unseen.mean():.1f}%)")
            report["inductive"] = {}
            for eid in conds:
                k = (src, eid)
                if k not in scores or not scores[k]["per_seed"][0]:
                    continue
                sd = scores[k]["per_seed"][0]
                if len(sd["test"]) != len(unseen):
                    print(f"  {eid}: score length does not match the test split; skipped")
                    continue
                thr = results[eid]["aggregate"].get("threshold_mean")
                part = evaluate_by_partition(sd["test"], sd["test_labels"],
                                             unseen, threshold=thr)
                t, i = part.get("transductive", {}), part.get("inductive", {})
                if t.get("auprc") and i.get("auprc"):
                    print(f"  {eid}: transductive lift {t['auprc_lift']:.1f}x "
                          f"({t['n_positive']} pos) | inductive lift "
                          f"{i['auprc_lift']:.1f}x ({i['n_positive']} pos)")
                report["inductive"][eid] = part
        except Exception as e:
            print(f"  inductive analysis unavailable: {str(e)[:120]}")

    #  5. seed figures ---
    F.seed_comparison_figures(results, C.FIGURES_DIR, depth=depth)

    C.save(report, f"exp7_statistics_depth{depth}")
    print(f"\nSaved -> {C.RESULTS_DIR}/exp7_statistics_depth{depth}.json")
    print(f"Figures -> {C.FIGURES_DIR}")


if __name__ == "__main__":
    main()
