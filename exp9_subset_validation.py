"""
EXPERIMENT 9 -- Is the 30% subset representative of the full dataset?
(Chapter 4, Section 4.3.1)
Chapter 4 asserts that proportional subsampling preserves the structure of the
data. This tests it, across the three property groups a graph model depends on:
transaction attributes, graph topology, and temporal and class structure.
"""
import argparse

import common as C
import figures as F
from data_understanding import load_transactions
from subset_validation import compare, paste_ready


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fraction", type=float, default=None,
                    help="subset fraction to validate; defaults to the configured one")
    ap.add_argument("--full-path", default=None,
                    help="path to the full dataset; defaults to AML_DATA")
    a = ap.parse_args()

    C.ensure_dirs()
    frac = a.fraction if a.fraction is not None else C.SUBSET_FRACTION
    if not frac or frac >= 1.0:
        raise SystemExit("No subset is in use, so there is nothing to validate. "
                         "Set AML_SUBSET_FRACTION or pass --fraction.")

    full_path = a.full_path or C.DATA_PATH
    C.logger.info("Loading the full dataset from %s", full_path)
    full_df = load_transactions(full_path)

    C.logger.info("Loading the %.0f%% subset", 100 * frac)
    subset_df = load_transactions(C.subset_path())

    out = compare(subset_df, full_df, verbose=True)
    paste_ready(out)

    C.logger.info("Generating figures")
    F.subset_validation_figures(subset_df, full_df, C.FIGURES_DIR)

    C.save(out, "exp9_subset_validation")
    print(f"\nSaved -> {C.RESULTS_DIR}/exp9_subset_validation.json")
    print(f"Figures 4.16-4.18 -> {C.FIGURES_DIR}")


if __name__ == "__main__":
    main()
