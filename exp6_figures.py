"""
EXPERIMENT 6 -- Figures, Tables and Result Export  (Chapter 4/5 deliverables)
"""
import argparse

import common as C
from figures import (data_understanding_figures, matrix_figures, depth_figures,
                     strength_figures, results_tables, efficiency_figures,
                     operational_figures, master_comparison)
from results_export import export_all


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export-only", action="store_true",
                    help="write the CSV exports and skip the figures")
    ap.add_argument("--depth", type=int, default=None,
                    help="depth for the single-depth figures (default: reference depth)")
    a = ap.parse_args()
    depth = a.depth if a.depth is not None else C.REFERENCE_DEPTH

    C.ensure_dirs()

    #Master exports first: these are the durable record 
    print("=" * 64)
    print("EXPORTING RAW RESULTS")
    print("=" * 64)
    export_all(C.RESULTS_DIR, C.FIGURES_DIR)

    if a.export_only:
        print(f"\nExports written to {C.FIGURES_DIR}. Skipping figures.")
        return

    print("\n" + "=" * 64)
    print("GENERATING FIGURES")
    print("=" * 64)
    data_understanding_figures(C.subset_path(), C.FIGURES_DIR)   # 4.1-4.5
    matrix_figures(C.RESULTS_DIR, C.FIGURES_DIR, depth=depth)    # 5.1-5.4
    depth_figures(C.RESULTS_DIR, C.FIGURES_DIR, depths=C.DEPTHS) # 5.5-5.8
    strength_figures(C.RESULTS_DIR, C.FIGURES_DIR)               # 5.9-5.10
    efficiency_figures(C.RESULTS_DIR, C.FIGURES_DIR, depth=depth)   # 5.11-5.13
    operational_figures(C.RESULTS_DIR, C.FIGURES_DIR, depth=depth)  # 5.14-5.15
    master_comparison(C.RESULTS_DIR, C.FIGURES_DIR, depth=depth)    # 5.16

    print("\n" + "=" * 64)
    print("TABLES")
    print("=" * 64)
    results_tables(C.RESULTS_DIR, C.FIGURES_DIR, depths=C.DEPTHS,
                   reference_depth=depth)

    print(f"\nAll outputs -> {C.FIGURES_DIR}")
    print("Keep results_long.csv: every future figure can be built from it.")


if __name__ == "__main__":
    main()
