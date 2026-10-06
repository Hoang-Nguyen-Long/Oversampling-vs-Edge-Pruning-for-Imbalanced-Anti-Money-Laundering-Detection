"""
EXPERIMENT 1 -- Data Understanding  (Chapter 4, CRISP-DM: Data Understanding)
Runs the schema/missingness/duplicate/imbalance/temporal profiling required by
Section 3.4, plus the leakage-safety check on the temporal split. Produces the
report that populates the data-understanding tables of Chapter 4.

Run FIRST: it is cheap, needs no GPU, and if it reveals a problem with the data
every later experiment would inherit that problem.

    python exp1_data_understanding.py
"""
import common as C
from data_understanding import load_transactions, analyse, check_temporal_leakage_safe_split

def main():
    path = C.subset_path()
    df = load_transactions(path)
    report = analyse(df)                      # prints + returns the full profile

    # Leakage check at the 60/20/20 chronological boundaries used everywhere else.
    ts = df["Timestamp"].sort_values().to_numpy()
    n = len(ts)
    leak = check_temporal_leakage_safe_split(df, ts[int(0.6*n)], ts[int(0.8*n)])

    C.save({"profile": report.__dict__, "leakage_check": leak}, "exp1_data_understanding")
    print("\n" + report.summary_text())

if __name__ == "__main__":
    main()
