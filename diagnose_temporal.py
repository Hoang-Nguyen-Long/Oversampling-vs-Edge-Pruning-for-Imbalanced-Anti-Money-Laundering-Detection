"""
diagnose_temporal.py
Diagnostic for the apparent "all fraud is in days 10-17" pattern.

Run this on the FULL formatted file to see what is actually happening:

    python scripts/diagnose_temporal.py /content/drive/MyDrive/aml_project/formatted_transactions.csv

It prints, per day, the transaction COUNT alongside the illicit rate. If the
late days hold only a handful of transactions, a high percentage there is a
small-sample artefact, not a concentration of laundering -- 8 illicit out of 10
transactions is 80%, but it is 8 transactions.
"""
import sys
import pandas as pd

path = sys.argv[1] if len(sys.argv) > 1 else "formatted_transactions.csv"
df = pd.read_csv(path, usecols=["Timestamp", "Is Laundering"])
day = ((df["Timestamp"] - df["Timestamp"].min()) // 86400).astype(int)
g = df.groupby(day)["Is Laundering"].agg(["size", "sum", "mean"])
g.columns = ["transactions", "illicit", "illicit_rate"]
g["illicit_rate_%"] = g["illicit_rate"] * 100
g["% of all txns"] = 100 * g["transactions"] / len(df)
g["% of all illicit"] = 100 * g["illicit"] / max(df["Is Laundering"].sum(), 1)

pd.set_option("display.width", 200)
print(f"\nTotal: {len(df):,} transactions | {int(df['Is Laundering'].sum()):,} illicit "
      f"({df['Is Laundering'].mean()*100:.4f}%)\n")
print(g[["transactions", "illicit", "illicit_rate_%", "% of all txns", "% of all illicit"]]
      .to_string(float_format=lambda v: f"{v:,.4f}"))

bulk = g[g["% of all txns"] >= 1.0]
tail = g[g["% of all txns"] < 1.0]
print(f"\nDays holding >=1% of transactions: {list(bulk.index)}")
print(f"  -> they contain {bulk['% of all illicit'].sum():.2f}% of ALL illicit transactions")
if len(tail):
    print(f"Sparse days (<1% of transactions): {list(tail.index)}")
    print(f"  -> only {tail['transactions'].sum():,} transactions total, "
          f"holding {tail['% of all illicit'].sum():.2f}% of all illicit")
    print("  -> a high PERCENTAGE on these days is a small-sample artefact.")
