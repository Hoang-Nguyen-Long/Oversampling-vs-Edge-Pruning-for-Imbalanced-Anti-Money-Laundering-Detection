"""
format_kaggle_pandas.py
Pure-pandas re-implementation of the IBM Multi-GNN `format_kaggle_files.py`
(Egressy et al., 2024). It converts a raw Kaggle IBM-AML transactions file
(e.g. HI-Small_Trans.csv) into the `formatted_transactions.csv` schema that this
project's pipeline expects, WITHOUT requiring the `datatable` package (which is
difficult to install on Windows / recent Python).

Output schema (identical to the reference formatter):
    EdgeID, from_id, to_id, Timestamp, Amount Sent, Sent Currency,
    Amount Received, Received Currency, Payment Format, Is Laundering
"""

import os
import sys

import numpy as np
import pandas as pd


def format_file(in_path: str) -> str:
    out_path = os.path.join(os.path.dirname(os.path.abspath(in_path)),
                            "formatted_transactions.csv")
    print(f"Reading raw Kaggle file: {in_path}")
    # Read all columns as strings first (mirrors dt.fread(columns=str32)).
    raw = pd.read_csv(in_path, dtype=str)
    n = len(raw)
    print(f"  {n:,} rows read")

    # The raw IBM-AML column order is:
    #   0 Timestamp | 1 From Bank | 2 Account (sender) | 3 To Bank |
    #   4 Account (receiver) | 5 Amount Received | 6 Receiving Currency |
    #   7 Amount Paid | 8 Payment Currency | 9 Payment Format | 10 Is Laundering
    # The reference formatter accesses the two "Account" columns positionally
    # (indices 2 and 4), which we replicate with .iloc to avoid the duplicate
    # column-name problem.
    ts = pd.to_datetime(raw.iloc[:, 0], format="%Y/%m/%d %H:%M")

    # firstTs: midnight of the first row's date, minus 10 seconds (as reference).
    first = ts.iloc[0]
    start = pd.Timestamp(year=first.year, month=first.month, day=first.day)
    first_ts = start.timestamp() - 10
    ts_rel = ts.map(lambda x: x.timestamp()) - first_ts

    # Accounts: "Bank"+"Account" string, shared id space for sender & receiver.
    from_acc = raw["From Bank"].astype(str) + raw.iloc[:, 2].astype(str)
    to_acc = raw["To Bank"].astype(str) + raw.iloc[:, 4].astype(str)
    acc_codes, _ = pd.factorize(pd.concat([from_acc, to_acc], ignore_index=True))
    from_id = acc_codes[:n]
    to_id = acc_codes[n:]

    # Currencies: receiving and payment share one id space (as reference).
    recv_cur = raw["Receiving Currency"].astype(str)
    pay_cur = raw["Payment Currency"].astype(str)
    cur_codes, _ = pd.factorize(pd.concat([recv_cur, pay_cur], ignore_index=True))
    recv_cur_id = cur_codes[:n]
    pay_cur_id = cur_codes[n:]

    fmt_id, _ = pd.factorize(raw["Payment Format"].astype(str))

    out = pd.DataFrame({
        "EdgeID": np.arange(n, dtype=np.int64),
        "from_id": from_id.astype(np.int64),
        "to_id": to_id.astype(np.int64),
        "Timestamp": ts_rel.astype(np.int64),
        "Amount Sent": raw["Amount Paid"].astype(float),
        "Sent Currency": pay_cur_id.astype(np.int64),
        "Amount Received": raw["Amount Received"].astype(float),
        "Received Currency": recv_cur_id.astype(np.int64),
        "Payment Format": fmt_id.astype(np.int64),
        "Is Laundering": raw["Is Laundering"].astype(int),
    })

    # Sort chronologically (reference sorts on the Timestamp column at the end).
    out = out.sort_values("Timestamp").reset_index(drop=True)
    out.to_csv(out_path, index=False)

    illicit = out["Is Laundering"].mean() * 100
    print(f"Wrote {out_path}")
    print(f"  {len(out):,} transactions | {out['from_id'].max()+1:,} accounts | "
          f"illicit ratio {illicit:.4f}%")
    return out_path


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python format_kaggle_pandas.py /path/to/HI-Small_Trans.csv")
        sys.exit(1)
    format_file(sys.argv[1])
