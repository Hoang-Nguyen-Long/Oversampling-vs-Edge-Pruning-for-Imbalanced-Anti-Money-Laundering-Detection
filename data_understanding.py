"""
data_understanding.py
=====================

Data-understanding stage of the CRISP-DM pipeline (Chapter 4, Section B).
The IBM AML transaction schema produced by ``format_kaggle_files.py``
(Altman et al., 2023) is:
    EdgeID, from_id, to_id, Timestamp, Amount Sent, Sent Currency,
    Amount Received, Received Currency, Payment Format, Is Laundering
References
Altman et al. (2023) Realistic synthetic financial transactions for
    anti-money-laundering models. NeurIPS.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Canonical column names in the formatted IBM AML file. Centralised here so a
# schema change only needs editing in one place.
COL_FROM = "from_id"
COL_TO = "to_id"
COL_TIME = "Timestamp"
COL_AMT_SENT = "Amount Sent"
COL_CUR_SENT = "Sent Currency"
COL_AMT_RECV = "Amount Received"
COL_CUR_RECV = "Received Currency"
COL_FORMAT = "Payment Format"
COL_LABEL = "Is Laundering"

# Exact duplication on these keys indicates a repeated *record* rather than a
# legitimate repeat transfer. Legitimate repeats differ in timestamp/amount and
# are kept, because they are valid parallel edges in the multigraph.
_DUP_KEYS = [COL_FROM, COL_TO, COL_TIME, COL_AMT_RECV, COL_LABEL]


@dataclass
class DataUnderstandingReport:
    """Container for every statistic the data-understanding stage produces.

    Kept as a dataclass (rather than a bare dict) so Chapter 4 can pull named
    fields directly and the smoke tests can assert on them.
    """

    n_transactions: int
    n_accounts: int
    n_illicit: int
    illicit_ratio: float
    missing_by_column: dict = field(default_factory=dict)
    n_exact_duplicates: int = 0
    timestamp_min: float = 0.0
    timestamp_max: float = 0.0
    n_days: int = 0
    currencies: dict = field(default_factory=dict)
    payment_formats: dict = field(default_factory=dict)
    self_loops: int = 0
    amount_summary: dict = field(default_factory=dict)

    def summary_text(self) -> str:
        """Human-readable summary used for logging and the Chapter 4 narrative."""
        ir = max(self.illicit_ratio, 1e-12)
        lines = [
            "=" * 62,
            "DATA UNDERSTANDING REPORT",
            "=" * 62,
            f"Transactions (edges)            : {self.n_transactions:,}",
            f"Accounts (nodes)                : {self.n_accounts:,}",
            f"Illicit transactions            : {self.n_illicit:,}",
            f"Illicit ratio                   : {self.illicit_ratio * 100:.4f}%",
            f"Imbalance ratio (licit:illicit) : {(1 - ir) / ir:.0f} : 1",
            f"Exact duplicate records         : {self.n_exact_duplicates:,}",
            f"Self-loop transactions          : {self.self_loops:,}",
            f"Temporal span (days)            : {self.n_days}",
            f"Distinct currencies             : {len(self.currencies)}",
            f"Distinct payment formats        : {len(self.payment_formats)}",
            "-" * 62,
            "Missing values by column:",
        ]
        for col, n in self.missing_by_column.items():
            lines.append(f"    {col:<22}: {n:,}")
        lines.append("=" * 62)
        return "\n".join(lines)


def load_transactions(path: str) -> pd.DataFrame:
    """Load the formatted IBM AML transactions CSV.

    No cleaning is performed here on purpose: the data-understanding stage must
    observe the data *as delivered* by the formatting step so that any anomaly
    is reported rather than silently absorbed.
    """
    logger.info("Loading transactions from %s", path)
    df = pd.read_csv(path)
    logger.info("Loaded %d rows, %d columns", df.shape[0], df.shape[1])
    return df


def analyse(df: pd.DataFrame) -> DataUnderstandingReport:
    """Run every data-understanding check and return a populated report.

    Each block below is labelled with the supervisor requirement it satisfies.
    """
    #Requirement: schema / fraud-label distribution 
    n_transactions = len(df)
    accounts = pd.unique(df[[COL_FROM, COL_TO]].values.ravel())
    n_accounts = int(accounts.shape[0])
    n_illicit = int(df[COL_LABEL].sum())
    illicit_ratio = n_illicit / max(n_transactions, 1)

    #  Requirement: missing-value analysis 
    # Reported per column so the impute/drop/encode decision can be made column
    # by column (a missing currency differs from a missing amount).
    missing_by_column = {k: int(v) for k, v in df.isna().sum().to_dict().items()}

    #  Requirement: duplicate-transaction checks 
    dup_keys = [c for c in _DUP_KEYS if c in df.columns]
    n_exact_duplicates = int(df.duplicated(subset=dup_keys).sum())

    # Self-loops are legal in the schema but distort degree features; report them.
    self_loops = int((df[COL_FROM] == df[COL_TO]).sum())

    #  Requirement: temporal-range analysis 
    tmin = float(df[COL_TIME].min())
    tmax = float(df[COL_TIME].max())
    n_days = int((tmax - tmin) / (3600 * 24) + 1)

    #  Categorical cardinality (informs encoding choices) 
    currencies = df[COL_CUR_RECV].value_counts().to_dict() if COL_CUR_RECV in df else {}
    payment_formats = df[COL_FORMAT].value_counts().to_dict() if COL_FORMAT in df else {}

    #  Amount distribution (informs log-transform decision) 
    amt = df[COL_AMT_RECV].astype(float)
    amount_summary = {
        "min": float(amt.min()),
        "max": float(amt.max()),
        "mean": float(amt.mean()),
        "median": float(amt.median()),
        "std": float(amt.std()),
        # Large positive skew justifies log1p before standardisation (point 3).
        "skew": float(amt.skew()),
    }

    report = DataUnderstandingReport(
        n_transactions=n_transactions,
        n_accounts=n_accounts,
        n_illicit=n_illicit,
        illicit_ratio=illicit_ratio,
        missing_by_column=missing_by_column,
        n_exact_duplicates=n_exact_duplicates,
        timestamp_min=tmin,
        timestamp_max=tmax,
        n_days=n_days,
        currencies=currencies,
        payment_formats=payment_formats,
        self_loops=self_loops,
        amount_summary=amount_summary,
    )
    logger.info("\n%s", report.summary_text())
    return report


def check_temporal_leakage_safe_split(
    df: pd.DataFrame, train_end_time: float, val_end_time: float
) -> dict:
    """Verify a proposed temporal split is strictly ordered and report inductive accounts.

    This is the entity-leakage check from the supervisor's requirements,
    performed on the timestamps that define the split. It confirms the three
    periods are strictly ordered in time and reports how many accounts appear
    for the first time in validation/test (inductive accounts with no
    training-time representation), which are reported separately at evaluation.
    """
    train = df[df[COL_TIME] < train_end_time]
    val = df[(df[COL_TIME] >= train_end_time) & (df[COL_TIME] < val_end_time)]
    test = df[df[COL_TIME] >= val_end_time]

    tr_acc = set(pd.unique(train[[COL_FROM, COL_TO]].values.ravel()))
    va_acc = set(pd.unique(val[[COL_FROM, COL_TO]].values.ravel()))
    te_acc = set(pd.unique(test[[COL_FROM, COL_TO]].values.ravel()))

    result = {
        "train_transactions": len(train),
        "val_transactions": len(val),
        "test_transactions": len(test),
        "strictly_ordered": bool(
            (train[COL_TIME].max() if len(train) else -np.inf)
            < (val[COL_TIME].min() if len(val) else np.inf)
            <= (val[COL_TIME].max() if len(val) else -np.inf)
            < (test[COL_TIME].min() if len(test) else np.inf)
        ),
        "inductive_val_accounts": len(va_acc - tr_acc),
        "inductive_test_accounts": len(te_acc - tr_acc - va_acc),
    }
    logger.info("Temporal split leakage check: %s", result)
    return result


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("csv", help="Path to formatted_transactions.csv")
    a = p.parse_args()
    analyse(load_transactions(a.csv))
