"""Generate a small synthetic IBM-AML-shaped CSV to smoke-test the pipeline."""
import numpy as np, pandas as pd, sys

def make(path, n_edges=6000, n_accounts=400, illicit_ratio=0.02, seed=0):
    rng = np.random.default_rng(seed)
    frm = rng.integers(0, n_accounts, n_edges)
    to = rng.integers(0, n_accounts, n_edges)
    # timestamps spread across ~10 days (seconds)
    ts = np.sort(rng.integers(0, 10*24*3600, n_edges)).astype(float)
    amt = rng.lognormal(mean=6, sigma=1.5, size=n_edges)  # skewed amounts
    cur = rng.integers(0, 3, n_edges)
    fmt = rng.integers(0, 5, n_edges)
    y = (rng.random(n_edges) < illicit_ratio).astype(int)
    df = pd.DataFrame({
        "EdgeID": np.arange(n_edges), "from_id": frm, "to_id": to,
        "Timestamp": ts, "Amount Sent": amt, "Sent Currency": cur,
        "Amount Received": amt, "Received Currency": cur,
        "Payment Format": fmt, "Is Laundering": y,
    })
    df.to_csv(path, index=False)
    print(f"wrote {path}: {n_edges} edges, {y.sum()} illicit ({y.mean()*100:.2f}%)")

if __name__ == "__main__":
    make(sys.argv[1] if len(sys.argv) > 1 else "/tmp/synthetic_aml.csv")
