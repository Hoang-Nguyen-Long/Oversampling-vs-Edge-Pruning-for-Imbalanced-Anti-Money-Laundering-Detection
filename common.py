"""
common.py
=========
Shared setup for every experiment script (Chapter 4).
"""

from __future__ import annotations

import json, logging, os, sys, tempfile

# Reduce CUDA memory fragmentation. Full-batch GNN training allocates large,
# irregularly sized tensors, which leaves the allocator holding blocks it cannot
# reuse -- the "reserved but unallocated" memory reported in out-of-memory
# errors. Expandable segments let the allocator grow existing blocks instead,
# which typically recovers a few hundred megabytes. This must be set BEFORE
# torch is imported, so it sits above the torch import deliberately.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import pandas as pd
import torch

# make src/ importable regardless of where the script is launched from
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "src"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("aml")

# CONFIGURATION -- edit these once; every experiment inherits them.
DATA_PATH = os.environ.get("AML_DATA", "/content/formatted_transactions.csv")
RESULTS_DIR = os.environ.get("AML_RESULTS", "/content/drive/MyDrive/aml_project/results")
FIGURES_DIR = os.environ.get("AML_FIGURES", "/content/drive/MyDrive/aml_project/figures")

# Memory dial. SUBSET_FRACTION keeps every day but thins each one, so the day
# count (and hence the temporal split) survives while memory drops. Lower it if
# depth 16 runs out of memory; then keep the value FIXED for all experiments.
# These may be overridden by environment variables, which is the ROBUST way to
# change them: an environment variable survives the project being re-extracted
# from the zip, and propagates automatically to every subprocess. Editing this
# file works too, but the edit is lost if the zip is unpacked again afterwards.
#
#     os.environ["AML_SUBSET_FRACTION"] = "0.2"     # in the notebook
#     export AML_SUBSET_FRACTION=0.2                # on the command line
#
def _env_float(name, default):
    v = os.environ.get(name)
    if v is None or v.strip().lower() in ("", "none"):
        return default
    return float(v)


def _env_int_list(name, default):
    v = os.environ.get(name)
    if not v:
        return default
    return [int(x) for x in v.replace(",", " ").split()]


def _env_bool(name, default):
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


SUBSET_DAYS = None        # e.g. 4, or None for the whole span
SUBSET_FRACTION = _env_float("AML_SUBSET_FRACTION", 0.30)

# Per-layer edge-update MLP. Applied UNIFORMLY to every condition (never tied to
# the backbone), for two reasons:
#   Fairness -- if it were on for Multi-GAT and off for GAT, the backbones would
#     differ in two respects at once and the comparison would be confounded.
#   Memory   -- the edge MLP concatenates [x_src || x_dst || edge_attr], a tensor
#     of width 3*hidden at EVERY layer, roughly tripling activation memory. It is
#     the main reason deep runs exhaust the GPU.
# False allows a deeper sweep; True is closer to the reference implementation.
# The same value must be used for every experiment under comparison.
EDGE_UPDATES = _env_bool("AML_EDGE_UPDATES", False)

SEEDS = _env_int_list("AML_SEEDS", [0, 1, 2, 3, 4])
# Section 3.9.3: five seeds per condition, reported as mean +/- standard
# deviation. Five rather than three because the dispersion across seeds is the
# criterion by which a difference is judged to be an effect: with three runs the
# standard deviation is itself estimated from very few observations, so a
# borderline comparison cannot be resolved. Five does not make the estimate
# precise, but it materially reduces the chance that an apparent ordering is an
# artefact of which three initialisations happened to be drawn.
REFERENCE_DEPTH = 2       # depth used by the main matrix (Experiment 3)
DEPTHS = _env_int_list("AML_DEPTHS", [2, 4, 8])
                          # Section 3.9.2: the pre-specified depth sweep.
                          # Capped at 8 by the memory available under full-batch
                          # training; depth 16 is reported as beyond the ceiling.
MAX_EPOCHS = int(_env_float("AML_MAX_EPOCHS", 100))  # early stopping at patience 20

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def ensure_dirs():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)


def subset_path():
    """Return a path to the (possibly subsetted) transactions CSV.

    Applies SUBSET_DAYS then SUBSET_FRACTION. Caches the result so repeated
    experiment scripts in the same session do not re-derive it.
    """
    if SUBSET_DAYS is None and SUBSET_FRACTION is None:
        return DATA_PATH

    cache = os.path.join(tempfile.gettempdir(),
                         f"aml_subset_d{SUBSET_DAYS}_f{SUBSET_FRACTION}.csv")
    if os.path.exists(cache):
        logger.info("Reusing cached subset: %s", cache)
        return cache

    df = pd.read_csv(DATA_PATH).sort_values("Timestamp").reset_index(drop=True)
    t0 = df["Timestamp"].min()
    if SUBSET_DAYS is not None:
        df = df[df["Timestamp"] < t0 + SUBSET_DAYS * 86400]
    if SUBSET_FRACTION is not None:
        day = ((df["Timestamp"] - t0) // 86400).astype(int)
        df = (df.groupby(day, group_keys=False)
                .apply(lambda g: g.head(max(1, int(len(g) * SUBSET_FRACTION))))
                .sort_values("Timestamp").reset_index(drop=True))
    df.to_csv(cache, index=False)
    span = int((df["Timestamp"].max() - df["Timestamp"].min()) / 86400) + 1
    logger.info("Subset: %d transactions over ~%d day(s), %.4f%% illicit -> %s",
                len(df), span, df["Is Laundering"].mean() * 100, cache)
    return cache


def save(obj, name):
    """Write a results object to RESULTS_DIR/name.json."""
    ensure_dirs()
    path = os.path.join(RESULTS_DIR, f"{name}.json")
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
    logger.info("Saved -> %s", path)
    return path


def load(name):
    path = os.path.join(RESULTS_DIR, f"{name}.json")
    return json.load(open(path)) if os.path.exists(path) else None


def free_gpu():
    """Release cached GPU memory. Call between depths or after a failed run.

    Full-batch training at depth is memory-bound, and PyTorch keeps freed blocks
    in its caching allocator. Emptying the cache makes that memory available
    again, which matters when the next configuration is deeper than the last.
    """
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        free, total = torch.cuda.mem_get_info()
        logger.info("GPU memory: %.2f GiB free of %.2f GiB",
                    free / 1024**3, total / 1024**3)


def estimate_memory(n_edges, depth, hidden=64, edge_updates=False,
                    multi_gat=False, fp16=False):
    """Rough activation-memory estimate in GiB, to predict what will fit.

    Counts only the dominant per-layer activations; parameters, optimiser state
    and allocator fragmentation add more, so treat the figure as a lower bound.
    The backward pass roughly doubles the forward cost, which is included.
    """
    bytes_per = 2 if fp16 else 4
    width = 3 * hidden if edge_updates else hidden   # edge MLP concatenation
    edges = n_edges * 2 if multi_gat else n_edges    # reverse message passing
    forward = edges * width * bytes_per * depth / 1024 ** 3
    return forward * 2                               # + backward


def memory_report(n_edges, depths=(2, 4, 8, 16), hidden=64, budget_gib=24.0):
    """Print a table of estimated memory for each depth and configuration."""
    print(f"Estimated activation memory (GiB), {n_edges:,} edges, budget {budget_gib} GiB")
    print(f"{'config':<34}" + "".join(f"{f'L={d}':>10}" for d in depths))
    for name, eu, mg, fp16 in [
        ("GAT, no edge updates",            False, False, False),
        ("GAT, edge updates",               True,  False, False),
        ("Multi-GAT, no edge updates",      False, True,  False),
        ("Multi-GAT, edge updates",         True,  True,  False),
        ("Multi-GAT, no edge updates, fp16", False, True, True),
    ]:
        row = f"{name:<34}"
        for d in depths:
            g = estimate_memory(n_edges, d, hidden, eu, mg, fp16)
            row += f"{g:>9.1f}{'*' if g > budget_gib else ' '}"
        print(row)
    print("* exceeds the budget")


# GPU visibility helpers
def gpu_status(label=""):
    """Print what the GPU currently holds, and return (free_gib, total_gib).

    Useful before and after each step. Note that the NOTEBOOK KERNEL itself holds
    a CUDA context of a few hundred megabytes once torch has touched the GPU;
    that is normal and unavoidable while the kernel is alive. What matters is
    that the large per-run allocations are released, which they are when each
    step runs as its own process.
    """
    if not torch.cuda.is_available():
        print(f"{label}no GPU visible")
        return (0.0, 0.0)
    free_b, total_b = torch.cuda.mem_get_info()
    free, total = free_b / 1024 ** 3, total_b / 1024 ** 3
    used = total - free
    reserved = torch.cuda.memory_reserved() / 1024 ** 3
    allocated = torch.cuda.memory_allocated() / 1024 ** 3
    bar_len = 30
    filled = int(bar_len * used / total)
    print(f"{label}GPU  [{'#' * filled}{'.' * (bar_len - filled)}]  "
          f"{used:5.2f} / {total:5.2f} GiB used, {free:5.2f} GiB free")
    if reserved or allocated:
        print(f"{' ' * len(label)}     this process: {allocated:.2f} GiB allocated, "
              f"{reserved:.2f} GiB reserved")
    return (free, total)


def clear_gpu(label=""):
    """Release everything this process can, then report the result.

    A caveat worth understanding: this cannot free memory held by a DIFFERENT
    process, and it cannot defragment. If a training step is run as its own
    process, as run_all.py invokes them, its memory is returned in full on exit.
    That is a stronger guarantee than this function can provide from within a
    live interpreter.
    """
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    return gpu_status(label)


REQUIRED_GPU = os.environ.get("AML_REQUIRE_GPU", "L4")


def require_gpu(name_fragment=None, hard=True):
    """Refuse to proceed unless the attached GPU matches what the study used.

    A hosted runtime allocates whatever accelerator is free, so a session may
    begin on a smaller card. Two consequences follow: the heaviest depth may
    exhaust memory partway through a sweep, and results become a mixture of
    devices, which is not the "identical conditions" the methodology claims.

    Called at the top of every run, this converts a silent inconsistency into an
    immediate, obvious stop.
    """
    name_fragment = name_fragment or REQUIRED_GPU
    if not torch.cuda.is_available():
        msg = "No GPU attached. Runtime > Change runtime type > GPU."
        if hard:
            raise SystemExit(msg)
        logger.warning(msg); return False
    actual = torch.cuda.get_device_name(0)
    total = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
    if name_fragment.lower() not in actual.lower():
        msg = (f"Attached GPU is {actual} ({total:.1f} GiB), but this study requires "
               f"{name_fragment}. Results from different devices must not be mixed. "
               f"Restart the runtime until {name_fragment} is allocated, or set "
               f"AML_REQUIRE_GPU to the device used for every run.")
        if hard:
            raise SystemExit(msg)
        logger.warning(msg); return False
    logger.info("GPU check passed: %s (%.1f GiB)", actual, total)
    return True

