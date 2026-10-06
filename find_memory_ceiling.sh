#!/usr/bin/env bash
# =============================================================================
# find_memory_ceiling.sh — determine the largest subset that survives depth 16
#
# Usage:   bash scripts/find_memory_ceiling.sh
#
# Full-batch training holds every edge's activations for every layer at once, so
# memory scales with (edges x depth). Depth 16 on the Multi-GAT backbone is the
# heaviest configuration in the whole project: if it fits, everything fits.
#
# This tries progressively smaller subsets until one succeeds, then tells you the
# value to fix in experiments/common.py. That value must then be used for EVERY
# experiment — if different depths used different amounts of data, a change in
# performance could not be attributed to depth rather than to dataset size, and
# the depth comparison (H2, H4) would be invalid.
# =============================================================================

set -u
source .venv/bin/activate 2>/dev/null || true

FRACTIONS=(0.60 0.50 0.40 0.30 0.20 0.15 0.10)
COMMON="experiments/common.py"
BACKUP="${COMMON}.bak"
cp "${COMMON}" "${BACKUP}"

echo "=============================================================="
echo " Finding the memory ceiling (depth 8, Multi-GAT, condition E8, no edge updates)"
echo "=============================================================="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

for F in "${FRACTIONS[@]}"; do
    echo
    echo "--------------------------------------------------------------"
    echo " Trying SUBSET_FRACTION = ${F}"
    echo "--------------------------------------------------------------"
    # Rewrite the fraction in common.py for this attempt.
    sed -i "s/^SUBSET_FRACTION = .*/SUBSET_FRACTION = ${F}    # set by find_memory_ceiling.sh/" "${COMMON}"

    # A fresh Python process each time, so no memory is carried over between attempts.
    if (cd experiments && python exp4_depth_sweep.py --depths 8 --only E8 --no-edge-updates 2>&1 \
          | tee /tmp/ceiling_attempt.log | tail -5); then
        if grep -qi "out of memory" /tmp/ceiling_attempt.log; then
            echo "  -> OUT OF MEMORY at ${F}"
            continue
        fi
        echo
        echo "=============================================================="
        echo " SUCCESS at SUBSET_FRACTION = ${F}"
        echo
        echo " This value is now set in ${COMMON}."
        echo " Do NOT change it again: every experiment, and every depth, must"
        echo " use the same subset or the depth comparison is confounded."
        echo
        echo " Record the transaction count printed above — it belongs in your"
        echo " methodology."
        echo
        echo " Now delete the probe result so it is not mistaken for a real run:"
        echo "     rm -f \"\$AML_RESULTS/exp4_depth8.json\""
        echo "=============================================================="
        rm -f "${BACKUP}"
        exit 0
    else
        echo "  -> failed at ${F}"
    fi
done

echo
echo "Even the smallest fraction failed. Restoring the original common.py."
cp "${BACKUP}" "${COMMON}"; rm -f "${BACKUP}"
echo "Options: cap the depth sweep at 4, or switch to mini-batch neighbour sampling."
exit 1
