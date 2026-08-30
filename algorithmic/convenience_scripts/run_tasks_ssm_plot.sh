#!/usr/bin/env bash

tasks=(
    "addition"
    "bin_majority_interleave"
    "bin_majority"
    "flipflop"
    "majority"
    "mkar"
    "mqar"
    "parity"
    "repeat_copy"
    "selective_copy"
    "sort"
    "unique_copy"
)

for TASK in "${tasks[@]}"; do
    python algorithmic/convenience_scripts/generate_plot_df.py \
        --task "$TASK" \
        --keep arch=ssm,olmossm \
        --group-by arch \
        --output "exports/plots/ssm/tasks/${TASK}.svg" \
        --csv exports/summary.csv \
        --legend-loc "lower left" \
        --num-bins 3 \
        --merge-bins
done