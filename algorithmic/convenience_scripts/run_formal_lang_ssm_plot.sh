#!/usr/bin/env bash

tasks=(
"012_star_0_2_star"
"aa_star"
"aa_star_bb_star"
"abab_star"
"ab_star_d_bc_star"
"an_star_a2"
"d_2"
"d_3"
"d_4"
"d_12"
"tomita_1"
"tomita_2"
"tomita_3"
"tomita_4"
"tomita_5"
"tomita_6"
"tomita_7"

)

for TASK in "${tasks[@]}"; do
    python algorithmic/convenience_scripts/generate_plot_df.py \
        --task "$TASK" \
        --keep arch=ssm,olmossm \
        --group-by arch \
        --output "exports/plots/ssm/formal/${TASK}.svg" \
        --csv exports/summary.csv \
        --legend-loc "lower left" \
        --num-bins 3 \
        --merge-bins
done