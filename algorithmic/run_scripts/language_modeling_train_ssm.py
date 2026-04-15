"""
Backward-compatible entrypoint. Prefer ``language_modeling_train.main(RunConfig)`` and
edit hyperparameters in ``utils.RunConfig`` / ``default_ssm_sweep``.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from language_modeling_train import main
from utils import default_ssm_sweep, ArchSlot


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, choices=["bin_majority", "majority", "bin_majority_interleave", "unique_copy", "repeat_copy", "sort", "parity", "addition", "mqar"])
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--job-id", type=str, default="")
    parser.add_argument("--save-final-weights", action="store_true")
    parser.add_argument("--train-steps", type=int, default=None)
    parser.add_argument("--warmup-steps", type=int, default=None)
    parser.add_argument("--monoid", type=str, default="parity", choices=["parity", "cyclic"])
    parser.add_argument("--monoid_n", type=int, default=2)
    parser.add_argument("--key_size", type=int, default=32)
    parser.add_argument("--query_fraction", type=float, default=0.2)
    args = parser.parse_args()
    archs = [
        ArchSlot(n_layer=l, d_model=d, dropout=dr, lr=lr, between_block_mlp_layers=btwmlp)
        for l in [4, 8]
        for d in [16, 256]
        for dr in [0, 0.1]
        for btwmlp in [2, 4]
        for lr in [1e-3]
    ]
    rc = default_ssm_sweep()

    rc.architectures = archs
    # rc.train_length_range = (0, 25)
    # rc.num_test_bins = 6
    rc.save_final_weights = args.save_final_weights
    rc.task = args.task
    rc.seeds = args.seeds
    rc.monoid = args.monoid
    rc.monoid_n = args.monoid_n
    rc.key_size = args.key_size
    rc.query_fraction = args.query_fraction

    if args.train_steps is not None:
        rc.max_steps_default = args.train_steps
        rc.max_steps_large = args.train_steps
    if args.warmup_steps is not None:
        rc.warmup_default = args.warmup_steps
        rc.warmup_large = args.warmup_steps
    rc.job_id = args.job_id
    main(rc)
