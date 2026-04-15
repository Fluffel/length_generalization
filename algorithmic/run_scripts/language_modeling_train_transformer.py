"""
Backward-compatible entrypoint. Prefer ``language_modeling_train.main(RunConfig)`` and
edit hyperparameters in ``utils.RunConfig`` / ``default_transformer_sweep``.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from language_modeling_train import main
from utils import ArchSlot, default_transformer_sweep


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, choices=["bin_majority", "majority", "bin_majority_interleave", "unique_copy", "repeat_copy", "sort", "parity", "addition", "mqar"])
    parser.add_argument("--nope", action="store_true")
    parser.add_argument("--regularize", type=float, default=0.0)
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--job-id", type=str, default="")
    parser.add_argument("--monoid", type=str, default="parity", choices=["parity", "cyclic"])
    parser.add_argument("--monoid_n", type=int, default=2)
    parser.add_argument("--key_size", type=int, default=32)
    parser.add_argument("--query_fraction", type=float, default=0.2)
    args = parser.parse_args()
    archs = [
        ArchSlot(n_layer=l, n_head=h, d_model=d, lr=lr, between_block_mlp_layers=btwmlp, dropout=dr)
        for l in [2, 4]
        for h in [2, 4]
        for d in [64, 256]
        for dr in [0, 0.1]
        for btwmlp in [1, 2]
        for lr in [1e-3, 1e-4]
    ]
    
    rc = default_transformer_sweep()
    rc.architectures = archs
    rc.train_length_range = (0, 25)
    rc.num_test_bins = 6


    rc.task = args.task
    rc.use_nope = args.nope
    rc.regularize = args.regularize
    rc.seeds = args.seeds
    rc.job_id = args.job_id
    rc.monoid = args.monoid
    rc.monoid_n = args.monoid_n
    rc.key_size = args.key_size
    rc.query_fraction = args.query_fraction
    main(rc)
