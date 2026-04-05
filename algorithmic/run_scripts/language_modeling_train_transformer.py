"""
Backward-compatible entrypoint. Prefer ``language_modeling_train.main(RunConfig)`` and
edit hyperparameters in ``utils.RunConfig`` / ``default_transformer_sweep``.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from language_modeling_train import main
from utils import default_transformer_sweep


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
    rc = default_transformer_sweep()
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
