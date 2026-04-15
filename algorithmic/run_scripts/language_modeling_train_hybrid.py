"""
Backward-compatible entrypoint. Prefer ``language_modeling_train.main(RunConfig)`` and
edit hyperparameters in ``utils.RunConfig`` / ``default_hybrid_sweep``.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from language_modeling_train import main
from utils import ArchSlot, default_hybrid_sweep


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, choices=["bin_majority", "majority", "bin_majority_interleave", "unique_copy", "repeat_copy", "sort", "parity", "addition", "mqar"])
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--job-id", type=str, default="")
    parser.add_argument("--nope", action="store_true")
    parser.add_argument("--save-final-weights", action="store_true")
    parser.add_argument("--hybrid-layer-pattern", type=str, default="sa")
    parser.add_argument("--monoid", type=str, default="parity", choices=["parity", "cyclic"])
    parser.add_argument("--monoid_n", type=int, default=2)
    parser.add_argument("--key_size", type=int, default=32)
    parser.add_argument("--query_fraction", type=float, default=0.2)
    args = parser.parse_args()
    archs = [
        ArchSlot(n_layer=l, n_head=h, d_model=d, dropout=dr, lr=lr, between_block_mlp_layers=btwmlp, layer_norm=False)
        for l in [2, 8]
        for h in [2, 4]
        for d in [16, 256]
        for dr in [0, 0.1]
        for lr in [1e-3]
        for btwmlp in [2]
    ]
    rc = default_hybrid_sweep()

    rc.architectures = archs
    rc.use_nope = args.nope
    rc.hybrid_layer_pattern = args.hybrid_layer_pattern.strip().lower()
    
    rc.task = args.task
    rc.seeds = args.seeds
    rc.monoid = args.monoid
    rc.monoid_n = args.monoid_n
    rc.key_size = args.key_size
    rc.query_fraction = args.query_fraction
    
    rc.save_final_weights = args.save_final_weights
    rc.job_id = args.job_id
    main(rc)
