"""
Backward-compatible entrypoint. Prefer ``language_modeling_train.main(RunConfig)`` and
edit hyperparameters in ``utils.RunConfig`` / ``default_hybrid_sweep``.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils import ArchSlot, default_hybrid_sweep
from run_scripts.language_modeling_train_shared import run_from_cli


def build_architectures(args) -> list[ArchSlot]:
    return [
        ArchSlot(
            n_layer=l,
            n_head=h,
            d_model=d,
            dropout=dr,
            lr=lr,
            between_block_mlp_layers=btwmlp,
            layer_norm=not args.noln,
        )
        for l in [1, 2]
        for h in [2, 4]
        for d in [64, 256]
        for dr in [0, 0.1]
        for lr in [1e-3]
        for btwmlp in [2]
    ]


if __name__ == "__main__":
    run_from_cli(default_hybrid_sweep, build_architectures)
