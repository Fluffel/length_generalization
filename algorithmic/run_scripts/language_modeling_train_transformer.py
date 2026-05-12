"""
Backward-compatible entrypoint. Prefer ``language_modeling_train.main(RunConfig)`` and
edit hyperparameters in ``utils.RunConfig`` / ``default_transformer_sweep``.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils import ArchSlot, default_transformer_sweep
from run_scripts.language_modeling_train_shared import run_from_cli


def build_architectures(_args) -> list[ArchSlot]:
    return [
        ArchSlot(n_layer=l, n_head=h, d_model=d, dropout=dr, lr=lr, between_block_mlp_layers=btwmlp)
        for l in [1, 2, 4]
        for h in [1, 2, 4]
        for d in [16, 64]
        for btwmlp in [2]
        for dr in [0, 0.1]
        for lr in [1e-3]
    ]


if __name__ == "__main__":
    run_from_cli(default_transformer_sweep, build_architectures)
