"""
Backward-compatible entrypoint. Prefer ``language_modeling_train.main(RunConfig)`` and
edit hyperparameters in ``utils.RunConfig`` / ``default_ssm_sweep``.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils import default_ssm_sweep, ArchSlot
from run_scripts.language_modeling_train_shared import run_from_cli


def build_architectures(_args) -> list[ArchSlot]:
    return [
        ArchSlot(n_layer=l, d_model=d, dropout=dr, lr=lr, between_block_mlp_layers=1)
        for l in [1]
        for d in [16]
        for dr in [0, 0.1]
        for lr in [1e-3]
    ]


if __name__ == "__main__":
    run_from_cli(default_ssm_sweep, build_architectures)
