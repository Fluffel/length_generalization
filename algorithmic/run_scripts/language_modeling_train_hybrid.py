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
    parser.add_argument("--task", type=str, choices=["bin_majority", "majority", "bin_majority_interleave", "unique_copy", "repeat_copy", "sort", "parity", "addition", "mqar", "flipflop"])
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--job-id", type=str, default="")
    parser.add_argument(
        "--use-olmo",
        "--use_olmo",
        action="store_true",
        help=(
            "Use OLMo-core hybrid blocks. "
            "When enabled, --ssm-kernel is ignored (OLMo uses GatedDeltaNet for 's' blocks), "
            "--noln is ignored (OLMo always uses layer norm), --nope as OLMO uses ROPE instead of the GPT-2 absolute positional embeddings, and local between-block MLP depth is ignored."
        ),
    )
    parser.add_argument("--nope", action="store_true")
    parser.add_argument("--ssm-kernel", type=str, default="s4", choices=["s4", "mamba"])
    parser.add_argument("--hybrid-layer-pattern", type=str, default="sa")
    parser.add_argument("--train-steps", type=int, default=None)
    parser.add_argument("--warmup-steps", type=int, default=None)
    parser.add_argument("--logging-steps", type=int, default=None)
    parser.add_argument("--eval-steps", type=int, default=None)
    parser.add_argument("--noln", action="store_true", help="Disable layer norm")
    parser.add_argument("--save-final-weights", action="store_true")
    parser.add_argument("--monoid", type=str, default="parity", choices=["parity", "cyclic"])
    parser.add_argument("--monoid_n", type=int, default=2)
    parser.add_argument("--key_size", type=int, default=32)
    parser.add_argument("--query_fraction", type=float, default=0.2)
    parser.add_argument("--report-to", type=str, default="wandb", choices=["none", "wandb"])
    parser.add_argument("--wandb-project", type=str, default=None)
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-group", type=str, default=None)
    args = parser.parse_args()
    archs = [
        ArchSlot(n_layer=l, n_head=h, d_model=d, dropout=dr, lr=lr, between_block_mlp_layers=btwmlp, layer_norm=not args.noln)
        for l in [1, 2, 4]
        for h in [1, 2, 4]
        for d in [16, 64]
        for dr in [0, 0.1]
        for lr in [1e-3]
        for btwmlp in [2]
    ]
    rc = default_hybrid_sweep()

    rc.architectures = archs
    rc.use_nope = args.nope
    rc.use_olmo_core = args.use_olmo
    rc.hybrid_layer_pattern = args.hybrid_layer_pattern.strip().lower()
    rc.ssm_kernel = args.ssm_kernel
    rc.task = args.task
    rc.seeds = args.seeds
    rc.monoid = args.monoid
    rc.monoid_n = args.monoid_n
    rc.key_size = args.key_size
    rc.query_fraction = args.query_fraction
    rc.report_to = args.report_to
    rc.wandb_project = args.wandb_project
    rc.wandb_entity = args.wandb_entity
    rc.wandb_group = args.wandb_group

    if args.train_steps is not None:
        rc.max_steps_default = args.train_steps
        rc.max_steps_large = args.train_steps
    if args.warmup_steps is not None:
        rc.warmup_default = args.warmup_steps
        rc.warmup_large = args.warmup_steps
    if args.logging_steps is not None:
        rc.logging_steps = args.logging_steps
    if args.eval_steps is not None:
        rc.eval_steps = args.eval_steps
    
    rc.save_final_weights = args.save_final_weights
    rc.job_id = args.job_id
    main(rc)
