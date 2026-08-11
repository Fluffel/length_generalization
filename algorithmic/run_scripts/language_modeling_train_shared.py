"""
Shared CLI runner for algorithmic training scripts.

Model-specific run scripts should only define architecture sweeps and pass them
to ``run_from_cli``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from language_modeling_train import main
from utils import ArchSlot, CurriculumConfig, RunConfig

TASK_CHOICES = [
    # Algorithmic tasks (dataset_generators._make_task_dataset)
    "bin_majority",
    "majority",
    "bin_majority_interleave",
    "unique_copy",
    "repeat_copy",
    "sort",
    "parity",
    "addition",
    "mqar",
    "flipflop",
    "selective_copy",
    "mkar",
    # Formal-language tasks (dataset_generators._build_formal_datasets)
    "tomita_1",
    "tomita_2",
    "tomita_3",
    "tomita_4",
    "tomita_5",
    "tomita_6",
    "tomita_7",
    "d_2",
    "d_3",
    "d_4",
    "d_12",
    "abab_star",
    "aa_star",
    "an_star_a2",
    "aa_star_bb_star",
    "ab_star_d_bc_star",
    "012_star_0_2_star",
]


def _parse_length_range(value: str) -> tuple[int, int]:
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("train length range must be 'min,max'")
    try:
        start, end = int(parts[0]), int(parts[1])
    except ValueError as exc:
        raise argparse.ArgumentTypeError("train length range values must be integers") from exc
    return (start, end)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, choices=TASK_CHOICES)
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--job-id", type=str, default="")

    parser.add_argument(
        "--use-olmo",
        "--use_olmo",
        action="store_true",
        help=(
            "Use OLMo-core blocks. Depending on model family, this may ignore "
            "--regularize, --ssm-kernel, --noln, and/or --nope."
        ),
    )
    parser.add_argument("--nope", action="store_true")
    parser.add_argument("--noln", action="store_true", help="Disable layer norm in architecture slots")
    parser.add_argument("--regularize", type=float, default=0.0)
    parser.add_argument("--ssm-kernel", type=str, default="s4", choices=["s4", "mamba", "mamba2", "mamba3"])
    parser.add_argument("--hybrid-layer-pattern", type=str, default="sa")

    parser.add_argument(
        "--freeze",
        type=str,
        default=None,
        choices=["attention", "ssm"],
        help=(
            "Freeze this sub-architecture's weights within a hybrid model for the initial "
            "fraction of training (see --freeze-fraction), then train the whole model for the "
            "remaining steps. With curriculum learning, freezing is re-applied at the start of "
            "every curriculum stage. Hybrid models only."
        ),
    )
    parser.add_argument(
        "--freeze-fraction",
        type=float,
        default=0.5,
        help=(
            "Fraction (0, 1] of training steps (or, with curriculum learning, of each stage's "
            "steps) during which --freeze weights are frozen. Requires --freeze."
        ),
    )

    parser.add_argument("--train-steps", type=int, default=None)
    parser.add_argument("--warmup-steps", type=int, default=None)
    parser.add_argument("--eval-steps", type=int, default=None)
    parser.add_argument("--logging-steps", type=int, default=None)
    parser.add_argument(
        "--train-length-range",
        type=_parse_length_range,
        default=(0, 50),
        help="Comma-separated min,max for training sequence length (e.g., 0,50). Ignored if curriculum flags are set.",
    )

    parser.add_argument(
        "--curriculum-num-steps",
        type=int,
        default=None,
        help="Number of curriculum stages. Requires --curriculum-step-size and --curriculum-steps-per-stage.",
    )
    parser.add_argument(
        "--curriculum-step-size",
        type=int,
        default=None,
        help="Train length grows by this much each stage (stage i trains on lengths up to step_size*(i+1)).",
    )
    parser.add_argument(
        "--curriculum-steps-per-stage",
        type=int,
        default=None,
        help="Number of trainer steps to run at each curriculum stage before growing the length.",
    )

    parser.add_argument("--save-final-weights", action="store_true")
    parser.add_argument("--report-to", type=str, default="wandb", choices=["none", "wandb"])
    parser.add_argument("--wandb-project", type=str, default=None)
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-group", type=str, default=None)

    parser.add_argument("--monoid", type=str, default="parity", choices=["parity", "cyclic"])
    parser.add_argument("--monoid_n", type=int, default=2)
    parser.add_argument("--key_size", type=int, default=32)
    parser.add_argument("--query-fraction-upper", type=float, default=0.2)
    parser.add_argument("--query-fraction-lower", type=float, default=0.2)

    parser.add_argument("--key-len", type=int, default=4)
    parser.add_argument("--mkar-vocab-size", type=int, default=128)
    parser.add_argument("--marker-vocab-size", type=int, default=16)
    return parser


def apply_args_to_config(rc: RunConfig, args: argparse.Namespace) -> None:
    rc.task = args.task
    rc.seeds = args.seeds
    rc.job_id = args.job_id

    rc.use_nope = args.nope
    rc.use_olmo_core = args.use_olmo
    rc.regularize = args.regularize
    rc.ssm_kernel = args.ssm_kernel
    rc.hybrid_layer_pattern = args.hybrid_layer_pattern.strip().lower()

    if args.freeze is not None and args.freeze_fraction is None:
        raise SystemExit("--freeze requires --freeze-fraction to also be set.")
    if args.freeze_fraction is not None:
        if not (0.0 < args.freeze_fraction <= 1.0):
            raise SystemExit(f"--freeze-fraction must be in (0, 1], got {args.freeze_fraction}.")
    if args.freeze is not None and rc.model_family != "hybrid":
        raise SystemExit(
            f"--freeze is only supported for hybrid models, got model_family={rc.model_family!r}."
        )
    rc.freeze_arch = args.freeze
    rc.freeze_fraction = args.freeze_fraction if args.freeze_fraction is not None else 0.0

    rc.train_length_range = args.train_length_range

    curriculum_args = {
        "--curriculum-num-steps": args.curriculum_num_steps,
        "--curriculum-step-size": args.curriculum_step_size,
        "--curriculum-steps-per-stage": args.curriculum_steps_per_stage,
    }
    num_curriculum_args_set = sum(v is not None for v in curriculum_args.values())
    if num_curriculum_args_set > 0:
        missing = [name for name, val in curriculum_args.items() if val is None]
        if missing:
            raise SystemExit(
                "Curriculum learning requires all of --curriculum-num-steps, "
                f"--curriculum-step-size, --curriculum-steps-per-stage; missing: {', '.join(missing)}"
            )
        rc.curriculum = CurriculumConfig(
            num_steps=args.curriculum_num_steps,
            step_size=args.curriculum_step_size,
            steps_per_stage=args.curriculum_steps_per_stage,
        )

    rc.save_final_weights = args.save_final_weights
    rc.report_to = args.report_to
    rc.wandb_project = args.wandb_project
    rc.wandb_entity = args.wandb_entity
    rc.wandb_group = args.wandb_group

    rc.monoid = args.monoid
    rc.monoid_n = args.monoid_n
    rc.key_size = args.key_size
    rc.query_fraction_upper = args.query_fraction_upper
    rc.query_fraction_lower = args.query_fraction_lower

    rc.key_len = args.key_len
    rc.mkar_vocab_size = args.mkar_vocab_size
    rc.marker_vocab_size = args.marker_vocab_size

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


def run_from_cli(
    default_config_factory: Callable[[], RunConfig],
    architecture_builder: Callable[[argparse.Namespace], list[ArchSlot]],
) -> None:
    parser = build_parser()
    args = parser.parse_args()
    rc = default_config_factory()
    rc.architectures = architecture_builder(args)
    apply_args_to_config(rc, args)
    main(rc)
