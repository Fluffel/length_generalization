from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

ModelFamily = Literal["transformer", "ssm", "hybrid"]


@dataclass
class ArchSlot:
    """One architecture + optimizer entry in a sweep."""

    n_layer: int
    n_head: int = 1
    d_model: int = 64
    between_block_mlp_layers: int = 1
    layer_norm: bool = True
    dropout: float = 0.0
    lr: float = 1e-3


@dataclass
class RunConfig:
    """All settings for training (no CLI); pass to `language_modeling_train.main`."""

    model_family: ModelFamily
    architectures: list[ArchSlot]

    task: str = "parity"
    seeds: int = 1
    job_id: str = ""

    monoid: str = "parity"
    monoid_n: int = 2
    key_size: int = 32
    query_fraction: float = 0.2

    train_length_range: tuple[int, int] = (0, 50)
    test_length_ranges: Optional[list[tuple[int, int]]] = None
    batch_size: int = 64
    test_num: int = 2000

    use_nope: bool = False
    
    # Transformer
    regularize: float = 0.0

    # Hybrid layout: repeat `hybrid_layer_pattern` this many times (same meaning as former `n_layer`).
    hybrid_layer_pattern: str = "sa"

    # SSM / hybrid SSM blocks
    ssm_kernel: str = "s4"

    # Step budgets
    max_steps_default: int = 30_000
    max_steps_large: int = 60_000
    warmup_default: int = 0
    warmup_large: int = 3000
    large_if_transformer_layers_gt: int = 4
    large_if_ssm_layers_gt: int = 4
    large_if_hybrid_repeats_gt: int = 2

    eval_steps: int = 3000
    logging_steps: int = 3000
    weight_decay: float = 0.01
    optim: str = "adamw_torch"
    lr_scheduler_type: str = "linear"

    log_dir: str = "./logs"
    summary_basename: str = "summary.txt"
    report_to: str = "none"

    save_final_weights: bool = False
    print_example_sequences: int = 3

    def resolved_test_length_ranges(self) -> list[tuple[int, int]]:
        if self.test_length_ranges is not None:
            return list(self.test_length_ranges)
        tr = self.train_length_range
        return [tr, (51, 100), (101, 150)]

    # def resolved_hybrid_pattern(self) -> str:
    #     pat = self.hybrid_layer_pattern.strip().lower()
    #     if self.hybrid_start_with_attention is True:
    #         return "as"
    #     if self.hybrid_start_with_attention is False:
    #         return "sa"
    #     return pat

    def train_steps_k(self) -> float:
        """Largest step budget used for logging (actual steps depend on arch slot)."""
        return max(self.max_steps_default, self.max_steps_large) / 1000.0


def default_transformer_sweep() -> RunConfig:
    archs = [
        ArchSlot(n_layer=l, n_head=h, d_model=d, lr=lr)
        for l in [1, 2, 4]
        for h in [1, 2, 4]
        for d in [16, 64, 256]
        for lr in [1e-3, 1e-4]
    ]
    return RunConfig(model_family="transformer", architectures=archs)


def default_ssm_sweep() -> RunConfig:
    archs = [
        ArchSlot(n_layer=l, d_model=d, dropout=dr, lr=lr)
        for l in [16, 32]
        for d in [64, 256]
        for dr in [0, 0.1]
        for lr in [1e-3]
    ]
    return RunConfig(model_family="ssm", architectures=archs)


def default_hybrid_sweep() -> RunConfig:
    archs = [
        ArchSlot(n_layer=l, n_head=h, d_model=d, dropout=dr, lr=lr)
        for l in [1, 2, 4]
        for h in [1, 2, 4]
        for d in [16, 64, 256]
        for dr in [0, 0.1]
        for lr in [1e-3, 1e-4]
    ]
    return RunConfig(
        model_family="hybrid",
        architectures=archs,
        hybrid_layer_pattern="sa",
    )
