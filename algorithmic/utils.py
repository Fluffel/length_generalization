from __future__ import annotations

from dataclasses import dataclass
from itertools import groupby
from typing import Iterator, Literal, Optional, Tuple

# from mambapy.mamba2 import Mamba2Config
from mambapy.mamba import MambaConfig
from transformers import GPT2Config

ModelFamily = Literal["transformer", "ssm", "hybrid"]


# ================================
# Configuration classes
# ================================
@dataclass
class HybridConfig:
    vocab_size: int
    n_positions: int
    n_embd: int = 256
    n_head: int = 4
    dropout: float = 0.0
    bos_token_id: Optional[int] = None
    eos_token_id: Optional[int] = None
    pad_token_id: Optional[int] = None
    nope: bool = False
    # Repeat `layer_pattern` this many times (each char is one block: "a" = GPT-2, "s" = SSM).
    n_pattern_repeats: int = 1
    layer_pattern: str = "sa"
    between_block_mlp_layers: int = 1
    layer_norm: bool = True
    # between_block_mlp_norm: bool = False
    ssm_kernel: str = "s4"

@dataclass
class SSMConfig:
    vocab_size: int
    n_embd: int = 256
    n_layers: int = 4
    d_head: int = 4 # Used only in Mamba2. There is a comment in their code, questioning whether this shouldn't be n_heads. I'm not sure what to do with it. It will be used as n_heads for now.
    dropout: float = 0.2
    ssm_kernel: str = "s4"
    between_block_mlp_layers: int = 1
    layer_norm: bool = True
    # between_block_mlp_norm: bool = False

def mamba_config_from_ssm_config(ssm_config: SSMConfig) -> MambaConfig:
    return MambaConfig(
        d_model=ssm_config.n_embd,
        n_layers=ssm_config.n_layers
    )

def create_ssm_config(tokenizer, config: RunConfig, arch: ArchSlot) -> SSMConfig:
    return SSMConfig(
        vocab_size=len(tokenizer),
        n_embd=arch.d_model,
        n_layers=arch.n_layer,
        dropout=arch.dropout,
        ssm_kernel=config.ssm_kernel,
        between_block_mlp_layers=arch.between_block_mlp_layers,
        layer_norm=arch.layer_norm,
    )

def create_transformer_config(tokenizer, n_positions: int, arch: ArchSlot) -> GPT2Config:
    return GPT2Config(
                vocab_size=len(tokenizer),
                n_positions=n_positions,
                n_embd=arch.d_model,
                n_layer=arch.n_layer,
                n_head=arch.n_head,
                between_block_mlp_layers=arch.between_block_mlp_layers,
                layer_norm=arch.layer_norm,
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                attn_pdrop=0,
                resid_pdrop=0,
                embd_pdrop=0,
            )

def create_hybrid_config(tokenizer, n_positions: int, config: RunConfig, arch: ArchSlot) -> HybridConfig:
    return HybridConfig(
                vocab_size=len(tokenizer),
                n_positions=n_positions,
                n_embd=arch.d_model,
                n_head=arch.n_head,
                dropout=arch.dropout,
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                nope=config.use_nope,
                n_pattern_repeats=arch.n_layer,
                layer_pattern=config.hybrid_layer_pattern,
                between_block_mlp_layers=arch.between_block_mlp_layers,
                layer_norm=arch.layer_norm,
                ssm_kernel=config.ssm_kernel,
            )

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
    num_test_bins: int = 3
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

    @property
    def test_length_ranges(self) -> list[tuple[int, int]]:
        tr = self.train_length_range
        test_length_ranges = []
        length_delta = tr[1] - tr[0]
        for i in range(self.num_test_bins):
            start = tr[0] + i * length_delta # 0, 50 -> 0, 50; 51, 
            end = start + length_delta - 1
            test_length_ranges.append((start, end))
        return test_length_ranges

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

def run_length_encode(data: str) -> Iterator[Tuple[str, int]]:
    """Returns run length encoded Tuples for string"""
    # A memory efficient (lazy) and pythonic solution using generators
    return ((x, sum(1 for _ in y)) for x, y in groupby(data))