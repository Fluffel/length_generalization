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
                attn_pdrop=arch.dropout,
                resid_pdrop=arch.dropout,
                embd_pdrop=arch.dropout,
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
class CurriculumConfig:
    """Curriculum learning schedule over the training sequence length.

    Training proceeds in ``num_steps`` stages. Stage ``i`` (0-indexed) trains on a
    ``step_size``-wide *window* of lengths, sliding forward each stage rather than
    growing from 0: ``(0, step_size - 1)``, then ``(step_size, 2 * step_size - 1)``,
    then ``(2 * step_size, 3 * step_size - 1)``, etc. Each stage runs for
    ``steps_per_stage`` trainer steps (or fewer if solved early), after which the
    window shifts forward by ``step_size`` for the next stage.     Evaluation at the end
    of each stage covers 1x, 2x, and 3x of that stage's cumulative length reached so
    far (recomputed per stage instead of once for the whole run) — e.g. stage 1
    (cumulative length 20) evaluates at (0,19), (20,39), (40,59), independent of the
    narrower (10,19) window it actually trains on.

    When set on ``RunConfig``, this replaces ``train_length_range`` /
    ``test_length_ranges`` / ``num_test_bins`` as the source of truth for
    training and evaluation lengths.
    """

    num_steps: int
    step_size: int
    steps_per_stage: int

    def stage_size(self, stage_idx: int) -> int:
        """Cumulative max length reached by stage ``stage_idx`` (0-indexed); used for eval bins."""
        assert 0 <= stage_idx < self.num_steps
        return self.step_size * (stage_idx + 1)

    def stage_train_range(self, stage_idx: int) -> tuple[int, int]:
        """Train window for stage ``stage_idx``: a ``step_size``-wide slice, not
        cumulative from 0. E.g. step_size=10 -> stage 0 trains on (0,9), stage 1 on
        (10,19), stage 2 on (20,29), ...
        """
        size = self.stage_size(stage_idx)
        prev_size = self.stage_size(stage_idx - 1) if stage_idx > 0 else 0
        return (prev_size, size - 1)

    def stage_test_ranges(self, stage_idx: int) -> list[tuple[int, int]]:
        """1x, 2x, 3x length bins for the given stage, e.g. size=10 -> [(0,9),(10,19),(20,29)]."""
        size = self.stage_size(stage_idx)
        return [(0, size - 1), (size, 2 * size - 1), (2 * size, 3 * size - 1)]

    @property
    def max_steps(self) -> int:
        """Total trainer steps across all stages."""
        return self.num_steps * self.steps_per_stage

    @property
    def max_test_length(self) -> int:
        """Longest length needed anywhere in the curriculum (last stage's 3x bin)."""
        return 3 * self.stage_size(self.num_steps - 1)


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
    """All settings for training (no CLI); pass to `language_modeling_train.main`.

    The model fields below (``model_family``, ``architectures``, ``use_nope``,
    ``ssm_kernel``, ...) are what a spec in ``model_specs/`` sets; see
    ``model_spec.load_model_spec``.
    """

    model_family: ModelFamily
    architectures: list[ArchSlot]

    # Name of the spec the model fields came from, recorded for reproducibility.
    model_spec: Optional[str] = None

    task: str = "parity"
    # Number of training iterations. Each iteration draws a random training seed
    # unless ``seed`` is set, in which case there is always exactly one iteration.
    seeds: int = 1
    # If set, run a single iteration with this training seed (model init and the
    # on-the-fly training stream). Independent of ``dataset_seed``.
    seed: Optional[int] = None
    # Seed used once to materialize eval bins (and formal-language train corpora).
    # Independent of the training-loop seed, which only affects model init and the
    # on-the-fly training stream. None draws a random seed at the start of `main`
    # (the CLI default unless --dataset-seed is passed).
    dataset_seed: Optional[int] = None
    job_id: str = ""

    # MQAR / selective_state_tracking configs. Key vocab size (MQAR) is derived
    # from the longest eval length (see task_datasets.mqar_key_vocab_size), not a
    # separate hyperparameter. SST filler vocab is 2 * max_test_length.
    # ``s5_limited`` samples identity + transpositions; answers still live in S_5.
    monoid: str = "parity"
    monoid_n: int = 2 # Only used for cyclic
    query_fraction_upper: float = 0.2
    query_fraction_lower: float = 0.2

    # Sort configs. None keeps the historical vocab of max_test_length tokens.
    sort_vocab_size: Optional[int] = None

    # MKAR configs
    key_len = 4
    mkar_vocab_size = 128

    # SelectiveCopy configs
    marker_vocab_size = 16
    marker_frequency = 0.2  # lower bound; actual frequency is Uniform[this, 1]

    train_length_range: tuple[int, int] = (0, 50)
    num_test_bins: int = 3
    batch_size: int = 64
    test_num: int = 2000

    # Formal-language tasks only: emit one target token per source token (as
    # formal_lang_suite does) instead of packing the whole target after a separator.
    # See task_datasets.FormalLanguageDataset for why the packed form is much harder
    # to length-generalize than the language itself is.
    formal_aligned_targets: bool = True

    # When set, curriculum learning is used and train_length_range/test_length_ranges/
    # num_test_bins above are ignored (see CurriculumConfig for details).
    curriculum: Optional["CurriculumConfig"] = None

    use_nope: bool = False
    # If true, use olmo_core TransformerConfig-backed builders instead of local GPT2/S4/Mamba ones.
    use_olmo_core: bool = False
    
    # Transformer
    regularize: float = 0.0

    # Hybrid layout: repeat `hybrid_layer_pattern` this many times (same meaning as former `n_layer`).
    hybrid_layer_pattern: str = "sa"

    # Freeze a hybrid model's attention or SSM weights for the initial `freeze_fraction` of
    # training steps (or, with curriculum learning, of *each* curriculum stage's steps), then
    # train the whole model for the remainder. Hybrid `model_family` only (validated in `main`).
    freeze_arch: Optional[Literal["attention", "ssm"]] = None
    freeze_fraction: float = 0.0

    # SSM / hybrid SSM blocks
    ssm_kernel: str = "s4"
    # OLMo GatedDeltaNet (used when use_olmo_core=True for ssm/hybrid).
    # Keep gdn1 as the default so existing configurations and checkpoints retain
    # their architecture; set this to "gdn2" for the channel-wise GDN2 mixer.
    olmo_gdn_variant: Literal["gdn1", "gdn2"] = "gdn2"
    olmo_gdn_allow_neg_eigval: bool = True
    olmo_gdn_expand_v: float = 2.0
    # Head dim is int(olmo_gdn_head_dim_multiplier * d_model / n_head)
    olmo_gdn_head_dim_multiplier: float = 1.0

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
    early_stop = True
    # Stop as soon as every eval length bin reaches this accuracy, even without
    # ``early_stop``. Set above 1.0 to disable. ``early_stop`` still separately
    # stops when the train-length bin is ~perfect.
    solved_acc_threshold: float = 0.98

    log_dir: str = "./logs"
    # Same {task}/summary*.json layout as log_dir, but a separate tree so text
    # summaries and machine-readable run records can be collected independently.
    json_log_dir: str = "./json_logs"
    summary_basename: str = "summary.txt"
    report_to: str = "none"
    wandb_project: Optional[str] = None
    wandb_entity: Optional[str] = None
    wandb_group: Optional[str] = None

    save_final_weights: bool = False
    print_example_sequences: int = 3

    @property
    def test_length_ranges(self) -> list[tuple[int, int]]:
        """Inclusive eval bins: first bin is the train window; later bins are
        contiguous and start at the previous hi + 1.

        For the default ``train_length_range=(0, 50)`` and ``num_test_bins=3`` this
        is ``[(0, 50), (51, 100), (101, 150)]``.
        """
        lo, hi = self.train_length_range
        bins = [(lo, hi)]
        length_delta = hi - lo
        for _ in range(self.num_test_bins - 1):
            lo, hi = hi + 1, hi + length_delta
            bins.append((lo, hi))
        return bins

    def train_steps_k(self) -> float:
        """Largest step budget used for logging (actual steps depend on arch slot)."""
        return max(self.max_steps_default, self.max_steps_large) / 1000.0


def run_length_encode(data: str) -> Iterator[Tuple[str, int]]:
    """Returns run length encoded Tuples for string"""
    # A memory efficient (lazy) and pythonic solution using generators
    return ((x, sum(1 for _ in y)) for x, y in groupby(data))