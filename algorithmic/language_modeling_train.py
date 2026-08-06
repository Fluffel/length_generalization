from __future__ import annotations

import argparse
import logging
import os
import random
import sys
from typing import Any, Optional

import numpy as np
import torch
from transformers import Trainer, TrainerCallback, TrainingArguments

from dataset_generators import build_curriculum_datasets, build_datasets
from models import build_model
from utils import (
    ArchSlot,
    CurriculumConfig,
    RunConfig,
    default_hybrid_sweep,
    default_ssm_sweep,
    default_transformer_sweep,
)

LOGGER = logging.getLogger(__name__)

try:
    import wandb
except ImportError:  # pragma: no cover - optional dependency
    wandb = None


# ---------------------------------------------------------------------------
# Backward compatibility for scripts that ``from ... import *`` and rely on
# module-level names inside Trainer callbacks (e.g. run_multiple_seeds.py).
# ---------------------------------------------------------------------------
train_length_range: tuple[int, int] = (0, 50)
test_length_ranges: list[tuple[int, int]] = []
summary_f: Any = None
n_layer: int = 0
n_head: int = 0
d_model: int = 0
lr: float = 0.0
threshold: float = 1.0
results: dict[str, list[float]] = {}


def compute_metrics(eval_preds):
    logits, labels = eval_preds
    shift_logits = logits[:, :-1]
    shift_labels = labels[:, 1:]
    predictions = np.argmax(shift_logits, axis=-1)
    correct = np.all((predictions == shift_labels) | (shift_labels == -100), axis=1)
    return {"acc": correct.sum() / len(correct)}


def _max_steps_warmup(run_config: RunConfig, arch: ArchSlot) -> tuple[int, int]:
    fam = run_config.model_family
    if fam == "hybrid":
        large = arch.n_layer > run_config.large_if_hybrid_repeats_gt
    elif fam == "ssm":
        large = arch.n_layer > run_config.large_if_ssm_layers_gt
    else:
        large = arch.n_layer > run_config.large_if_transformer_layers_gt
    if large:
        return run_config.max_steps_large, run_config.warmup_large
    return run_config.max_steps_default, run_config.warmup_default


def _summary_rel_path(run_config: RunConfig) -> str:
    if run_config.model_family == "transformer":
        if run_config.use_nope:
            mid = "-nope"
        elif run_config.regularize != 0:
            mid = f"-reg{run_config.regularize}"
        else:
            mid = ""
        return f"summarylm{mid}{run_config.job_id}.txt"
    if run_config.model_family == "ssm":
        return f"summaryssm{run_config.job_id}.txt"
    return f"summaryhybrid{run_config.job_id}.txt"


def format_log_prefix(
    run_config: RunConfig,
    arch: ArchSlot,
    max_steps: int,
) -> str:
    """Model hyperparameters for summary logs."""
    step_k = max_steps / 1000.0

    ln_str = "ln" if arch.layer_norm else "noln"
    pe = "nope" if run_config.use_nope else "pe"
    reg = f"{run_config.regularize}reg"
    btw_blocks = f"{arch.between_block_mlp_layers}mlp"
    neg_eig = ""

    if run_config.use_olmo_core:
        reg = ""
        btw_blocks = ""
        ln_str = ""
        pe = ""
        neg_eig = "ne" if run_config.olmo_gdn_allow_neg_eigval else "none"

    parts: list[str] = []

    if run_config.model_family == "transformer":
        arch_str = "lm" if not run_config.use_olmo_core else "olmolm"
        parts += [
            arch_str,
            reg,
            f"{arch.n_layer}l",
            f"{arch.n_head}h",
            f"{arch.d_model}d",
            f"{arch.dropout}dr",
            btw_blocks,
            pe,
            ln_str,
        ]
    elif run_config.model_family == "ssm":
        arch_str = "ssm" if not run_config.use_olmo_core else "olmo"
        kernel_str = run_config.ssm_kernel if not run_config.use_olmo_core else "gdn"
        parts += [
            arch_str,
            kernel_str,
            f"{arch.n_layer}l",
            f"{arch.d_model}d",
            f"{arch.dropout}dr",
            btw_blocks,
            neg_eig,
            ln_str,
        ]
    else:
        arch_str = "hyb" if not run_config.use_olmo_core else "olmohyb"
        pat = run_config.hybrid_layer_pattern
        kernel_str = run_config.ssm_kernel if not run_config.use_olmo_core else ""
        parts += [
            arch_str,
            kernel_str,
            pat,
            f"{arch.n_layer}l",
            f"{arch.n_head}h",
            f"{arch.d_model}d",
            f"{arch.dropout}dr",
            btw_blocks,
            neg_eig,
            pe,
            ln_str,
        ]
    parts += [f"stp{step_k:.3g}k",
            f"{arch.lr}lr",
    ]
    return "".join(parts)


def configure_logging() -> None:
    """Configure root logging so INFO messages reach cluster stdout."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        stream=sys.stdout,
        force=True,
    )
    LOGGER.info("Root logging configured")


def _perfect_train_acc(acc: Optional[float]) -> bool:
    """Metrics are float; near-1.0 should count as solved."""
    return acc is not None and float(acc) >= 0.9999


class AlgorithmicTrainCallback(TrainerCallback):
    def __init__(
        self,
        run_config: RunConfig,
        arch: ArchSlot,
        train_range: tuple[int, int],
        test_ranges: list[tuple[int, int]],
        summary_file,
        max_steps: int,
        stop_state: dict[str, Any],
        use_wandb: bool = False,
        metric_prefix: str = "",
    ):
        self.run_config = run_config
        self.arch = arch
        self.train_length_range = train_range
        self.test_length_ranges = test_ranges
        self.summary_file = summary_file
        self.max_steps = max_steps
        self.stop_state = stop_state
        self.log_prefix = format_log_prefix(run_config, arch, max_steps)
        self.latest_acc: dict[str, float] = {}
        self.current_epoch: float = 0.0
        self.use_wandb = use_wandb
        self.metric_prefix = metric_prefix
        # Eval metrics use length bins from ``test_ranges`` (e.g. eval_len0-49_acc), not train_length_range
        # (which can differ by one from the first bin). Wrong keys → no early stop / no train/acc in W&B.
        self._eval_acc_keys: list[str] = [f"eval_len{a}-{b}_acc" for a, b in test_ranges]
        self._train_bin_key: str = self._eval_acc_keys[0]
        self._mid_bin_key: str = self._eval_acc_keys[1] if len(self._eval_acc_keys) > 1 else self._train_bin_key
        self._logged_epoch_summary: bool = False

    def _metric_name(self, key: str) -> str:
        if not self.metric_prefix:
            return key
        return f"{self.metric_prefix}/{key}"

    def _log_to_wandb(self, payload: dict[str, Any], trainer_step: int) -> None:
        if not self.use_wandb or wandb is None or wandb.run is None:
            return
        full: dict[str, Any] = {f"{self.metric_prefix}/trainer_step": trainer_step, **payload}
        wandb.log(full)

    def on_evaluate(self, args, state, control, metrics=None, logs=None, eval_dataloader=None, **kwargs):
        metrics = metrics or {}
        assert metrics["epoch"] >= self.current_epoch
        if metrics["epoch"] > self.current_epoch:
            self.latest_acc = {}
            self.current_epoch = metrics["epoch"]
        # Single wandb.log per eval: multiple log(..., step=s) calls for the same step can collapse
        # to one partial row in history so each chart only retains the last key (often one point at max step).
        wandb_eval: dict[str, Any] = {}
        for key in self._eval_acc_keys:
            if key in metrics:
                self.latest_acc[key] = metrics[key]
                wandb_eval[self._metric_name(f"eval/acc/{key.removeprefix('eval_')}")] = metrics[key]
        if self._train_bin_key in self.latest_acc:
            wandb_eval[self._metric_name("train/acc")] = self.latest_acc[self._train_bin_key]
        if wandb_eval:
            self._log_to_wandb(wandb_eval, state.global_step)
        if len(self.latest_acc) == len(self.test_length_ranges):
            solved_train = _perfect_train_acc(self.latest_acc.get(self._train_bin_key))
            epoch_done_one = self.current_epoch >= 1.0 and not self._logged_epoch_summary
            if solved_train:
                control.should_training_stop = True
                self.stop_state["fit_train_data"] = True
                msg = f"early stop {self.current_epoch}\t\t"
                train_show = float(self.latest_acc.get(self._train_bin_key, 0) or 0)
                if train_show >= 0.99:
                    msg = ">> " + msg
                line = "\t".join(
                    [
                        self.log_prefix,
                        msg,
                        "\t\t".join(f"{k}: {v}" for k, v in self.latest_acc.items()),
                        f"\tlr: {self.arch.lr}",
                    ]
                )
                print(line, file=self.summary_file)
                self.summary_file.flush()
                if _perfect_train_acc(self.latest_acc.get(self._mid_bin_key)):
                    self.stop_state["should_stop"] = True
            elif epoch_done_one:
                self._logged_epoch_summary = True
                msg = "reach max step\t\t"
                train_show = float(self.latest_acc.get(self._train_bin_key, 0) or 0)
                if train_show >= 0.99:
                    msg = ">> " + msg
                line = "\t".join(
                    [
                        self.log_prefix,
                        msg,
                        "\t\t".join(f"{k}: {v}" for k, v in self.latest_acc.items()),
                        f"\tlr: {self.arch.lr}",
                    ]
                )
                print(line, file=self.summary_file)
                self.summary_file.flush()

    def on_log(self, args, state, control, logs=None, **kwargs):
        logs = logs or {}
        if "loss" in logs:
            self._log_to_wandb({self._metric_name("train/loss"): logs["loss"]}, state.global_step)


def _apply_curriculum_stage_range(train_dataset, desired_range: tuple[int, int], task_floor: int) -> None:
    """Set ``train_dataset``'s length window to ``desired_range``, clamped below by
    ``task_floor`` (the task's own minimum feasible length, e.g. MKAR needs
    ``length >= 2 * key_len + 1``; see each dataset class's ``__init__``).

    Mutates ``range_min``/``range_max`` in place (rather than constructing a new
    dataset instance) since ``Trainer`` already holds a reference to this exact
    object for its train dataloader; the ``IterableDataset.__iter__`` loops read
    these attributes fresh on every sample draw, so the new window takes effect
    immediately without recreating the dataloader.
    """
    lo, hi = desired_range
    lo = max(lo, task_floor)
    assert lo <= hi, (
        f"curriculum window {desired_range} is entirely below the minimum length "
        f"({task_floor}) required by this task; use a larger --curriculum-step-size."
    )
    train_dataset.range_min = lo
    train_dataset.range_max = hi


class CurriculumTrainCallback(TrainerCallback):
    """Drives curriculum learning and logs one summary line per curriculum step.

    ``eval_steps`` is set to (at most) ``curriculum.steps_per_stage`` so evaluation
    runs periodically *within* a stage — not just at its very end — against the eval
    datasets for that stage's 1x/2x/3x length bins (``stage_eval_datasets[stage_idx]``,
    active since either the previous stage's transition or run setup). This lets a
    stage finish early once it's solved, instead of always burning its full step budget.

    When ``Trainer.eval_dataset`` is a dict, ``Trainer.evaluate()`` recurses once per
    named sub-dataset, calling ``on_evaluate`` separately for *each* length bin with
    only that bin's metrics (not once for the whole stage) — so results are
    accumulated in ``latest_acc`` across calls, same as ``AlgorithmicTrainCallback``.

    Once all of the current stage's bins have reported in, the stage is considered
    "done" (tracked via the per-stage ``stop_state``, reset at every transition) when
    either:
      - the 1x-length bin reaches ~perfect accuracy ("solved"), or
      - ``steps_per_stage`` steps have elapsed since the stage began ("step cap").

    A stage being "done" only means *that stage* stops — it advances the curriculum
    (grows the train dataset's max length, swaps ``trainer.eval_dataset`` to the next
    stage's eval datasets) and training continues uninterrupted. ``control
    .should_training_stop`` (the only thing that actually halts ``Trainer.train()``)
    is set only when the *last* stage is done, i.e. there is no next stage to advance
    to.

    ``trainer`` must be assigned after ``Trainer(...)`` is constructed (the callback
    is needed to build the ``Trainer``, so it can't be passed in up front).
    """

    def __init__(
        self,
        run_config: RunConfig,
        arch: ArchSlot,
        curriculum: CurriculumConfig,
        train_dataset,
        stage_eval_datasets: list[dict[str, Any]],
        summary_file,
        task_floor: int,
        use_wandb: bool = False,
        metric_prefix: str = "",
    ):
        self.run_config = run_config
        self.arch = arch
        self.curriculum = curriculum
        self.train_dataset = train_dataset
        self.stage_eval_datasets = stage_eval_datasets
        self.summary_file = summary_file
        self.task_floor = task_floor
        self.log_prefix = format_log_prefix(run_config, arch, curriculum.max_steps)
        self.use_wandb = use_wandb
        self.metric_prefix = metric_prefix
        self.trainer: Optional[Trainer] = None

        self.num_stages = curriculum.num_steps
        self.stage_idx = 0
        self.stage_start_step = 0
        self._eval_acc_keys = self._stage_eval_keys(self.stage_idx)
        self._train_bin_key = self._eval_acc_keys[0]
        self.latest_acc: dict[str, float] = {}
        # Per-stage stop signal: "should_stop" means *this stage* is done (solved or
        # step-capped), not that training as a whole should halt. Reset every time a
        # new stage begins; see class docstring.
        self.stop_state: dict[str, Any] = {"should_stop": False, "fit_train_data": False}

    def _stage_eval_keys(self, stage_idx: int) -> list[str]:
        return [f"eval_{name}_acc" for name in self.stage_eval_datasets[stage_idx]]

    def _metric_name(self, key: str) -> str:
        if not self.metric_prefix:
            return key
        return f"{self.metric_prefix}/{key}"

    def _log_to_wandb(self, payload: dict[str, Any], trainer_step: int) -> None:
        if not self.use_wandb or wandb is None or wandb.run is None:
            return
        full: dict[str, Any] = {f"{self.metric_prefix}/trainer_step": trainer_step, **payload}
        wandb.log(full)

    def _advance_to_next_stage(self, global_step: int) -> None:
        """Slide the train window forward to the next stage and reset per-stage state.

        Only called when the current stage isn't the last one — see ``on_evaluate``.
        """
        self.stage_idx += 1
        self.stage_start_step = global_step
        _apply_curriculum_stage_range(
            self.train_dataset, self.curriculum.stage_train_range(self.stage_idx), self.task_floor
        )
        self._eval_acc_keys = self._stage_eval_keys(self.stage_idx)
        self._train_bin_key = self._eval_acc_keys[0]
        self.stop_state = {"should_stop": False, "fit_train_data": False}
        if self.trainer is not None:
            self.trainer.eval_dataset = self.stage_eval_datasets[self.stage_idx]

    def on_evaluate(self, args, state, control, metrics=None, logs=None, eval_dataloader=None, **kwargs):
        metrics = metrics or {}
        for key in self._eval_acc_keys:
            if key in metrics:
                self.latest_acc[key] = metrics[key]
        if len(self.latest_acc) < len(self._eval_acc_keys):
            return  # still waiting on the other length bins for this stage

        solved = _perfect_train_acc(self.latest_acc.get(self._train_bin_key))
        step_cap_reached = (state.global_step - self.stage_start_step) >= self.curriculum.steps_per_stage
        if not (solved or step_cap_reached):
            # Stage still in progress: don't log or advance yet, just keep training
            # (mirrors AlgorithmicTrainCallback, which only logs on early-stop/epoch-done).
            self.latest_acc = {}
            return

        if solved:
            self.stop_state["fit_train_data"] = True
        self.stop_state["should_stop"] = True  # this stage is done; see class docstring

        stage_size = self.curriculum.stage_size(self.stage_idx)
        wandb_eval: dict[str, Any] = {
            self._metric_name(f"eval/acc/{key.removeprefix('eval_')}"): val
            for key, val in self.latest_acc.items()
        }
        wandb_eval[self._metric_name("curriculum/stage")] = self.stage_idx + 1
        wandb_eval[self._metric_name("curriculum/size")] = stage_size
        self._log_to_wandb(wandb_eval, state.global_step)

        msg = "early stop" if solved else "reach step cap"
        train_show = float(self.latest_acc.get(self._train_bin_key, 0) or 0)
        if train_show >= 0.99:
            msg = ">> " + msg
        marker = f"[curriculum step {self.stage_idx + 1}/{self.num_stages} size={stage_size}] {msg}"
        line = "\t".join(
            [
                self.log_prefix,
                marker,
                "\t\t".join(f"{k}: {v}" for k, v in self.latest_acc.items()),
                f"\tlr: {self.arch.lr}",
            ]
        )
        print(line, file=self.summary_file)
        self.summary_file.flush()

        self.latest_acc = {}
        is_last_stage = self.stage_idx >= self.num_stages - 1
        if is_last_stage:
            # No next curriculum to advance to: this is the one case a per-stage
            # stop turns into an actual full stop of training.
            control.should_training_stop = True
        else:
            self._advance_to_next_stage(state.global_step)

    def on_log(self, args, state, control, logs=None, **kwargs):
        logs = logs or {}
        if "loss" in logs:
            self._log_to_wandb({self._metric_name("train/loss"): logs["loss"]}, state.global_step)


def _is_wandb_enabled(run_config: RunConfig) -> bool:
    return run_config.report_to.strip().lower() == "wandb"


def _wandb_group_for_experiment(run_config: RunConfig) -> str:
    """Shared group ID so multiple seeds overlay in W&B Compare (same metric keys per seed)."""
    g = run_config.wandb_group or os.environ.get("WANDB_GROUP")
    if g:
        return g
    return f"{run_config.model_family}-{run_config.task}-{run_config.job_id or 'nojobid'}"


def _init_wandb_run_for_seed(run_config: RunConfig, seed: int) -> None:
    """One W&B run per seed: identical metric names + shared ``group`` → one chart, multiple colored runs."""
    if wandb is None:
        raise RuntimeError(
            "W&B logging requested but wandb is not installed. "
            "Install it with `pip install wandb` or set report_to='none'."
        )
    project = run_config.wandb_project or os.environ.get("WANDB_PROJECT", "length_generalization")
    entity = run_config.wandb_entity or os.environ.get("WANDB_ENTITY")
    group = _wandb_group_for_experiment(run_config)
    run_name = f"{run_config.job_id or 'run'}-seed{seed}"
    config = {
        "task": run_config.task,
        "model_family": run_config.model_family,
        "job_id": run_config.job_id,
        "seed": seed,
        "eval_steps": run_config.eval_steps,
        "logging_steps": run_config.logging_steps,
        "num_seeds": run_config.seeds,
    }
    wandb.init(project=project, entity=entity, group=group, name=run_name, config=config, reinit=True)


def _wandb_define_arch_metrics(run_config: RunConfig, metric_prefix: str) -> None:
    """Bind each architecture's curves to its own x-axis (trainer_step), not the run-global step.

    Several ``Trainer.train()`` calls in one W&B run all restart ``global_step`` at 0; logging with
    ``wandb.log(..., step=global_step)`` overwrites the same rows for every architecture so only
    one phase shows up. Per-prefix ``trainer_step`` avoids that.
    """
    if wandb is None or wandb.run is None:
        return
    step_key = f"{metric_prefix}/trainer_step"
    wandb.define_metric(step_key)
    for sub in ("train/loss", "train/acc"):
        wandb.define_metric(f"{metric_prefix}/{sub}", step_metric=step_key)
    for ra, rb in run_config.test_length_ranges:
        lab = f"len{ra}-{rb}_acc"
        wandb.define_metric(f"{metric_prefix}/eval/acc/{lab}", step_metric=step_key)


class customCollator:
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, examples):
        input_ids, pos_ids, labels = tuple(zip(*examples))
        max_len = max(len(item) for item in input_ids)

        [item.extend([self.pad_id] * (max_len - len(item))) for item in input_ids]
        input_ids = torch.LongTensor(input_ids)
        [item.extend([self.pad_id] * (max_len - len(item))) for item in labels]
        labels = torch.LongTensor(labels)
        labels[labels == self.pad_id] = -100
        [item.extend([item[-1]] * (max_len - len(item))) for item in pos_ids]
        pos_ids = torch.LongTensor(pos_ids)

        return {"input_ids": input_ids, "position_ids": pos_ids, "labels": labels}



def main(run_config: RunConfig) -> None:
    global train_length_range, test_length_ranges
    configure_logging()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _ = device

    curriculum = run_config.curriculum
    stage_eval_datasets: Optional[list[dict[str, Any]]] = None
    curriculum_task_floor = 0
    if curriculum is not None:
        train_dataset, stage_eval_datasets = build_curriculum_datasets(run_config)
        # The task's own minimum feasible length: stage 0's desired range starts at 0,
        # so whatever __init__ clamped range_min up to *is* that floor. Captured once,
        # before any stage-transition mutation, and reused for every later stage/arch/seed.
        curriculum_task_floor = train_dataset.range_min
        # Compat globals + example-printing below reflect the full curriculum span.
        train_length_range = curriculum.stage_train_range(curriculum.num_steps - 1)
        test_length_ranges = curriculum.stage_test_ranges(curriculum.num_steps - 1)
        example_eval_datasets = stage_eval_datasets[0]
        example_test_ranges = curriculum.stage_test_ranges(0)
    else:
        train_dataset, test_dataset, train_length_range, test_length_ranges = build_datasets(run_config)
        example_eval_datasets = test_dataset
        example_test_ranges = test_length_ranges
    n_positions = train_dataset.n_positions
    tokenizer = train_dataset.tokenizer

    task_path = os.path.join(run_config.log_dir, run_config.task)
    os.makedirs(task_path, exist_ok=True)
    summary_path = os.path.join(task_path, _summary_rel_path(run_config))
    LOGGER.info("Task output path: %s", task_path)
    LOGGER.info("Summary path: %s", summary_path)

    per_device_bz = (
        run_config.batch_size // torch.cuda.device_count()
        if torch.cuda.is_available()
        else run_config.batch_size
    )
    use_wandb = _is_wandb_enabled(run_config)

    for seed in range(run_config.seeds):
        if use_wandb:
            _init_wandb_run_for_seed(run_config, seed)
        torch.manual_seed(seed)
        random.seed(seed)

        try:
            with open(summary_path, "a") as summary_file:
                # Sanity check: print example sequences from first test length range
                first_range = example_test_ranges[0]
                key0 = f"len{first_range[0]}-{first_range[1]}"
                for i in range(run_config.print_example_sequences):
                    print("\ninput example:", flush=True)
                    print(" ".join(tokenizer.convert_ids_to_tokens(example_eval_datasets[key0][i][0])), flush=True)
                    print("label example:", flush=True)
                    print(" ".join(tokenizer.convert_ids_to_tokens(example_eval_datasets[key0][i][2])), flush=True)

                stop_state: dict[str, Any] = {"should_stop": False, "fit_train_data": False}

                for arch in run_config.architectures:
                    max_steps, warmup_steps = _max_steps_warmup(run_config, arch)
                    if curriculum is not None:
                        max_steps = curriculum.max_steps
                        # Eval at least once per stage's step budget so a stage can be detected as
                        # "solved" and advance early instead of always burning steps_per_stage steps.
                        eval_steps = min(run_config.eval_steps, curriculum.steps_per_stage)
                        # Reset the shared train_dataset (reused across archs/seeds) back to stage 0;
                        # a prior arch's run may have advanced it to a later stage's window.
                        _apply_curriculum_stage_range(
                            train_dataset, curriculum.stage_train_range(0), curriculum_task_floor
                        )
                        initial_eval_dataset = stage_eval_datasets[0]
                    else:
                        eval_steps = run_config.eval_steps
                        initial_eval_dataset = test_dataset

                    output_tag = format_log_prefix(run_config, arch, max_steps)
                    # Same metric keys for every seed; seeds differ by separate W&B runs in one group.
                    metric_prefix = f"{run_config.task}/{output_tag}"
                    if use_wandb and wandb is not None and wandb.run is not None:
                        _wandb_define_arch_metrics(run_config, metric_prefix)
                        wandb.config.update(
                            {
                                f"architectures.{metric_prefix}": {
                                    "max_steps": max_steps,
                                    "lr": arch.lr,
                                    "n_layer": arch.n_layer,
                                    "n_head": arch.n_head,
                                    "d_model": arch.d_model,
                                    "dropout": arch.dropout,
                                    "between_block_mlp_layers": arch.between_block_mlp_layers,
                                    "layer_norm": arch.layer_norm,
                                }
                            },
                            allow_val_change=True,
                        )

                    model = build_model(run_config, arch, tokenizer, n_positions)
                    print("wte std:", model.wte.weight.std().item() if hasattr(model, "wte") else model.transformer.wte.weight.std().item())
                    training_args = TrainingArguments(
                        output_dir=task_path,  # save_strategy="no" → nothing written here; one dir for all runs
                        per_device_train_batch_size=per_device_bz,
                        per_device_eval_batch_size=per_device_bz,
                        max_steps=max_steps,
                        eval_strategy="steps",
                        eval_steps=eval_steps,
                        save_strategy="no",
                        logging_strategy="steps",
                        logging_steps=run_config.logging_steps,
                        learning_rate=arch.lr,
                        weight_decay=run_config.weight_decay,
                        optim=run_config.optim,
                        lr_scheduler_type=run_config.lr_scheduler_type,
                        warmup_steps=warmup_steps,
                        report_to="none",
                        run_name=metric_prefix,
                    )

                    if curriculum is not None:
                        cb = CurriculumTrainCallback(
                            run_config,
                            arch,
                            curriculum,
                            train_dataset,
                            stage_eval_datasets,
                            summary_file,
                            curriculum_task_floor,
                            use_wandb=use_wandb,
                            metric_prefix=metric_prefix,
                        )
                    else:
                        cb = AlgorithmicTrainCallback(
                            run_config,
                            arch,
                            train_length_range,
                            test_length_ranges,
                            summary_file,
                            max_steps,
                            stop_state,
                            use_wandb=use_wandb,
                            metric_prefix=metric_prefix,
                        )

                    trainer = Trainer(
                        model=model,
                        args=training_args,
                        train_dataset=train_dataset,
                        eval_dataset=initial_eval_dataset,
                        data_collator=customCollator(tokenizer.pad_token_id),
                        compute_metrics=compute_metrics,
                        callbacks=[cb],
                    )
                    if curriculum is not None:
                        cb.trainer = trainer
                    trainer.train()

                    if run_config.save_final_weights:
                        wpath = os.path.join(task_path, f"{output_tag}_weights_seed{seed}_id{run_config.job_id}.pt")
                        torch.save(trainer.model.state_dict(), wpath)
        finally:
            if use_wandb and wandb is not None and wandb.run is not None:
                wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train with a preset RunConfig from utils.py (edit there or pass --preset).")
    parser.add_argument(
        "--preset",
        choices=["transformer", "ssm", "hybrid"],
        default="transformer",
        help="Which default RunConfig factory to use; full settings live in utils.RunConfig.",
    )
    parser.add_argument("--report-to", type=str, default="wandb", choices=["none", "wandb"])
    parser.add_argument("--wandb-project", type=str, default=None)
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-group", type=str, default=None)
    parser.add_argument("--logging-steps", type=int, default=None)
    parser.add_argument("--eval-steps", type=int, default=None)
    args = parser.parse_args()
    presets = {
        "transformer": default_transformer_sweep,
        "ssm": default_ssm_sweep,
        "hybrid": default_hybrid_sweep,
    }
    rc = presets[args.preset]()
    rc.report_to = args.report_to
    rc.wandb_project = args.wandb_project
    rc.wandb_entity = args.wandb_entity
    rc.wandb_group = args.wandb_group
    if args.logging_steps is not None:
        rc.logging_steps = args.logging_steps
    if args.eval_steps is not None:
        rc.eval_steps = args.eval_steps
    main(rc)
