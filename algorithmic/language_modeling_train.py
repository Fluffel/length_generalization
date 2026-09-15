from __future__ import annotations

import argparse
import logging
import math
import os
import secrets
import sys
from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np
import torch
from transformers import Trainer, TrainerCallback, TrainingArguments
from transformers.trainer_utils import set_seed

from dataset_generators import ALL_TASKS, build_curriculum_datasets, build_datasets, is_formal_task
from model_spec import available_model_specs, load_model_spec
from models import build_model, hybrid_group_parameters
from run_record import (
    accs_jsonable,
    arch_run_entry,
    new_run_record,
    run_record_path,
    summary_rel_path,
    write_run_record,
)
from utils import ArchSlot, CurriculumConfig, RunConfig

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


# Backward-compatible alias; the canonical helper lives in ``run_record``.
_summary_rel_path = summary_rel_path


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
        if run_config.model_family == "ssm":
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
        kernel_str = (
            run_config.ssm_kernel
            if not run_config.use_olmo_core
            else run_config.olmo_gdn_variant
        )
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
        kernel_str = (
            run_config.ssm_kernel
            if not run_config.use_olmo_core
            else run_config.olmo_gdn_variant
        )
        parts += [
            arch_str,
            pat,
            kernel_str,
            f"{arch.n_layer}l",
            f"{arch.n_head}h",
            f"{arch.d_model}d",
            f"{arch.dropout}dr",
            btw_blocks,
            neg_eig,
            pe,
            ln_str,
        ]
        if run_config.freeze_arch is not None:
            # e.g. "frza0.5" (freeze attention for the first 50% of steps) or "frzssm0.8".
            abbrev = "a" if run_config.freeze_arch == "attention" else run_config.freeze_arch
            parts.append(f"frz{abbrev}{run_config.freeze_fraction:g}")
    if is_formal_task(run_config.task) and not run_config.formal_aligned_targets:
        # Only the non-default serialization is marked, so aligned runs keep the log
        # names the plotting scripts already expect while the two cannot be conflated.
        parts.append("packedtgt")

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


@dataclass
class _EvalSnapshot:
    """One complete eval (all length bins) kept on the per-bin best list."""

    accs: dict[str, float]
    epoch: float
    global_step: int


def update_bin_best_snapshots(
    best: list[_EvalSnapshot],
    new: _EvalSnapshot,
    keys: list[str],
) -> list[_EvalSnapshot]:
    """Keep eval snapshots that are maximal in at least one length bin.

    - If ``new`` is at least as good as the running max in every bin and strictly
      better in at least one, it replaces the whole list.
    - If it is strictly better in some bins but not all, it is appended and
      any snapshot that is no longer maximal in any bin is dropped.
    - If it is not strictly better in any bin, the list is unchanged.
    """
    if not best:
        return [new]
    maxima = {k: max(s.accs[k] for s in best) for k in keys}
    better = [k for k in keys if new.accs[k] > maxima[k]]
    if not better:
        return best
    at_least = [k for k in keys if new.accs[k] >= maxima[k]]
    if len(at_least) == len(keys):
        return [new]
    combined = best + [new]
    new_maxima = {k: max(s.accs[k] for s in combined) for k in keys}
    return [s for s in combined if any(s.accs[k] == new_maxima[k] for k in keys)]


def _all_bins_at_least(accs: dict[str, float], keys: list[str], threshold: float) -> bool:
    return all(float(accs.get(k, 0.0) or 0.0) >= threshold for k in keys)


class AlgorithmicTrainCallback(TrainerCallback):
    """Tracks per-bin best evals and writes them as summary lines when training ends.

    HuggingFace ``Trainer.evaluate()`` calls ``on_evaluate`` once per length bin, so
    accuracies are accumulated in ``latest_acc`` until a full eval is in. Each complete
    eval updates a list of snapshots that are maximal in at least one bin (see
    ``update_bin_best_snapshots``). Training curves still go to W&B every eval; the
    summary file is written in ``on_train_end``, one line per kept snapshot.

    Training also stops (without requiring ``--early-stop``) as soon as every bin
    reaches ``run_config.solved_acc_threshold``. ``--early-stop`` still stops when
    the train-length bin is ~perfect, even if the OOD bins are not.
    """

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
        self.early_stop = run_config.early_stop
        self.solved_acc_threshold = run_config.solved_acc_threshold
        self.metric_prefix = metric_prefix
        # Eval metrics use length bins from ``test_ranges`` (e.g. eval_len0-50_acc), not train_length_range
        # (which can differ by one from the first bin). Wrong keys → no early stop / no train/acc in W&B.
        self._eval_acc_keys: list[str] = [f"eval_len{a}-{b}_acc" for a, b in test_ranges]
        self._train_bin_key: str = self._eval_acc_keys[0]
        self._mid_bin_key: str = self._eval_acc_keys[1] if len(self._eval_acc_keys) > 1 else self._train_bin_key
        self._best_snapshots: list[_EvalSnapshot] = []
        self._logged_snapshots: list[dict[str, Any]] = []
        self._stopped_early: bool = False
        self._summary_written: bool = False

    def _metric_name(self, key: str) -> str:
        if not self.metric_prefix:
            return key
        return f"{self.metric_prefix}/{key}"

    def _log_to_wandb(self, payload: dict[str, Any], trainer_step: int) -> None:
        if not self.use_wandb or wandb is None or wandb.run is None:
            return
        full: dict[str, Any] = {f"{self.metric_prefix}/trainer_step": trainer_step, **payload}
        wandb.log(full)

    def _write_summary_line(self, accs: dict[str, float], msg: str) -> None:
        train_show = float(accs.get(self._train_bin_key, 0) or 0)
        if train_show >= 0.99:
            msg = ">> " + msg
        line = "\t".join(
            [
                self.log_prefix,
                msg,
                "\t\t".join(f"{k}: {accs[k]}" for k in self._eval_acc_keys if k in accs),
                f"\tlr: {self.arch.lr}",
            ]
        )
        print(line, file=self.summary_file)
        self.summary_file.flush()

    def _logged_snapshot_from(self, snap: _EvalSnapshot) -> dict[str, Any]:
        """One summary-file row: full-bin accuracies for a Pareto-kept eval."""
        if self._stopped_early:
            status = f"early stop {snap.epoch}"
        else:
            status = "reach max step"
        train_show = float(snap.accs.get(self._train_bin_key, 0) or 0)
        if train_show >= 0.99:
            status = ">> " + status
        # Bin order matches the log line (and is at most ``len(test_ranges)`` long).
        accs = accs_jsonable({k: snap.accs[k] for k in self._eval_acc_keys if k in snap.accs})
        return {
            "status": status,
            "epoch": snap.epoch,
            "global_step": snap.global_step,
            "accs": accs,
        }

    def reported_evals(self) -> list[dict[str, Any]]:
        """Logged snapshots: one dict per summary line, ≤ number of eval bins."""
        if self._summary_written:
            return list(self._logged_snapshots)
        return [self._logged_snapshot_from(snap) for snap in self._best_snapshots]

    def _write_best_results(self) -> None:
        if self._summary_written:
            return
        self._summary_written = True
        self._logged_snapshots = [self._logged_snapshot_from(snap) for snap in self._best_snapshots]
        for snap in self._best_snapshots:
            if self._stopped_early:
                msg = f"early stop {snap.epoch}\t\t"
            else:
                msg = "reach max step\t\t"
            self._write_summary_line(snap.accs, msg)

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
        if len(self.latest_acc) != len(self.test_length_ranges):
            return

        snapshot = _EvalSnapshot(
            accs=dict(self.latest_acc),
            epoch=self.current_epoch,
            global_step=state.global_step,
        )
        self._best_snapshots = update_bin_best_snapshots(
            self._best_snapshots, snapshot, self._eval_acc_keys
        )

        solved_all_bins = _all_bins_at_least(
            snapshot.accs, self._eval_acc_keys, self.solved_acc_threshold
        )
        solved_train = self.early_stop and _perfect_train_acc(snapshot.accs.get(self._train_bin_key))
        if not (solved_all_bins or solved_train):
            return

        control.should_training_stop = True
        self._stopped_early = True
        self.stop_state["fit_train_data"] = True
        if solved_all_bins or _perfect_train_acc(snapshot.accs.get(self._mid_bin_key)):
            self.stop_state["should_stop"] = True

    def on_train_end(self, args, state, control, **kwargs):
        self._write_best_results()

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


def _wandb_define_curriculum_metrics(metric_prefix: str, curriculum: CurriculumConfig) -> None:
    """Bind curriculum W&B plots to sensible x-axes instead of raw trainer step.

    - ``eval/acc/{1x,2x,3x}``, ``train/acc``, and ``curriculum/size`` use a fixed
      ``curriculum_step`` (1..num_steps) x-axis, so results from *every* stage land
      on the same three eval plots (+ one train/acc plot) instead of getting a brand
      new one-point chart per stage per length-range (the literal length range used
      to be baked into the metric name, and it changes every stage).
    - ``train/loss`` gets one *separate* metric per stage (``stage{i}/train/loss``),
      each bound to its own local step counter (``stage{i}/train_step``, steps since
      that stage began), so W&B renders ``num_steps`` independent loss curves
      instead of stitching differently-lengthed training phases onto one continuous,
      jumbled chart.
    """
    if wandb is None or wandb.run is None:
        return
    stage_step_key = f"{metric_prefix}/curriculum_step"
    wandb.define_metric(stage_step_key)
    for sub in ("train/acc", "eval/acc/1x", "eval/acc/2x", "eval/acc/3x", "curriculum/size"):
        wandb.define_metric(f"{metric_prefix}/{sub}", step_metric=stage_step_key)
    for stage_idx in range(curriculum.num_steps):
        stage_num = stage_idx + 1
        step_key = f"{metric_prefix}/stage{stage_num}/train_step"
        wandb.define_metric(step_key)
        wandb.define_metric(f"{metric_prefix}/stage{stage_num}/train/loss", step_metric=step_key)


class FreezeCallback(TrainerCallback):
    """Freezes a hybrid model's attention or SSM parameters for the first ``fraction`` of steps
    in each training "phase".

    A phase is the whole run for plain training, or -- with curriculum learning -- a single
    curriculum stage; ``CurriculumTrainCallback`` calls ``freeze_now()`` again on every stage
    transition so freezing is re-applied at the start of each stage (see its docstring).
    ``phase_start``/``phase_len`` are callables (not plain ints) so this callback can read the
    curriculum callback's live ``stage_start_step``, which changes as stages advance.

    Freezing/unfreezing toggles ``requires_grad`` rather than excluding parameters from the
    optimizer. HF ``Trainer`` builds the optimizer's parameter groups once -- from
    ``requires_grad`` at that time -- *before* ``on_train_begin`` (this callback's first hook)
    ever runs; a parameter excluded there would never be updated again even after being
    unfrozen. Toggling ``requires_grad`` after the optimizer already contains every parameter
    works instead, because e.g. ``AdamW.step()`` simply skips any parameter whose ``.grad`` is
    ``None`` (as is the case while frozen, since autograd skips parameters that don't require
    grad).
    """

    def __init__(
        self,
        model: Any,
        freeze_group: str,
        fraction: float,
        phase_start: Callable[[], int],
        phase_len: Callable[[], int],
    ):
        self.freeze_group = freeze_group
        self.fraction = fraction
        self.phase_start = phase_start
        self.phase_len = phase_len
        self._params = list(hybrid_group_parameters(model, freeze_group))
        if not self._params:
            LOGGER.warning("--freeze %s matched no parameters; nothing will be frozen.", freeze_group)
        self._frozen = False

    def freeze_now(self) -> None:
        for p in self._params:
            p.requires_grad_(False)
        self._frozen = True
        LOGGER.info("Froze %s parameters (%d tensors).", self.freeze_group, len(self._params))

    def unfreeze_now(self) -> None:
        for p in self._params:
            p.requires_grad_(True)
        self._frozen = False
        LOGGER.info("Unfroze %s parameters (%d tensors).", self.freeze_group, len(self._params))

    def on_train_begin(self, args, state, control, **kwargs):
        self.freeze_now()

    def on_step_begin(self, args, state, control, **kwargs):
        if not self._frozen:
            return
        threshold = self.phase_start() + math.ceil(self.fraction * self.phase_len())
        if state.global_step >= threshold:
            self.unfreeze_now()


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

    W&B logging (see ``_wandb_define_curriculum_metrics``): the three eval bins are
    logged under fixed names (``eval/acc/1x`` etc.) against a ``curriculum_step``
    x-axis so every stage contributes a point to the *same* three plots, and
    per-step training loss is logged under a separate ``stage{i}/train/loss`` metric
    per stage so stages don't get stitched into one chart.

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
        freeze_callback: Optional["FreezeCallback"] = None,
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
        # Re-freezes `freeze_callback`'s weights at the start of every curriculum stage; see
        # `_advance_to_next_stage` and `FreezeCallback`'s docstring.
        self.freeze_callback = freeze_callback
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
        self.early_stop = run_config.early_stop
        self._reported_evals: list[dict[str, Any]] = []

    def reported_evals(self) -> list[dict[str, Any]]:
        """Per-stage evals matching the lines written to the summary file."""
        return list(self._reported_evals)

    def _stage_eval_keys(self, stage_idx: int) -> list[str]:
        return [f"eval_{name}_acc" for name in self.stage_eval_datasets[stage_idx]]

    def _metric_name(self, key: str) -> str:
        if not self.metric_prefix:
            return key
        return f"{self.metric_prefix}/{key}"

    def _log_to_wandb(self, payload: dict[str, Any]) -> None:
        if not self.use_wandb or wandb is None or wandb.run is None:
            return
        wandb.log(payload)

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
        if self.freeze_callback is not None:
            self.freeze_callback.freeze_now()

    def on_evaluate(self, args, state, control, metrics=None, logs=None, eval_dataloader=None, **kwargs):
        metrics = metrics or {}
        for key in self._eval_acc_keys:
            if key in metrics:
                self.latest_acc[key] = metrics[key]
        if len(self.latest_acc) < len(self._eval_acc_keys):
            return  # still waiting on the other length bins for this stage

        solved = False
        if self.early_stop:
            solved = _perfect_train_acc(self.latest_acc.get(self._train_bin_key))
            
        step_cap_reached = (state.global_step - self.stage_start_step) >= self.curriculum.steps_per_stage
        if not (solved or step_cap_reached):
            # Stage still in progress: don't log or advance yet, just keep training.
            self.latest_acc = {}
            return

        if solved:
            self.stop_state["fit_train_data"] = True
        self.stop_state["should_stop"] = True  # this stage is done; see class docstring

        stage_size = self.curriculum.stage_size(self.stage_idx)
        # Fixed metric names (1x/2x/3x) instead of the literal, ever-changing length
        # range so every stage's point lands on the same three plots — see
        # _wandb_define_curriculum_metrics.
        eval_1x, eval_2x, eval_3x = (self.latest_acc[key] for key in self._eval_acc_keys)
        wandb_eval: dict[str, Any] = {
            self._metric_name("curriculum_step"): self.stage_idx + 1,
            self._metric_name("curriculum/size"): stage_size,
            self._metric_name("train/acc"): eval_1x,
            self._metric_name("eval/acc/1x"): eval_1x,
            self._metric_name("eval/acc/2x"): eval_2x,
            self._metric_name("eval/acc/3x"): eval_3x,
        }
        self._log_to_wandb(wandb_eval)

        msg = "early stop" if solved else "reach step cap"
        train_show = float(self.latest_acc.get(self._train_bin_key, 0) or 0)
        if train_show >= 0.99:
            msg = ">> " + msg
        marker = f"[curriculum step {self.stage_idx + 1}/{self.num_stages} size={stage_size}] {msg}"
        self._reported_evals.append(
            {
                "status": msg,
                "stage": self.stage_idx + 1,
                "num_stages": self.num_stages,
                "stage_size": stage_size,
                "global_step": state.global_step,
                "accs": accs_jsonable(self.latest_acc),
            }
        )
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
        if "loss" not in logs:
            return
        # Separate metric + local step counter per stage (see class docstring / the
        # helper's docstring above), so W&B renders one independent loss curve per
        # curriculum step instead of one chart spanning every (differently-lengthed)
        # stage.
        stage_num = self.stage_idx + 1
        step_in_stage = state.global_step - self.stage_start_step
        self._log_to_wandb(
            {
                self._metric_name(f"stage{stage_num}/train_step"): step_in_stage,
                self._metric_name(f"stage{stage_num}/train/loss"): logs["loss"],
            }
        )


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
        "dataset_seed": run_config.dataset_seed,
        "eval_steps": run_config.eval_steps,
        "logging_steps": run_config.logging_steps,
        "num_seeds": run_config.seeds,
        "solved_acc_threshold": run_config.solved_acc_threshold,
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



def _training_seeds(run_config: RunConfig) -> list[int]:
    """Seeds for the training loop: one specified seed, or ``seeds`` random draws."""
    if run_config.seed is not None:
        return [run_config.seed]
    return [secrets.randbelow(2**32) for _ in range(run_config.seeds)]


def _validate_freeze_config(run_config: RunConfig) -> None:
    if run_config.freeze_arch is None:
        return
    if run_config.model_family != "hybrid":
        raise ValueError(
            f"freeze_arch={run_config.freeze_arch!r} is only supported for hybrid models "
            f"(got model_family={run_config.model_family!r})."
        )
    if not (0.0 < run_config.freeze_fraction <= 1.0):
        raise ValueError(f"freeze_fraction must be in (0, 1], got {run_config.freeze_fraction}.")


def main(run_config: RunConfig) -> None:
    global train_length_range, test_length_ranges
    configure_logging()
    _validate_freeze_config(run_config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _ = device

    curriculum = run_config.curriculum
    stage_eval_datasets: Optional[list[dict[str, Any]]] = None
    curriculum_task_floor = 0

    # Eval bins (and formal-language train corpora) are materialized once at
    # ``run_config.dataset_seed`` and then reused for every architecture and
    # every training seed. Later ``set_seed`` calls only affect model init and
    # the on-the-fly training stream. Draw a seed if the caller left it unset
    # (CLI default) so the value can still be recorded for reproducibility.
    dataset_seed = (
        secrets.randbelow(2**32)
        if run_config.dataset_seed is None
        else run_config.dataset_seed
    )
    run_config.dataset_seed = dataset_seed
    LOGGER.info("Dataset seed: %s", dataset_seed)
    set_seed(dataset_seed)
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
    json_path = run_record_path(run_config)
    run_record = new_run_record(
        run_config,
        dataset_seed=run_config.dataset_seed,
        n_positions=n_positions,
        vocab_size=len(tokenizer),
        wandb_group=_wandb_group_for_experiment(run_config) if use_wandb else None,
    )
    write_run_record(json_path, run_record)
    LOGGER.info("Run record path: %s", json_path)

    for seed in _training_seeds(run_config):
        LOGGER.info("Training seed: %s", seed)
        if use_wandb:
            _init_wandb_run_for_seed(run_config, seed)
        set_seed(seed)

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
                        if curriculum is not None:
                            _wandb_define_curriculum_metrics(metric_prefix, curriculum)
                        else:
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

                    # Re-seed before each architecture so later sweep slots don't inherit
                    # leftover RNG from the previous Trainer, and so OLMo/non-OLMo init
                    # both see this seed. TrainingArguments.seed is required: Trainer.__init__
                    # (and train() when model_init is set) call set_seed(args.seed), which
                    # defaults to 42 and would otherwise wipe the loop seed before the first
                    # batch. Eval datasets were built once above (dataset_seed) and are the
                    # same object for every architecture; this re-seed only affects init and
                    # the streaming train iterator.
                    set_seed(seed)
                    model = build_model(run_config, arch, tokenizer, n_positions, seed=seed)
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
                        seed=seed,
                    )

                    freeze_cb: Optional[FreezeCallback] = None
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
                        if run_config.freeze_arch is not None:
                            # `phase_start`/`phase_len` read `cb`'s live state, since it
                            # re-freezes (via `cb.freeze_callback`, set below) and updates
                            # `stage_start_step` at every curriculum stage transition.
                            freeze_cb = FreezeCallback(
                                model,
                                run_config.freeze_arch,
                                run_config.freeze_fraction,
                                phase_start=lambda: cb.stage_start_step,
                                phase_len=lambda: curriculum.steps_per_stage,
                            )
                            cb.freeze_callback = freeze_cb
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
                        if run_config.freeze_arch is not None:
                            freeze_cb = FreezeCallback(
                                model,
                                run_config.freeze_arch,
                                run_config.freeze_fraction,
                                phase_start=lambda: 0,
                                phase_len=lambda max_steps=max_steps: max_steps,
                            )

                    callbacks = [cb] if freeze_cb is None else [cb, freeze_cb]
                    trainer = Trainer(
                        model=model,
                        args=training_args,
                        train_dataset=train_dataset,
                        eval_dataset=initial_eval_dataset,
                        data_collator=customCollator(tokenizer.pad_token_id),
                        compute_metrics=compute_metrics,
                        callbacks=callbacks,
                    )
                    if curriculum is not None:
                        cb.trainer = trainer
                    trainer.train()

                    wandb_run_id = None
                    if use_wandb and wandb is not None and wandb.run is not None:
                        wandb_run_id = wandb.run.id
                    run_record["runs"].append(
                        arch_run_entry(
                            run_config,
                            arch,
                            seed=seed,
                            max_steps=max_steps,
                            log_prefix=output_tag,
                            logged_snapshots=cb.reported_evals(),
                            stopped_early=bool(getattr(cb, "_stopped_early", False)),
                            wandb_run_id=wandb_run_id,
                        )
                    )
                    write_run_record(json_path, run_record)

                    if run_config.save_final_weights:
                        wpath = os.path.join(task_path, f"{output_tag}_weights_seed{seed}_id{run_config.job_id}.pt")
                        torch.save(trainer.model.state_dict(), wpath)
        finally:
            if use_wandb and wandb is not None and wandb.run is not None:
                wandb.finish()


# ---------------------------------------------------------------------------
# CLI. The model itself comes from a spec in ``model_specs/`` (--model);
# everything below is a task or training hyperparameter.
# ---------------------------------------------------------------------------
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
    parser = argparse.ArgumentParser(
        description=(
            "Train one model spec on one task. Model architecture (family, layers, "
            "NoPE, layer norm, SSM kernel, ...) lives in model_specs/*.yaml; every "
            "other hyperparameter is a flag here."
        )
    )
    parser.add_argument(
        "--model",
        "--model-specs",
        dest="model",
        type=str,
        default=None,
        help=(
            "Model spec to train, e.g. 'hybrid/olmo_sa'. Resolved inside "
            "algorithmic/model_specs/ (extension optional) or as a file path. "
            "See --list-models."
        ),
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="Print the available model specs and exit.",
    )

    parser.add_argument("--task", type=str, choices=list(ALL_TASKS))
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "Training seed (model init and the on-the-fly training stream). "
            "If set, run a single iteration with this seed. If omitted, each of "
            "the --seeds iterations draws a random seed (recorded in the run log)."
        ),
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=1,
        help=(
            "Number of training iterations, each with a randomly drawn seed. "
            "Cannot be combined with --seed, which always runs once."
        ),
    )
    parser.add_argument(
        "--dataset-seed",
        type=int,
        default=None,
        help=(
            "Seed used once to materialize eval bins (and formal-language train "
            "corpora). Independent of --seed/--seeds, which only vary model init "
            "and the training stream. If omitted, a random seed is drawn and "
            "recorded in the run log."
        ),
    )
    parser.add_argument("--job-id", type=str, default="")

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
        "--formal-packed-targets",
        action="store_true",
        help=(
            "Formal-language tasks only: serialize as '<bos> src <sep> tgt <eos>' instead of "
            "the default one-target-token-per-source-token alignment. The packed form also "
            "requires counting the source length back down to place <eos>, which fixed-state "
            "recurrent models do not extrapolate."
        ),
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

    parser.add_argument("--early-stop", action="store_true")
    parser.add_argument(
        "--solved-acc-threshold",
        type=float,
        default=None,
        help=(
            "Stop training when every eval length bin reaches this accuracy "
            "(default: 0.98). Independent of --early-stop. Set above 1.0 to disable."
        ),
    )

    parser.add_argument("--save-final-weights", action="store_true")
    parser.add_argument("--report-to", type=str, default="wandb", choices=["none", "wandb"])
    parser.add_argument("--wandb-project", type=str, default=None)
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-group", type=str, default=None)
    parser.add_argument(
        "--json-log-dir",
        type=str,
        default=None,
        help="Root directory for JSON run records (default: ./json_logs). Uses the same {task}/ layout as the text logs.",
    )

    parser.add_argument(
        "--monoid",
        type=str,
        default="parity",
        choices=["parity", "cyclic", "s5"],
        help="MQAR monoid: parity (Z_2 XOR), cyclic (Z_n addition), or s5 (S_5 composition).",
    )
    parser.add_argument("--monoid_n", type=int, default=2)
    parser.add_argument("--query-fraction-upper", type=float, default=0.2)
    parser.add_argument("--query-fraction-lower", type=float, default=0.2)

    parser.add_argument("--key-len", type=int, default=4)
    parser.add_argument("--mkar-vocab-size", type=int, default=128)
    parser.add_argument("--marker-vocab-size", type=int, default=16)
    parser.add_argument(
        "--marker-frequency",
        type=float,
        default=0.2,
        help=(
            "Selective copy only: fraction of content tokens that are numbered "
            "markers. The count is ceil(length * frequency), including the last "
            "marker, and is clamped to [1, length]. Must be in [0, 1]."
        ),
    )
    parser.add_argument(
        "--sort-vocab-size",
        type=int,
        default=None,
        help=(
            "Sort task only: number of distinct content tokens. If omitted, the "
            "vocabulary has max_test_length tokens (the current default). Raised to "
            "the maximum sequence length if smaller, so every example uses unique "
            "tokens. When set, training examples cover the full vocabulary so a "
            "total order over all tokens can be learned."
        ),
    )
    return parser


def apply_args_to_config(rc: RunConfig, args: argparse.Namespace) -> None:
    """Apply the non-model settings; the model fields already come from the spec."""
    rc.task = args.task
    if args.seeds < 1:
        raise SystemExit(f"--seeds must be >= 1, got {args.seeds}.")
    if args.seed is not None and args.seeds != 1:
        raise SystemExit("--seed always runs a single iteration; do not also pass --seeds.")
    if args.seed is not None:
        rc.seed = args.seed
        rc.seeds = 1
    else:
        rc.seed = None
        rc.seeds = args.seeds
    rc.dataset_seed = args.dataset_seed
    rc.job_id = args.job_id

    if args.freeze is not None and args.freeze_fraction is None:
        raise SystemExit("--freeze requires --freeze-fraction to also be set.")
    if args.freeze_fraction is not None:
        if not (0.0 < args.freeze_fraction <= 1.0):
            raise SystemExit(f"--freeze-fraction must be in (0, 1], got {args.freeze_fraction}.")
    if args.freeze is not None and rc.model_family != "hybrid":
        raise SystemExit(
            f"--freeze is only supported for hybrid models, but model spec {rc.model_spec!r} "
            f"has model_family={rc.model_family!r}."
        )
    rc.freeze_arch = args.freeze
    rc.freeze_fraction = args.freeze_fraction if args.freeze_fraction is not None else 0.0

    rc.train_length_range = args.train_length_range
    rc.formal_aligned_targets = not args.formal_packed_targets
    rc.early_stop = args.early_stop
    if args.solved_acc_threshold is not None:
        rc.solved_acc_threshold = args.solved_acc_threshold

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
    if args.json_log_dir is not None:
        rc.json_log_dir = args.json_log_dir

    rc.monoid = args.monoid
    rc.monoid_n = args.monoid_n
    rc.query_fraction_upper = args.query_fraction_upper
    rc.query_fraction_lower = args.query_fraction_lower

    rc.key_len = args.key_len
    rc.mkar_vocab_size = args.mkar_vocab_size
    rc.marker_vocab_size = args.marker_vocab_size
    if not (0.0 <= args.marker_frequency <= 1.0):
        raise SystemExit(f"--marker-frequency must be in [0, 1], got {args.marker_frequency}.")
    rc.marker_frequency = args.marker_frequency
    if args.sort_vocab_size is not None and args.sort_vocab_size < 1:
        raise SystemExit(f"--sort-vocab-size must be >= 1, got {args.sort_vocab_size}.")
    rc.sort_vocab_size = args.sort_vocab_size

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


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    if args.list_models:
        print("\n".join(available_model_specs()))
        raise SystemExit(0)
    if args.model is None:
        parser.error("--model is required; run with --list-models to see the available specs.")
    if args.task is None:
        parser.error("--task is required.")
    try:
        run_config = load_model_spec(args.model)
    except ValueError as exc:
        parser.error(str(exc))
    apply_args_to_config(run_config, args)
    main(run_config)
