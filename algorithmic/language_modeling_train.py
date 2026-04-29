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

from dataset_generators import build_datasets
from models import build_model
from utils import ArchSlot, RunConfig, default_hybrid_sweep, default_ssm_sweep, default_transformer_sweep

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

    train_dataset, test_dataset, train_length_range, test_length_ranges = build_datasets(run_config)
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
                first_range = test_length_ranges[0]
                key0 = f"len{first_range[0]}-{first_range[1]}"
                for i in range(run_config.print_example_sequences):
                    print("\ninput example:", flush=True)
                    print(" ".join(tokenizer.convert_ids_to_tokens(test_dataset[key0][i][0])), flush=True)
                    print("label example:", flush=True)
                    print(" ".join(tokenizer.convert_ids_to_tokens(test_dataset[key0][i][2])), flush=True)

                stop_state: dict[str, Any] = {"should_stop": False, "fit_train_data": False}

                for arch in run_config.architectures:
                    max_steps, warmup_steps = _max_steps_warmup(run_config, arch)

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
                        eval_steps=run_config.eval_steps,
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
                        eval_dataset=test_dataset,
                        data_collator=customCollator(tokenizer.pad_token_id),
                        compute_metrics=compute_metrics,
                        callbacks=[cb],
                    )
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
