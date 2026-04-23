from __future__ import annotations

import argparse
import logging
import os
import random
import sys
from typing import Any

import numpy as np
import torch
from transformers import Trainer, TrainerCallback, TrainingArguments

from dataset_generators import build_datasets
from models import build_model
from utils import ArchSlot, RunConfig, default_hybrid_sweep, default_ssm_sweep, default_transformer_sweep

LOGGER = logging.getLogger(__name__)


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

    parts: list[str] = []

    if run_config.model_family == "transformer":
        pe = "nope" if run_config.use_nope else "pe"
        parts += [
            "lm",
            f"{run_config.regularize}reg",
            f"{arch.n_layer}l",
            f"{arch.n_head}h",
            f"{arch.d_model}d",
            f"{arch.between_block_mlp_layers}mlp",
            f"{pe}",
            f"{ln_str}",
        ]
    elif run_config.model_family == "ssm":
        parts += [
            "ssm",
            f"{run_config.ssm_kernel}",
            f"{arch.n_layer}l",
            f"{arch.d_model}d",
            f"{arch.dropout}dr",
            f"{arch.between_block_mlp_layers}mlp",
            f"{ln_str}",
        ]
    else:
        pat = run_config.hybrid_layer_pattern
        pe = "nope" if run_config.use_nope else "pe"
        parts += [
            "hyb",
            f"{pat}",
            f"{run_config.ssm_kernel}",
            f"{arch.n_layer}l",
            f"{arch.n_head}h",
            f"{arch.d_model}d",
            f"{arch.dropout}dr",
            f"{arch.between_block_mlp_layers}mlp",
            f"{pe}",
            f"{ln_str}",
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

    def on_evaluate(self, state, args, control, metrics=None, logs=None, eval_dataloader=None, **kwargs):
        metrics = metrics or {}
        assert metrics["epoch"] >= self.current_epoch
        if metrics["epoch"] > self.current_epoch:
            self.latest_acc = {}
            self.current_epoch = metrics["epoch"]
        for key in metrics.keys():
            if key.endswith("acc"):
                self.latest_acc[key] = metrics[key]
        if len(self.latest_acc) == len(self.test_length_ranges):
            tr = self.train_length_range
            train_key = f"eval_len{tr[0]}-{tr[1]}_acc"
            if (self.latest_acc.get(train_key) == 1.0) or (self.current_epoch == 1.0):
                if self.latest_acc.get(train_key) == 1.0:
                    control.should_training_stop = True
                    self.stop_state["fit_train_data"] = True
                    msg = f"early stop {self.current_epoch}\t\t"
                else:
                    msg = "reach max step\t\t"
                if self.latest_acc.get(train_key, 0) >= 0.99:
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

                mid = self.test_length_ranges[1]
                mid_key = f"eval_len{mid[0]}-{mid[1]}_acc"
                if self.latest_acc.get(train_key) == 1.0 and self.latest_acc.get(mid_key) == 1.0:
                    self.stop_state["should_stop"] = True


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

    for seed in range(run_config.seeds):
        torch.manual_seed(seed)
        random.seed(seed)

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
                if max_steps == run_config.max_steps_large and stop_state["fit_train_data"]:
                    break

                output_tag = format_log_prefix(run_config, arch, max_steps)

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
                    report_to=run_config.report_to,
                )

                cb = AlgorithmicTrainCallback(
                    run_config,
                    arch,
                    train_length_range,
                    test_length_ranges,
                    summary_file,
                    max_steps,
                    stop_state,
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

                if stop_state["should_stop"]:
                    break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train with a preset RunConfig from utils.py (edit there or pass --preset).")
    parser.add_argument(
        "--preset",
        choices=["transformer", "ssm", "hybrid"],
        default="transformer",
        help="Which default RunConfig factory to use; full settings live in utils.RunConfig.",
    )
    args = parser.parse_args()
    presets = {
        "transformer": default_transformer_sweep,
        "ssm": default_ssm_sweep,
        "hybrid": default_hybrid_sweep,
    }
    main(presets[args.preset]())
