"""Machine-readable per-job run records (JSON) for reproducibility.

Text summaries still go to ``{log_dir}/{task}/summary*.txt``. The JSON record
mirrors that layout under ``{json_log_dir}/{task}/summary*.json`` and stores the
full ``RunConfig``, plus one entry per (seed, architecture). Each entry's
``logged_snapshots`` is the same list of evals written to the summary file
(Pareto per-bin maxima, so at most one snapshot per length bin).
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, is_dataclass
from typing import Any, Optional

from utils import ArchSlot, RunConfig

# RunConfig attributes that must appear in the JSON record. Class-level attributes
# (no annotation) are omitted by ``asdict``; dataclass fields listed here are copied
# again so task-specific flags such as ``sort_vocab_size`` cannot be dropped.
_RUNCONFIG_CLASS_ATTRS = (
    "key_len",
    "mkar_vocab_size",
    "marker_vocab_size",
    "misc_vocab_size",
    "marker_frequency",
    "early_stop",
    "sort_vocab_size",
)


def jsonable(obj: Any) -> Any:
    """Recursively convert dataclasses/tuples into JSON-friendly values."""
    if is_dataclass(obj) and not isinstance(obj, type):
        return jsonable(asdict(obj))
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, tuple):
        return [jsonable(v) for v in obj]
    if isinstance(obj, list):
        return [jsonable(v) for v in obj]
    if isinstance(obj, (float, int)) and not isinstance(obj, bool):
        return obj
    if type(obj).__module__ == "numpy" and hasattr(obj, "item"):
        try:
            return obj.item()
        except (ValueError, AttributeError):
            pass
    return obj


def accs_jsonable(accs: dict[str, Any]) -> dict[str, float]:
    return {str(k): float(v) for k, v in accs.items()}


def run_config_to_dict(run_config: RunConfig) -> dict[str, Any]:
    """Complete config, including class-level fields and derived test bins."""
    data = asdict(run_config)
    for name in _RUNCONFIG_CLASS_ATTRS:
        data[name] = getattr(run_config, name)
    data["test_length_ranges"] = list(run_config.test_length_ranges)
    return jsonable(data)


def summary_stem(run_config: RunConfig) -> str:
    """Basename (no extension) shared by the text summary and JSON record."""
    if run_config.model_family == "transformer":
        if run_config.use_nope:
            mid = "-nope"
        elif run_config.regularize != 0:
            mid = f"-reg{run_config.regularize}"
        else:
            mid = ""
        return f"summarylm{mid}{run_config.job_id}"
    if run_config.model_family == "ssm":
        return f"summaryssm{run_config.job_id}"
    return f"summaryhybrid{run_config.job_id}"


def summary_rel_path(run_config: RunConfig) -> str:
    return f"{summary_stem(run_config)}.txt"


def run_record_rel_path(run_config: RunConfig) -> str:
    """Filename matching the text summary, with a ``.json`` suffix."""
    return f"{summary_stem(run_config)}.json"


def run_record_path(run_config: RunConfig) -> str:
    return os.path.join(run_config.json_log_dir, run_config.task, run_record_rel_path(run_config))


def new_run_record(
    run_config: RunConfig,
    *,
    dataset_seed: int,
    n_positions: int,
    vocab_size: int,
    wandb_group: Optional[str] = None,
) -> dict[str, Any]:
    return {
        "job_id": run_config.job_id,
        "dataset_seed": dataset_seed,
        "wandb_group": wandb_group,
        "config": run_config_to_dict(run_config),
        "runtime": {
            "n_positions": n_positions,
            "vocab_size": vocab_size,
        },
        "runs": [],
    }


def arch_run_entry(
    run_config: RunConfig,
    arch: ArchSlot,
    *,
    seed: int,
    max_steps: int,
    log_prefix: str,
    logged_snapshots: list[dict[str, Any]],
    stopped_early: bool = False,
    wandb_run_id: Optional[str] = None,
) -> dict[str, Any]:
    """One (seed, architecture) slot.

    ``logged_snapshots`` is the same list written to the text summary: each item
    is one complete eval (all length bins), and the list is the Pareto front of
    per-bin maxima — so its length is at most the number of eval bins.
    """
    return {
        "seed": seed,
        "job_id": run_config.job_id,
        "wandb_run_id": wandb_run_id,
        "architecture": jsonable(arch),
        "log_prefix": log_prefix,
        "max_steps": max_steps,
        "stopped_early": stopped_early,
        "logged_snapshots": jsonable(logged_snapshots),
    }


def write_run_record(path: str, record: dict[str, Any]) -> None:
    """Atomically write ``record`` as pretty-printed JSON."""
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(jsonable(record), handle, indent=2)
        handle.write("\n")
    os.replace(tmp_path, path)
