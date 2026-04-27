#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
import ast
from dataclasses import dataclass
from typing import Any, Iterable


def _maybe_int_list(x: Any) -> list[int]:
    if x is None:
        return []
    if isinstance(x, list):
        return [int(v) for v in x]
    return [int(v) for v in x]


def _split_on_sep(tokens: list[str], sep_token: str) -> list[list[str]]:
    parts: list[list[str]] = [[]]
    for t in tokens:
        if t == sep_token:
            parts.append([])
        else:
            parts[-1].append(t)
    return parts


def _format_masked_label(label_tokens: list[str], pad_token: str) -> str:
    # Replace pad-masked targets with "·" so answer region stands out.
    return " ".join(("·" if t == pad_token else t) for t in label_tokens)


def _normalize_task_name(task: str) -> str:
    """
    Accept common formal task aliases and convert to internal names used by
    algorithmic/dataset_generators_formal.py.
    """
    t = task.strip().lower().replace("-", "_")
    aliases = {
        "tomita1": "tomita_1",
        "tomita2": "tomita_2",
        "tomita3": "tomita_3",
        "tomita4": "tomita_4",
        "tomita5": "tomita_5",
        "tomita6": "tomita_6",
        "tomita7": "tomita_7",
        "d2": "d_2",
        "d3": "d_3",
        "d4": "d_4",
        "d12": "d_12",
        "012star_0_2star": "012_star_0_2_star",
    }
    return aliases.get(t, t)


@dataclass(frozen=True)
class SampleView:
    input_ids: list[int]
    pos_ids: list[int]
    labels: list[int]
    tokens: list[str]
    label_tokens: list[str]


def _to_sample_view(dataset: Any, sample: Any) -> SampleView:
    # Datasets in algorithmic/dataset_generators.py yield (instance, pos_ids, label)
    input_ids, pos_ids, labels = sample
    input_ids = _maybe_int_list(input_ids)
    pos_ids = _maybe_int_list(pos_ids)
    labels = _maybe_int_list(labels)

    tok = getattr(dataset, "tokenizer", None)
    if tok is None or not hasattr(tok, "convert_ids_to_tokens"):
        raise TypeError("Dataset has no compatible `tokenizer.convert_ids_to_tokens()`.")

    tokens = tok.convert_ids_to_tokens(input_ids, rm_special=False)
    label_tokens = tok.convert_ids_to_tokens(labels, rm_special=False)
    return SampleView(
        input_ids=input_ids,
        pos_ids=pos_ids,
        labels=labels,
        tokens=tokens,
        label_tokens=label_tokens,
    )


def _pretty_print(dataset: Any, sv: SampleView, idx: int, *, show_ids: bool, show_pos: bool) -> None:
    tok = dataset.tokenizer
    sep = tok.sep_token
    pad = tok.pad_token

    print("=" * 88)
    print(f"sample {idx}")
    print(f"len(input_ids)={len(sv.input_ids)}")

    if show_ids:
        print("\ninput_ids:")
        print(" ".join(map(str, sv.input_ids)))
        print("\nlabels (pad-masked):")
        print(" ".join(map(str, sv.labels)))

    if show_pos:
        print("\npos_ids:")
        print(" ".join(map(str, sv.pos_ids)))

    print("\ntokens:")
    print(" ".join(sv.tokens))

    parts = _split_on_sep(sv.tokens, sep)
    if len(parts) >= 2:
        print("\nsegments (split on <sep>):")
        for i, p in enumerate(parts):
            print(f"  [{i}] " + " ".join(p))

    print("\nlabel tokens (· = masked):")
    print(_format_masked_label(sv.label_tokens, pad))

    # Heuristic: show last 12 tokens and label tokens for quick inspection.
    k = 12
    tail_tokens = sv.tokens[-k:] if len(sv.tokens) >= k else sv.tokens
    tail_labels = sv.label_tokens[-k:] if len(sv.label_tokens) >= k else sv.label_tokens
    print(f"\ntail[-{k}:] tokens:")
    print(" ".join(tail_tokens))
    print(f"tail[-{k}:] label tokens:")
    print(_format_masked_label(tail_labels, pad))


def _load_dataset_class(name: str):
    # Make `import algorithmic...` work even when running this script directly.
    # This file lives in `algorithmic/convenience_scripts/`, so:
    # parents[0] = convenience_scripts, parents[1] = algorithmic, parents[2] = repo root.
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    try:
        mod = importlib.import_module("algorithmic.dataset_generators")
    except ModuleNotFoundError as e:
        if e.name == "torch":
            raise SystemExit(
                "Failed to import `torch` (required by algorithmic/dataset_generators.py).\n"
                "Run this script inside the project's environment where PyTorch is installed "
                "(e.g. after `uv pip install -e '.[all]'` or your usual setup)."
            ) from e
        raise
    if not hasattr(mod, name):
        raise SystemExit(f"Unknown dataset class `{name}` in algorithmic/dataset_generators.py")
    return getattr(mod, name)


def _load_dataset_class_from_module(module_name: str, class_name: str):
    mod = _import_module_or_exit(module_name)
    if not hasattr(mod, class_name):
        raise SystemExit(f"Unknown dataset class `{class_name}` in {module_name}.")
    return getattr(mod, class_name)


def _load_build_datasets(module_name: str):
    mod = _import_module_or_exit(module_name)
    if not hasattr(mod, "build_datasets"):
        raise SystemExit(f"Module {module_name} has no build_datasets().")
    return getattr(mod, "build_datasets")


def _import_module_or_exit(module_name: str):
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as e:
        if e.name in {"torch", "numpy"}:
            raise SystemExit(
                f"Failed to import dependency `{e.name}` required by {module_name}.\n"
                "Run this script in the project environment where dependencies are installed "
                "(e.g. `uv pip install -e '.[all]'`)."
            ) from e
        raise


def _make_run_config(*, task: str, train_range: tuple[int, int], num_test_bins: int, test_num: int):
    utils = importlib.import_module("algorithmic.utils")
    run_config = utils.default_transformer_sweep()
    run_config.task = task
    run_config.train_length_range = train_range
    run_config.num_test_bins = num_test_bins
    run_config.test_num = test_num
    return run_config


def _iter_samples(d: Any) -> Iterable[Any]:
    # IterableDataset: iter(d) works. Dataset: also works but may be finite.
    return iter(d)


def main() -> int:
    p = argparse.ArgumentParser(
        description=(
            "Generate and print samples from algorithmic datasets one-by-one for debugging. "
            "Supports direct dataset classes and build_datasets(task)-based inspection."
        )
    )
    p.add_argument(
        "--module",
        default="algorithmic.dataset_generators",
        help=(
            "Python module containing dataset classes/build_datasets. "
            "Use algorithmic.dataset_generators_formal for formal languages."
        ),
    )
    p.add_argument(
        "--mode",
        choices=["class", "build"],
        default="class",
        help=(
            "`class`: instantiate --dataset with --dataset-kwargs. "
            "`build`: call build_datasets(run_config) and inspect the train/test streams."
        ),
    )
    p.add_argument(
        "--dataset",
        default="MQARWordProblemDataset",
        help="Dataset class name for --mode class.",
    )
    # Accept both correct flag and common typo for convenience.
    p.add_argument(
        "--dataset-kwargs",
        "--datset-kwargs",
        dest="dataset_kwargs",
        default="{}",
        help=(
            "Kwargs passed to the dataset constructor. "
            "Accepts JSON (recommended) or a Python dict literal."
        ),
    )
    p.add_argument("--num", type=int, default=20, help="Number of samples to print.")
    p.add_argument("--seed", type=int, default=None, help="Seed python's RNG via random.seed().")
    p.add_argument("--show-ids", action="store_true", help="Print raw input_ids and labels.")
    p.add_argument("--show-pos", action="store_true", help="Print pos_ids.")
    p.add_argument("--task", default="parity", help="Task name for --mode build.")
    p.add_argument(
        "--train-range",
        default="[0,50]",
        help="Train length range for --mode build. JSON list or Python tuple/list, e.g. [0,50].",
    )
    p.add_argument("--num-test-bins", type=int, default=3, help="Number of test bins for --mode build.")
    p.add_argument("--test-num", type=int, default=2000, help="Examples per test bin in --mode build.")
    p.add_argument(
        "--split",
        choices=["train", "test"],
        default="train",
        help="Which split to sample from in --mode build.",
    )
    p.add_argument(
        "--test-bin",
        default=None,
        help=(
            "Test bin key for --split test in --mode build "
            '(e.g. "len0-49"). If omitted, first bin is used.'
        ),
    )
    p.add_argument(
        "--no-pause",
        action="store_true",
        help="Do not pause between samples (useful for piping/redirecting).",
    )
    args = p.parse_args()

    if args.seed is not None:
        import random

        random.seed(args.seed)

    raw_kwargs = args.dataset_kwargs
    try:
        dataset_kwargs = json.loads(raw_kwargs)
    except json.JSONDecodeError:
        try:
            dataset_kwargs = ast.literal_eval(raw_kwargs)
        except Exception as e:
            raise SystemExit(
                "Failed to parse --dataset-kwargs/--datset-kwargs.\n"
                "Pass a single shell argument (quote it) containing either JSON or a Python dict literal.\n"
                'Example JSON: \'{"length_range":[20,30],"max_test_length":100}\'\n'
                'Example Python: \'{"length_range": (20, 30), "max_test_length": 100}\'\n'
                f"Got: {raw_kwargs!r}\n"
                f"Parse error: {e}"
            ) from e
    if not isinstance(dataset_kwargs, dict):
        raise SystemExit(f"dataset kwargs must be a dict, got {type(dataset_kwargs).__name__}")

    # Make `import algorithmic...` work even when running this script directly.
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    if args.mode == "class":
        if args.module == "algorithmic.dataset_generators":
            DatasetCls = _load_dataset_class(args.dataset)
        else:
            DatasetCls = _load_dataset_class_from_module(args.module, args.dataset)
        d = DatasetCls(**dataset_kwargs)
    else:
        raw_train_range = args.train_range
        try:
            train_range_val = json.loads(raw_train_range)
        except json.JSONDecodeError:
            try:
                train_range_val = ast.literal_eval(raw_train_range)
            except Exception as e:
                raise SystemExit(
                    "Failed to parse --train-range.\n"
                    "Pass JSON/Python list or tuple, e.g. [0,50] or (0,50).\n"
                    f"Got: {raw_train_range!r}\n"
                    f"Parse error: {e}"
                ) from e
        if not isinstance(train_range_val, (list, tuple)) or len(train_range_val) != 2:
            raise SystemExit(f"--train-range must be length-2 list/tuple, got: {train_range_val!r}")
        train_range = (int(train_range_val[0]), int(train_range_val[1]))

        task = _normalize_task_name(args.task)
        build_datasets = _load_build_datasets(args.module)
        run_config = _make_run_config(
            task=task,
            train_range=train_range,
            num_test_bins=args.num_test_bins,
            test_num=args.test_num,
        )
        train_dataset, test_dataset, _, _ = build_datasets(run_config)

        if args.split == "train":
            d = train_dataset
            print(f"[build mode] module={args.module} task={task} split=train")
        else:
            if not test_dataset:
                raise SystemExit("Test dataset dict is empty.")
            if args.test_bin is None:
                bin_key = next(iter(test_dataset.keys()))
            else:
                bin_key = args.test_bin
                if bin_key not in test_dataset:
                    available = ", ".join(test_dataset.keys())
                    raise SystemExit(f"Unknown --test-bin {bin_key!r}. Available: {available}")
            d = test_dataset[bin_key]
            print(f"[build mode] module={args.module} task={task} split=test bin={bin_key}")

    it = _iter_samples(d)
    for i in range(args.num):
        sample = next(it)
        sv = _to_sample_view(d, sample)
        _pretty_print(d, sv, i, show_ids=args.show_ids, show_pos=args.show_pos)

        if args.no_pause:
            continue

        try:
            s = input("\n[enter]=next, q=quit > ").strip().lower()
        except EOFError:
            break
        if s in {"q", "quit", "exit"}:
            break

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

