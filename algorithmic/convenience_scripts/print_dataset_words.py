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


def _iter_samples(d: Any) -> Iterable[Any]:
    # IterableDataset: iter(d) works. Dataset: also works but may be finite.
    return iter(d)


def main() -> int:
    p = argparse.ArgumentParser(
        description=(
            "Generate and print samples from datasets in algorithmic/dataset_generators.py "
            "one-by-one for debugging."
        )
    )
    p.add_argument(
        "--dataset",
        default="MQARWordProblemDataset",
        help="Dataset class name from algorithmic/dataset_generators.py",
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

    DatasetCls = _load_dataset_class(args.dataset)
    d = DatasetCls(**dataset_kwargs)

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

