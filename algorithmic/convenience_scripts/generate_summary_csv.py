#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean

LINE_RE = re.compile(
    r"^(?P<model>\S+)\s+.*?"
    r"eval_len0-50_acc:\s*(?P<acc_0_50>[0-9]*\.?[0-9]+)\s+"
    r"eval_len51-100_acc:\s*(?P<acc_51_100>[0-9]*\.?[0-9]+)\s+"
    r"eval_len101-150_acc:\s*(?P<acc_101_150>[0-9]*\.?[0-9]+)\s+"
    r"lr:\s*(?P<lr>[0-9]*\.?[0-9]+(?:e-?[0-9]+)?)\s*$"
)

BUCKETS = ("eval_len0-50", "eval_len51-100", "eval_len101-150")

CSV_COLUMNS = [
    "task",
    # "source_file",
    # "source_line",
    "model",
    "learning_rate",
    "bucket",
    "accuracy",
]


def parse_summary_line(line: str, task: str) -> list[dict[str, str | int | float]]:
    m = LINE_RE.match(line.strip())
    if m is None:
        return []

    model = m.group("model")
    lr = float(m.group("lr"))
    acc_0_50 = float(m.group("acc_0_50"))
    acc_51_100 = float(m.group("acc_51_100"))
    acc_101_150 = float(m.group("acc_101_150"))

    return [
        {
            "task": task,
            # "source_file": str(source_file),
            # "source_line": source_line,
            "model": model,
            "learning_rate": lr,
            "bucket": "eval_len0-50",
            "accuracy": acc_0_50,
        },
        {
            "task": task,
            # "source_file": str(source_file),
            # "source_line": source_line,
            "model": model,
            "learning_rate": lr,
            "bucket": "eval_len51-100",
            "accuracy": acc_51_100,
        },
        {
            "task": task,
            # "source_file": str(source_file),
            # "source_line": source_line,
            "model": model,
            "learning_rate": lr,
            "bucket": "eval_len101-150",
            "accuracy": acc_101_150,
        },
    ]


def load_csv_rows(csv_path: Path) -> list[dict[str, str]]:
    if not csv_path.exists():
        return []
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def write_csv_rows(csv_path: Path, rows: list[dict[str, str | int | float]]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_or_update_csv(logs_root: Path, csv_path: Path) -> list[dict[str, str]]:
    existing_rows = load_csv_rows(csv_path)

    seen_keys = set()
    merged_rows: list[dict[str, str | int | float]] = []

    for row in existing_rows:
        key = (
            # row["source_file"],
            # int(row["source_line"]),
            row["bucket"],
            row["model"],
            float(row["learning_rate"]),
            row["task"],
        )
        if key in seen_keys:
            continue
        seen_keys.add(key)
        merged_rows.append({col: row[col] for col in CSV_COLUMNS})

    summary_files = sorted(logs_root.glob("**/summary*.txt"))
    for summary_file in summary_files:
        task = summary_file.parent.name
        with summary_file.open("r") as f:
            for raw_line in f:
                parsed_rows = parse_summary_line(
                    raw_line, task=task
                )
                for row in parsed_rows:
                    key = (
                        # str(row["source_file"]),
                        # int(row["source_line"]),
                        str(row["bucket"]),
                        str(row["model"]),
                        float(row["learning_rate"]),
                        str(row["task"]),
                    )
                    if key in seen_keys:
                        continue
                    seen_keys.add(key)
                    merged_rows.append(row)

    write_csv_rows(csv_path, merged_rows)
    return load_csv_rows(csv_path)


def parse_include_rule(include_value: str) -> tuple[str, set[float] | None]:
    if ":" in include_value:
        model, lr_part = include_value.split(":", 1)
        model = model.strip()
        lr_part = lr_part.strip()
        if lr_part == "*" or lr_part == "":
            return model, None
        lr_values = {float(x.strip()) for x in lr_part.split(",") if x.strip()}
        return model, lr_values
    return include_value.strip(), None


def extract_feature_tokens(value: str) -> list[tuple[str, str]]:
    tokens = re.findall(
        r"[0-9]+(?:\.[0-9]+)?[a-z]+|[a-z]+[0-9]+(?:\.[0-9]+)?",
        value.lower(),
    )
    features: list[tuple[str, str]] = []
    for token in tokens:
        m_alpha_num = re.fullmatch(r"([a-z]+)([0-9]+(?:\.[0-9]+)?)", token)
        if m_alpha_num:
            features.append((m_alpha_num.group(1), m_alpha_num.group(2)))
            continue
        m_num_alpha = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)([a-z]+)", token)
        if m_num_alpha:
            features.append((m_num_alpha.group(2), m_num_alpha.group(1)))
    return features


def matches_pattern(model: str, pattern: str) -> bool:
    model_l = model.lower()
    pattern_l = pattern.lower().strip()
    if not pattern_l:
        return False

    # Parse pattern with the same non-overlapping chunk logic used for models.
    # This handles cases like "ssm16d" as ["ssm", "16d"].
    pattern_chunks = re.findall(r"[0-9]+(?:\.[0-9]+)?[a-z]+|[a-z]+", pattern_l)
    pattern_features = extract_feature_tokens(" ".join(pattern_chunks))
    # Model specs are concatenated chunks of {number?}{string}, e.g. ssm4l256d0.1dr.
    # Parse chunks this way so "4l" is recognized in "ssm4l..." (non-overlapping).
    model_chunks = re.findall(r"[0-9]+(?:\.[0-9]+)?[a-z]+|[a-z]+", model_l)
    model_features = set(extract_feature_tokens(" ".join(model_chunks)))
    for feat in pattern_features:
        if feat not in model_features:
            return False

    leftovers = re.sub(
        r"[a-z]+[0-9]+(?:\.[0-9]+)?|[0-9]+(?:\.[0-9]+)?[a-z]+",
        "",
        pattern_l,
    ).strip()
    if leftovers:
        return leftovers in model_l

    if not pattern_features:
        return pattern_l in model_l
    return True


def parse_tasks(task_args: list[str]) -> list[str]:
    tasks: list[str] = []
    for raw in task_args:
        parts = [p.strip() for p in raw.split(",")]
        tasks.extend([p for p in parts if p])
    return tasks


def filter_rows(
    rows: list[dict[str, str]],
    tasks: list[str],
    include_rules: list[tuple[str, set[float] | None]],
    include_patterns: list[str],
) -> list[dict[str, str]]:
    if tasks:
        task_set = set(tasks)
        rows = [r for r in rows if r["task"] in task_set]

    if not include_rules and not include_patterns:
        return rows

    expanded_patterns: list[str] = []
    for pattern in include_patterns:
        # Allow user input like "8l and 4l" or "8l,4l" as OR patterns.
        parts = [
            p.strip()
            for p in re.split(r"(?:\s+and\s+)|[,\s]+", pattern.lower())
            if p.strip()
        ]
        expanded_patterns.extend([p for p in parts if p != "and"])

    filtered: list[dict[str, str]] = []
    for row in rows:
        model = row["model"]
        lr = float(row["learning_rate"])

        matched_rule = False
        for m_name, lrs in include_rules:
            if model != m_name:
                continue
            if lrs is None or lr in lrs:
                matched_rule = True
                break

        matched_pattern = any(matches_pattern(model, p) for p in expanded_patterns)
        if matched_rule or matched_pattern:
            filtered.append(row)
    return filtered


def print_models(rows: list[dict[str, str]]) -> None:
    grouped: dict[tuple[str, str, float, str], list[float]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                row["task"],
                row["model"],
                float(row["learning_rate"]),
                row["bucket"],
            )
        ].append(float(row["accuracy"]))

    series_keys = sorted({(k[0], k[1], k[2]) for k in grouped.keys()})
    if not series_keys:
        print("No matching rows.")
        return

    by_task: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for task, model, lr in series_keys:
        by_task[task].append((model, lr))

    for task in sorted(by_task.keys()):
        print(f"\nTask: {task}")
        for model, lr in sorted(by_task[task], key=lambda x: (x[0], x[1])):
            means: list[str] = []
            counts: list[str] = []
            for bucket in BUCKETS:
                vals = grouped.get((task, model, lr, bucket), [])
                if vals:
                    means.append(f"{mean(vals):.4f}")
                    counts.append(str(len(vals)))
                else:
                    means.append("nan")
                    counts.append("0")
            print(
                f"- {model} | lr={lr:g} | "
                f"len0-50={means[0]} (n={counts[0]}) | "
                f"len51-100={means[1]} (n={counts[1]}) | "
                f"len101-150={means[2]} (n={counts[2]})"
            )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build/update unified CSV from logs/**/summary*.txt and optionally list "
            "models/results filtered by tasks and include patterns."
        )
    )
    repo_root = Path(__file__).resolve().parents[2]
    default_logs_root = repo_root / "logs"
    default_csv = repo_root / "exports" / "all_summary_results.csv"

    parser.add_argument("--logs-root", type=Path, default=default_logs_root)
    parser.add_argument("--csv", type=Path, default=default_csv)
    parser.add_argument("--create-csv", action="store_true")
    
    parser.add_argument(
        "--task",
        action="append",
        default=[],
        help=(
            "Task filter for listing. Repeat flag or pass comma-separated list. "
            "Example: --task bin_majority --task majority,mqar"
        ),
    )
    parser.add_argument(
        "--include",
        action="append",
        default=[],
        help=(
            "Exact model/lr filter for listing. Format: model or model:lr1,lr2 "
            "(repeat flag for multiple models)."
        ),
    )
    parser.add_argument(
        "--include-pattern",
        action="append",
        default=[],
        help=(
            "Pattern-based OR filter over model strings. Feature matching is "
            "order-insensitive (e.g., l1h1 and h1l1 are equivalent)."
        ),
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="Print matching models and per-bin mean results from the CSV.",
    )
    args = parser.parse_args()

    if args.create_csv:
        rows = build_or_update_csv(logs_root=args.logs_root, csv_path=args.csv)
        print(f"Wrote/updated CSV: {args.csv} ({len(rows)} rows)")
        return 0

    if not args.list_models:
        return 0

    rows = load_csv_rows(args.csv)
    include_rules = [parse_include_rule(v) for v in args.include]
    tasks = parse_tasks(args.task)
    filtered_rows = filter_rows(
        rows=rows,
        tasks=tasks,
        include_rules=include_rules,
        include_patterns=args.include_pattern,
    )
    print_models(filtered_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
