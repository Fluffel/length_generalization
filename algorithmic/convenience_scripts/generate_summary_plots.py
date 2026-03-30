#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
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
    "source_file",
    "source_line",
    "model",
    "learning_rate",
    "bucket",
    "accuracy",
]


def parse_summary_line(line: str, task: str, source_file: Path, source_line: int) -> list[dict[str, str | int | float]]:
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
            "source_file": str(source_file),
            "source_line": source_line,
            "model": model,
            "learning_rate": lr,
            "bucket": "eval_len0-50",
            "accuracy": acc_0_50,
        },
        {
            "task": task,
            "source_file": str(source_file),
            "source_line": source_line,
            "model": model,
            "learning_rate": lr,
            "bucket": "eval_len51-100",
            "accuracy": acc_51_100,
        },
        {
            "task": task,
            "source_file": str(source_file),
            "source_line": source_line,
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
            row["source_file"],
            int(row["source_line"]),
            row["bucket"],
            row["model"],
            float(row["learning_rate"]),
            row["task"],
        )
        if key in seen_keys:
            continue
        seen_keys.add(key)
        merged_rows.append(row)

    summary_files = sorted(logs_root.glob("**/summary.txt"))
    for summary_file in summary_files:
        task = summary_file.parent.name
        with summary_file.open("r") as f:
            for idx, raw_line in enumerate(f, start=1):
                parsed_rows = parse_summary_line(
                    raw_line, task=task, source_file=summary_file, source_line=idx
                )
                for row in parsed_rows:
                    key = (
                        str(row["source_file"]),
                        int(row["source_line"]),
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


def _parse_include_rule(include_value: str) -> tuple[str, set[float] | None]:
    if ":" in include_value:
        model, lr_part = include_value.split(":", 1)
        model = model.strip()
        lr_part = lr_part.strip()
        if lr_part == "*" or lr_part == "":
            return model, None
        lr_values = {float(x.strip()) for x in lr_part.split(",") if x.strip()}
        return model, lr_values
    return include_value.strip(), None


def _sample_std(values: list[float]) -> float:
    n = len(values)
    if n <= 1:
        return 0.0
    mu = mean(values)
    var = sum((v - mu) ** 2 for v in values) / (n - 1)
    return math.sqrt(var)


def _filter_rows(
    rows: list[dict[str, str]],
    task: str,
    include_rules: list[tuple[str, set[float] | None]],
) -> list[dict[str, str]]:
    task_rows = [r for r in rows if r["task"] == task]
    if not include_rules:
        return task_rows

    filtered = []
    for row in task_rows:
        model = row["model"]
        lr = float(row["learning_rate"])
        matched = False
        for m_name, lrs in include_rules:
            if model != m_name:
                continue
            if lrs is None or lr in lrs:
                matched = True
                break
        if matched:
            filtered.append(row)
    return filtered


def plot_task(
    rows: list[dict[str, str]],
    task: str,
    output_path: Path,
    include_rules: list[tuple[str, set[float] | None]],
) -> None:
    try:
        import matplotlib as mpl
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ModuleNotFoundError as e:
        raise SystemExit(
            "Plotting requires matplotlib and seaborn. Install them in your environment first."
        ) from e

    mpl.rcParams["axes.titleweight"] = "bold"
    sns.set_theme(style="whitegrid", palette="dark6", context="paper", font_scale=2.0)

    filtered_rows = _filter_rows(rows, task=task, include_rules=include_rules)
    if not filtered_rows:
        raise SystemExit(
            f"No rows to plot for task '{task}'. "
            "Check --task and --include filters."
        )

    grouped: dict[tuple[str, float, str], list[float]] = defaultdict(list)
    for row in filtered_rows:
        grouped[(row["model"], float(row["learning_rate"]), row["bucket"])].append(
            float(row["accuracy"])
        )

    series_keys = sorted({(k[0], k[1]) for k in grouped.keys()}, key=lambda x: (x[0], x[1]))
    if not series_keys:
        raise SystemExit("No model/lr series found after filtering.")

    fig, ax = plt.subplots(figsize=(10, 6))
    x_ticks = [10, 20, 30]
    x_labels = ["Bin 1", "Bin 2", "Bin 3"]
    bucket_to_x = dict(zip(BUCKETS, x_ticks))

    for model, lr in series_keys:
        means: list[float] = []
        stds: list[float] = []
        for bucket in BUCKETS:
            vals = grouped.get((model, lr, bucket), [])
            if vals:
                means.append(mean(vals) * 100.0)
                stds.append(_sample_std(vals) * 100.0)
            else:
                means.append(float("nan"))
                stds.append(float("nan"))

        label = f"{model} | lr={lr:g}"
        xs = [bucket_to_x[b] for b in BUCKETS]
        ax.errorbar(
            xs,
            means,
            yerr=stds,
            marker="o",
            linestyle="solid",
            linewidth=2.0,
            capsize=5,
            elinewidth=1.5,
            label=label,
        )

    ax.set_title(f"{task}")
    ax.set_xlabel("Validation bins")
    ax.set_ylabel("Accuracy (%)")
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels)
    ax.set_ylim(-10.0, 110.0)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=10, frameon=False)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build/update a unified CSV from logs/**/summary.txt and create per-task "
            "plots with optional model/lr filtering."
        )
    )
    repo_root = Path(__file__).resolve().parents[2]
    default_logs_root = repo_root / "logs"
    default_csv = repo_root / "exports" / "all_summary_results.csv"
    default_plot_dir = repo_root / "exports" / "plots"

    parser.add_argument("--logs-root", type=Path, default=default_logs_root)
    parser.add_argument("--csv", type=Path, default=default_csv)
    parser.add_argument(
        "--task",
        type=str,
        help="Task directory name (e.g., lm-out-new-sort). Required for plotting.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="If set, generate a plot for --task.",
    )
    parser.add_argument(
        "--plot-path",
        type=Path,
        default=None,
        help="Output PNG path. Default: exports/plots/<task>.png",
    )
    parser.add_argument(
        "--include",
        action="append",
        default=[],
        help=(
            "Filter entries for plotting. Format: model or model:lr1,lr2 "
            "(repeat flag for multiple models). Example: --include 1l1h64d:0.001,0.0001"
        ),
    )
    args = parser.parse_args()

    rows = build_or_update_csv(logs_root=args.logs_root, csv_path=args.csv)
    print(f"Wrote/updated CSV: {args.csv} ({len(rows)} rows)")

    if not args.plot:
        return 0
    if not args.task:
        raise SystemExit("--task is required when using --plot")

    include_rules = [_parse_include_rule(v) for v in args.include]
    plot_path = args.plot_path or (default_plot_dir / f"{args.task}.png")
    plot_task(
        rows=rows,
        task=args.task,
        output_path=plot_path,
        include_rules=include_rules,
    )
    print(f"Wrote plot: {plot_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
