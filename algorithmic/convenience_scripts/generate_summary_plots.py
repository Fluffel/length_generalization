#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean

from generate_summary_csv import (
    BUCKETS,
    filter_rows,
    load_csv_rows,
    parse_include_rule,
)


def _extract_feature_tokens(value: str) -> list[tuple[str, str]]:
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

    pattern_chunks = re.findall(r"[0-9]+(?:\.[0-9]+)?[a-z]+|[a-z]+", pattern_l)
    if not pattern_chunks:
        return pattern_l in model_l

    for chunk in pattern_chunks:
        if chunk not in model_l:
            return False

    pattern_features = _extract_feature_tokens(" ".join(pattern_chunks))
    model_chunks = re.findall(r"[0-9]+(?:\.[0-9]+)?[a-z]+|[a-z]+", model_l)
    model_features = set(_extract_feature_tokens(" ".join(model_chunks)))
    for feat in pattern_features:
        if feat not in model_features:
            return False

    remaining = pattern_l
    for chunk in pattern_chunks:
        pos = remaining.find(chunk)
        if pos != -1:
            remaining = remaining[:pos] + remaining[pos + len(chunk) :]
    remaining = remaining.strip()
    if remaining:
        return remaining in model_l

    return True


def _sample_std(values: list[float]) -> float:
    n = len(values)
    if n <= 1:
        return 0.0
    mu = mean(values)
    var = sum((v - mu) ** 2 for v in values) / (n - 1)
    return math.sqrt(var)


def _group_label_for_model(model: str, group_patterns: list[str]) -> str | None:
    for pattern in group_patterns:
        if matches_pattern(model, pattern):
            return pattern
    return None


def _group_spec_from_models(models: set[str]) -> str:
    if not models:
        return ""

    model_list = sorted(models)
    parsed_models: list[list[str]] = []
    for model in model_list:
        # Preserve original spec order from CSV model strings.
        # Chunks are either alpha-only (e.g. "ssm") or num+alpha (e.g. "4l", "0.1dr").
        chunks = re.findall(r"[0-9]+(?:\.[0-9]+)?[a-z]+|[a-z]+", model.lower())
        parsed_models.append(chunks)

    max_len = max(len(chunks) for chunks in parsed_models)
    parts: list[str] = []
    for i in range(max_len):
        tokens_at_pos = [chunks[i] for chunks in parsed_models if i < len(chunks)]
        if not tokens_at_pos:
            continue
        uniq_tokens = sorted(set(tokens_at_pos))
        if len(uniq_tokens) == 1:
            parts.append(uniq_tokens[0])
            continue

        # If position varies only by numeric value of the same key, collapse as x/y<key>.
        key_to_vals: dict[str, set[str]] = defaultdict(set)
        valid_num_key = True
        for token in uniq_tokens:
            m = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)([a-z]+)", token)
            if not m:
                valid_num_key = False
                break
            key_to_vals[m.group(2)].add(m.group(1))
        if valid_num_key and len(key_to_vals) == 1:
            key = next(iter(key_to_vals.keys()))
            vals = sorted(key_to_vals[key], key=lambda x: float(x))
            parts.append(f"{'/'.join(vals)}{key}")
        else:
            # Fallback for mixed tokens at same position.
            parts.append("/".join(uniq_tokens))

    return "".join(parts)


def plot_task(
    rows: list[dict[str, str]],
    task: str,
    output_path: Path,
    title: str,
    legend_loc: str,
    include_rules: list[tuple[str, set[float] | None]],
    include_patterns: list[str],
    group_patterns: list[str],
    include_max: bool,
    include_max_only: bool,
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

    filtered_rows = filter_rows(
        rows,
        tasks=[task],
        include_rules=include_rules,
        include_patterns=include_patterns,
    )
    print(f"Filtered rows: {len(filtered_rows)}")
    if not filtered_rows:
        raise SystemExit(
            f"No rows to plot for task '{task}'. "
            "Check --task and include filters/patterns."
        )

    # Keep the raw datapoints per (model, lr, bucket) so include-max can select
    # per-bin best contributors and then build a derived max{...} series.
    datapoint_bucket_vals: dict[tuple[str, float, str], list[float]] = defaultdict(list)
    for row in filtered_rows:
        key = (row["model"], float(row["learning_rate"]), row["bucket"])
        datapoint_bucket_vals[key].append(float(row["accuracy"]))

    series_to_bucket_vals: dict[tuple[str, str], list[float]] = defaultdict(list)
    series_to_datapoints: dict[str, set[tuple[str, float]]] = defaultdict(set)
    series_to_models: dict[str, set[str]] = defaultdict(set)

    if group_patterns:
        # Grouping behavior:
        # - explicit patterns group matching models into aggregated series
        # - unmatched models remain individual series by default
        # - if "*" is present, unmatched models are aggregated into one final group
        wildcard_rest = "*" in group_patterns
        explicit_group_patterns = [p for p in group_patterns if p != "*"]
        models_by_group: dict[str, set[str]] = defaultdict(set)
        grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
        grouped_datapoints: dict[str, set[tuple[str, float]]] = defaultdict(set)
        for row in filtered_rows:
            datapoint = (row["model"], float(row["learning_rate"]))
            group_label = _group_label_for_model(row["model"], explicit_group_patterns)
            if group_label is None:
                if wildcard_rest:
                    group_label = "*"
                    models_by_group[group_label].add(row["model"])
                    grouped[(group_label, row["bucket"])].append(float(row["accuracy"]))
                    grouped_datapoints[group_label].add(datapoint)
                else:
                    # Keep default behavior for non-grouped rows: one line per model/lr.
                    series_label = f"{row['model']} | lr={float(row['learning_rate']):g}"
                    grouped[(series_label, row["bucket"])].append(float(row["accuracy"]))
                    grouped_datapoints[series_label].add(datapoint)
            else:
                models_by_group[group_label].add(row["model"])
                grouped[(group_label, row["bucket"])].append(float(row["accuracy"]))
                grouped_datapoints[group_label].add(datapoint)

        relabeled_grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
        relabeled_datapoints: dict[str, set[tuple[str, float]]] = defaultdict(set)
        relabeled_models: dict[str, set[str]] = defaultdict(set)
        for (base_group, bucket), vals in grouped.items():
            if base_group in models_by_group:
                group_spec = _group_spec_from_models(models_by_group.get(base_group, set()))
                final_group = group_spec if group_spec else base_group
            else:
                final_group = base_group
            relabeled_grouped[(final_group, bucket)].extend(vals)
            relabeled_datapoints[final_group].update(grouped_datapoints.get(base_group, set()))
            relabeled_models[final_group].update({m for m, _ in grouped_datapoints.get(base_group, set())})
        grouped = relabeled_grouped

        series_labels = sorted({k[0] for k in grouped.keys()})
        if not series_labels:
            raise SystemExit("No grouped series found. Check --group-pattern values.")
        series_to_bucket_vals = grouped
        series_to_datapoints = relabeled_datapoints
        series_to_models = relabeled_models
    else:
        grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
        for row in filtered_rows:
            series_label = f"{row['model']} | lr={float(row['learning_rate']):g}"
            grouped[(series_label, row["bucket"])].append(float(row["accuracy"]))
            series_to_datapoints[series_label].add((row["model"], float(row["learning_rate"])))
            series_to_models[series_label].add(row["model"])
        series_labels = sorted({k[0] for k in grouped.keys()})
        if not series_labels:
            raise SystemExit("No model/lr series found after filtering.")
        series_to_bucket_vals = grouped

    fig, ax = plt.subplots(figsize=(12, 7))
    x_ticks = [10, 20, 30]
    x_labels = ["Bin 1", "Bin 2", "Bin 3"]
    bucket_to_x = dict(zip(BUCKETS, x_ticks))

    # Optional derived max-series per line:
    # For each plotted line, select datapoints that achieve the maximum mean in any bin.
    # Build max{...} from the union of those winners and plot dotted in the same color.
    max_series: dict[str, tuple[str, list[float], list[float]]] = {}
    if include_max or include_max_only:
        for series_label in series_labels:
            datapoints = series_to_datapoints.get(series_label, set())
            if not datapoints:
                continue

            point_vec: dict[tuple[str, float], list[float]] = {}
            for dp in datapoints:
                vec: list[float] = []
                for bucket in BUCKETS:
                    vals = datapoint_bucket_vals.get((dp[0], dp[1], bucket), [])
                    vec.append(mean(vals) if vals else float("-inf"))
                point_vec[dp] = vec

            winner_points: set[tuple[str, float]] = set()
            for bucket in BUCKETS:
                per_point_means: list[tuple[tuple[str, float], float]] = []
                for dp in datapoints:
                    vals = datapoint_bucket_vals.get((dp[0], dp[1], bucket), [])
                    if vals:
                        per_point_means.append((dp, mean(vals)))
                if not per_point_means:
                    continue
                max_mean = max(v for _, v in per_point_means)
                for dp, v in per_point_means:
                    if v == max_mean:
                        winner_points.add(dp)
            if not winner_points:
                continue

            # Exclude winners that are dominated by another datapoint in all bins.
            # A point p is dominated if another point q has q_i >= p_i for every bin
            # and q_j > p_j for at least one bin.
            pruned_winners: set[tuple[str, float]] = set()
            for p in winner_points:
                p_vec = point_vec[p]
                dominated = False
                for q in datapoints:
                    if q == p:
                        continue
                    q_vec = point_vec[q]
                    if all(qv >= pv for qv, pv in zip(q_vec, p_vec)) and any(
                        qv > pv for qv, pv in zip(q_vec, p_vec)
                    ):
                        dominated = True
                        break
                if not dominated:
                    pruned_winners.add(p)
            winner_points = pruned_winners
            if not winner_points:
                continue

            winner_models = {m for m, _ in winner_points}
            max_suffix = _group_spec_from_models(winner_models) or series_label
            max_label = f"max:{max_suffix}"
            max_means: list[float] = []
            max_stds: list[float] = []
            for bucket in BUCKETS:
                vals: list[float] = []
                for dp in winner_points:
                    vals.extend(datapoint_bucket_vals.get((dp[0], dp[1], bucket), []))
                if vals:
                    max_means.append(mean(vals) * 100.0)
                    max_stds.append(_sample_std(vals) * 100.0)
                else:
                    max_means.append(float("nan"))
                    max_stds.append(float("nan"))
            max_series[series_label] = (max_label, max_means, max_stds)

    stem_colors: dict[str, str] = {}
    for series_label in series_labels:
        xs = [bucket_to_x[b] for b in BUCKETS]

        if not include_max_only:
            means: list[float] = []
            stds: list[float] = []
            for bucket in BUCKETS:
                vals = series_to_bucket_vals.get((series_label, bucket), [])
                if vals:
                    means.append(mean(vals) * 100.0)
                    stds.append(_sample_std(vals) * 100.0)
                else:
                    means.append(float("nan"))
                    stds.append(float("nan"))

            stem_line = ax.errorbar(
                xs,
                means,
                yerr=stds,
                marker="o",
                linestyle="solid",
                linewidth=2.0,
                capsize=5,
                elinewidth=1.5,
                label=series_label,
            )
            stem_colors[series_label] = stem_line.lines[0].get_color()

        if series_label in max_series:
            max_label, max_means, max_stds = max_series[series_label]
            dotted_color = stem_colors.get(series_label, None)
            ax.errorbar(
                xs,
                max_means,
                yerr=max_stds,
                marker="o",
                linestyle="dotted",
                linewidth=2.0,
                capsize=5,
                elinewidth=1.5,
                color=dotted_color,
                label=max_label,
            )

    ax.set_title(title)
    ax.set_xlabel("Validation bins")
    ax.set_ylabel("Accuracy (%)")
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels)
    ax.set_ylim(-10.0, 110.0)
    ax.grid(alpha=0.3)
    if legend_loc != "none":
        ax.legend(loc=legend_loc, fontsize=10, frameon=False)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Create per-task plots from the unified CSV with optional model/lr "
            "filtering and feature-based grouping."
        )
    )
    repo_root = Path(__file__).resolve().parents[2]
    default_csv = repo_root / "exports" / "all_summary_results.csv"
    default_plot_dir = repo_root / "exports" / "plots"

    parser.add_argument(
        "--input-csv",
        "--csv",
        dest="input_csv",
        type=Path,
        default=default_csv,
        help="Input CSV path. Default: exports/all_summary_results.csv",
    )
    parser.add_argument(
        "--task",
        type=str,
        help="Task directory name (e.g., lm-out-new-sort). Required for plotting.",
    )
    # parser.add_argument(
    #     "--plot",
    #     action="store_true",
    #     help="If set, generate a plot for --task.",
    # )
    parser.add_argument(
        "--output",
        "--plot-path",
        dest="output_path",
        type=Path,
        default=None,
        help="Output plot file path. Default: exports/plots/<task>.png",
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Plot title. Default: --task value.",
    )
    parser.add_argument(
        "--legend-loc",
        type=str,
        default="best",
        choices=[
            "best",
            "upper right",
            "upper left",
            "lower left",
            "lower right",
            "right",
            "center left",
            "center right",
            "lower center",
            "upper center",
            "center",
            "none",
        ],
        help=(
            "Legend location. Default is 'best' (automatic placement). "
            "Use 'none' to hide the legend."
        ),
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
    parser.add_argument(
        "--include-pattern",
        action="append",
        default=[],
        help=(
            "Pattern-based OR filter over model strings. Supports feature matching "
            "(order-insensitive), e.g. --include-pattern l1h1 or --include-pattern h1l1. "
            "You can repeat the flag and values are combined with OR."
        ),
    )
    parser.add_argument(
        "--group-pattern",
        action="append",
        default=[],
        help=(
            "Aggregate matching models into one series per pattern. Unmatched models stay "
            "as individual series unless '*' is provided. Use --group-pattern \\* to group "
            "the remaining unmatched models into one final group (escape * in shell). "
            "Repeat for multiple groups. Example: --group-pattern ssm --group-pattern l1h1 "
            "--group-pattern \\*"
        ),
    )
    max_group = parser.add_mutually_exclusive_group()
    max_group.add_argument(
        "--include-max",
        action="store_true",
        help=(
            "For each plotted line, add a dotted max:... line from datapoints that "
            "are bin-wise maxima within that line."
        ),
    )
    max_group.add_argument(
        "--include-max-only",
        action="store_true",
        help=(
            "Plot only dotted max:... lines (omit the original stem lines). "
            "Mutually exclusive with --include-max."
        ),
    )
    args = parser.parse_args()

    # if not args.plot:
    #     raise SystemExit("Use --plot to generate a plot.")
    if not args.task:
        raise SystemExit("--task is required.")

    rows = load_csv_rows(args.input_csv)
    if not rows:
        raise SystemExit(
            f"No CSV rows found at {args.input_csv}. "
            "Generate it first with generate_summary_csv.py."
        )

    include_rules = [parse_include_rule(v) for v in args.include]
    plot_path = args.output_path or (default_plot_dir / f"{args.task}.png")
    title = args.title or args.task
    plot_task(
        rows=rows,
        task=args.task,
        output_path=plot_path,
        title=title,
        legend_loc=args.legend_loc,
        include_rules=include_rules,
        include_patterns=args.include_pattern,
        group_patterns=args.group_pattern,
        include_max=args.include_max,
        include_max_only=args.include_max_only,
    )
    print(f"Wrote plot: {plot_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
