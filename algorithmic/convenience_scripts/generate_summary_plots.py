#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean

from generate_summary_csv import (
    feature_from_token,
    filter_rows,
    load_csv_rows,
    row_matches_spec_pattern,
    tokenize_model,
)


def _sample_std(values: list[float]) -> float:
    n = len(values)
    if n <= 1:
        return 0.0
    mu = mean(values)
    var = sum((v - mu) ** 2 for v in values) / (n - 1)
    return math.sqrt(var)


_BUCKET_BOUNDS_RE = re.compile(r"^(\d+)\s*-\s*(\d+)$")


def _parse_bucket_bounds(bucket: str) -> tuple[int, int] | None:
    """Return inclusive ``(start, end)`` for ``\"0-24\"``-style bucket names."""
    m = _BUCKET_BOUNDS_RE.match(bucket.strip())
    if not m:
        return None
    a, b = int(m.group(1)), int(m.group(2))
    if a > b:
        return None
    return (a, b)


def _bucket_end(bucket: str) -> int | None:
    bounds = _parse_bucket_bounds(bucket)
    return bounds[1] if bounds else None


def _bucket_width(bucket: str) -> int | None:
    """Inclusive span length (finer bins ⇒ smaller width)."""
    bounds = _parse_bucket_bounds(bucket)
    if not bounds:
        return None
    return bounds[1] - bounds[0] + 1


def _bucket_sort_key_plot(bucket: str) -> tuple[int, str]:
    e = _bucket_end(bucket)
    return (e if e is not None else -1, bucket)


def _buckets_for_datapoint(
    dp: tuple[str, float], dcv: dict[tuple[str, float, str], list[float]]
) -> list[str]:
    m, lr = dp
    return sorted(
        {b for (m2, lr2, b) in dcv if m2 == m and lr2 == lr},
        key=_bucket_sort_key_plot,
    )


def _mean_for_dp_bucket(
    dp: tuple[str, float], bucket: str, dcv: dict[tuple[str, float, str], list[float]]
) -> float | None:
    vals = dcv.get((dp[0], dp[1], bucket), [])
    return mean(vals) if vals else None


def _finest_buckets_at_end_for_dp(
    dp: tuple[str, float], end: int, dcv: dict[tuple[str, float, str], list[float]]
) -> list[str]:
    """Buckets for *dp* with this inclusive upper bound and minimal width."""
    cands = [
        b
        for b in _buckets_for_datapoint(dp, dcv)
        if _bucket_end(b) == end
    ]
    if not cands:
        return []
    w_min = min(w for b in cands if (w := _bucket_width(b)) is not None)
    return [b for b in cands if _bucket_width(b) == w_min]


def _value_at_end_for_dp(
    dp: tuple[str, float], end: int, dcv: dict[tuple[str, float, str], list[float]]
) -> float:
    """Mean accuracy using the finest bucket(s) at *end*; max if ties."""
    finest = _finest_buckets_at_end_for_dp(dp, end, dcv)
    if not finest:
        return float("-inf")
    scores = [_mean_for_dp_bucket(dp, b, dcv) for b in finest]
    scores = [s for s in scores if s is not None]
    return max(scores) if scores else float("-inf")


def select_max_winners_for_series(
    datapoints: set[tuple[str, float]],
    dcv: dict[tuple[str, float, str], list[float]],
    *,
    all_ends_override: list[int] | None = None,
) -> tuple[set[tuple[str, float]], list[int]]:
    """Pick non-dominated bin-end maxima, preferring finer (narrower) bins at each end.

    1. For each bin upper bound *e*, keep only buckets with minimal width among
       those ending at *e*, then take datapoints achieving the maximum mean
       accuracy among those finest buckets.
    2. Union winners across ends, then remove points Pareto-dominated on the
       shared vector of values at every *e* (finest-bin value per dp at each *e*).

    If *all_ends_override* is set (e.g. all bin ends on the plot), use it as the
    comparison grid so maxima are comparable **across** series/groups on a
    shared axis; otherwise *e* ranges only over ends seen in *datapoints*.
    """
    if all_ends_override is not None:
        all_ends = sorted(set(all_ends_override))
    else:
        ends_set: set[int] = set()
        for dp in datapoints:
            for b in _buckets_for_datapoint(dp, dcv):
                e = _bucket_end(b)
                if e is not None:
                    ends_set.add(e)
        all_ends = sorted(ends_set)
    if not all_ends:
        return set(), []

    winner_dps: set[tuple[str, float]] = set()
    for e in all_ends:
        pairs: list[tuple[tuple[str, float], str]] = []
        for dp in datapoints:
            for b in _buckets_for_datapoint(dp, dcv):
                if _bucket_end(b) != e:
                    continue
                if not dcv.get((dp[0], dp[1], b)):
                    continue
                pairs.append((dp, b))
        if not pairs:
            continue
        widths = [w for _, b in pairs if (w := _bucket_width(b)) is not None]
        if not widths:
            continue
        w_min = min(widths)
        fine = [(dp, b) for dp, b in pairs if _bucket_width(b) == w_min]
        scored: list[tuple[tuple[str, float], str, float]] = []
        for dp, b in fine:
            m = _mean_for_dp_bucket(dp, b, dcv)
            if m is not None:
                scored.append((dp, b, m))
        if not scored:
            continue
        best = max(t[2] for t in scored)
        for dp, _, m in scored:
            if m == best:
                winner_dps.add(dp)

    if not winner_dps:
        return set(), all_ends

    point_vec: dict[tuple[str, float], list[float]] = {}
    for dp in winner_dps:
        point_vec[dp] = [_value_at_end_for_dp(dp, e, dcv) for e in all_ends]

    pruned: set[tuple[str, float]] = set()
    for p in winner_dps:
        pv = point_vec[p]
        dominated = False
        for q in winner_dps:
            if q == p:
                continue
            qv = point_vec[q]
            if all(qv[i] >= pv[i] for i in range(len(all_ends))) and any(
                qv[i] > pv[i] for i in range(len(all_ends))
            ):
                dominated = True
                break
        if not dominated:
            pruned.add(p)

    return pruned, all_ends


def _pool_raw_at_end_for_dps(
    dps: set[tuple[str, float]],
    end: int,
    dcv: dict[tuple[str, float, str], list[float]],
) -> list[float]:
    """Concatenate raw accuracies at *end* using each dp's finest bucket there."""
    chunk: list[float] = []
    for dp in dps:
        for b in _finest_buckets_at_end_for_dp(dp, end, dcv):
            chunk.extend(dcv.get((dp[0], dp[1], b), []))
    return chunk


def max_line_xy_for_winners(
    pruned: set[tuple[str, float]],
    all_ends: list[int],
    dcv: dict[tuple[str, float, str], list[float]],
    *,
    fallback_dps: set[tuple[str, float]] | None = None,
) -> tuple[list[float], list[float], list[float]]:
    """Build dotted max-line coordinates (x = bin end, y = %).

    At each upper bound in *all_ends*, pool values from *pruned* winners (finest
    bin).  If none of the pruned set has data at that end, fall back to pooling
    from *fallback_dps* (typically all datapoints in the series) so the curve
    spans every bin end that appears for that line — e.g. one model out to 149
    and another only to 124 still each get their full extent on the max curve.
    """
    fallback = fallback_dps if fallback_dps is not None else set()
    xs: list[float] = []
    means: list[float] = []
    stds: list[float] = []
    for e in all_ends:
        chunk = _pool_raw_at_end_for_dps(pruned, e, dcv)
        if not chunk and fallback:
            chunk = _pool_raw_at_end_for_dps(fallback, e, dcv)
        if chunk:
            xs.append(float(e))
            means.append(mean(chunk) * 100.0)
            stds.append(_sample_std(chunk) * 100.0)
    return xs, means, stds


_GROUP_BUCKET_TOKEN_RE = re.compile(r"^\d+-\d+$")


def _group_pattern_for_membership(pattern: str) -> str:
    """Strip ``N-M`` bin tokens so grouping is by model/config only.

    Otherwise a pattern like ``hyb,0-100`` would only match rows whose CSV
    ``bucket`` is ``0-100``, splitting long-bin rows into other groups and
    capping a series at the wrong upper bound.
    """
    parts = [p.strip() for p in pattern.split(",") if p.strip()]
    kept = [p for p in parts if not _GROUP_BUCKET_TOKEN_RE.fullmatch(p.lower())]
    if not kept:
        return pattern.strip().lower()
    return ",".join(kept).lower()


def _group_label_for_row(row: dict[str, str], group_patterns: list[str]) -> str | None:
    for pattern in group_patterns:
        if pattern == "*":
            continue
        if row_matches_spec_pattern(row, _group_pattern_for_membership(pattern)):
            return pattern
    return None


# Matches pure layer-ordering tokens: sequences of only 's' and 'a' characters.
# These tokens represent the order of SSM and Attention layers in a hybrid model
# (e.g. "sa", "as", "sas") and must be handled as a single semantic group so
# that different orderings are shown as "sa/as" rather than concatenated.
_LAYERING_TOKEN_RE = re.compile(r"^[sa]+$")


def _group_spec_from_models(models: set[str]) -> str:
    """Build a compact human-readable label for a set of model strings.

    Uses key-based aggregation rather than positional token comparison, so
    models with different token counts or ordering (e.g. kernel before vs.
    after the layer count) are handled correctly.  Layer-ordering tokens
    (``sa``, ``as``, ``sas``, …) are grouped into a single slot and displayed
    as ``sa/as`` when they differ across models.
    """
    if not models:
        return ""

    model_list = sorted(models)

    # Collect an ordered list of all unique "keys" seen across models.
    # For numeric feature tokens: key = the alpha identifier (e.g. "l", "mlp", "s").
    # For layer-ordering tokens:  key = the sentinel "__layering__".
    # For other pure-alpha flags: key = the token itself (e.g. "hyb", "nope").
    key_order: list[str] = []
    seen_keys: set[str] = set()

    # Per-model mapping from key → set of values.
    # Pure-alpha flags: value is None (flag = present).
    # Layer-ordering tokens: value is the token string itself.
    # Numeric features: value is the numeric string.
    per_model: list[dict[str, set]] = []

    for model in model_list:
        kv: dict[str, set] = defaultdict(set)
        for tok in tokenize_model(model):
            if _LAYERING_TOKEN_RE.match(tok) and feature_from_token(tok) is None:
                kv["__layering__"].add(tok)
                if "__layering__" not in seen_keys:
                    seen_keys.add("__layering__")
                    key_order.append("__layering__")
            else:
                feat = feature_from_token(tok)
                if feat is not None:
                    key, val = feat
                    kv[key].add(val)
                    if key not in seen_keys:
                        seen_keys.add(key)
                        key_order.append(key)
                else:
                    kv[tok].add(None)
                    if tok not in seen_keys:
                        seen_keys.add(tok)
                        key_order.append(tok)
        per_model.append(dict(kv))

    parts: list[str] = []
    for key in key_order:
        all_vals: set = set()
        for kv in per_model:
            all_vals |= kv.get(key, set())

        if key == "__layering__":
            # Show all unique layer orderings with "/" separator.
            unique = sorted(v for v in all_vals if v is not None)
            parts.append("/".join(unique) if unique else "")
        elif None in all_vals:
            # Pure-alpha flag: show the flag once.
            parts.append(key)
        else:
            # Numeric feature: show all values, collapsed by key.
            str_vals = [v for v in all_vals if v is not None]
            try:
                sorted_vals = sorted(str_vals, key=float)
            except (ValueError, TypeError):
                sorted_vals = sorted(str_vals)

            # Determine whether the original token is alpha+num ("s4") or
            # num+alpha ("1l") by inspecting any model that has this key.
            alpha_prefix = False
            for model in model_list:
                for tok in tokenize_model(model):
                    f = feature_from_token(tok)
                    if f and f[0] == key:
                        alpha_prefix = bool(re.match(r"^[a-z]", tok))
                        break
                else:
                    continue
                break

            if alpha_prefix:
                parts.append(f"{key}{'/'.join(sorted_vals)}")
            else:
                parts.append(f"{'/'.join(sorted_vals)}{key}")

    return "".join(parts)


def plot_task(
    rows: list[dict[str, str]],
    task: str,
    output_path: Path,
    title: str,
    legend_loc: str,
    include_patterns: list[str],
    remove_patterns: list[str],
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
        include_rules=[],
        include_patterns=include_patterns,
        remove_patterns=remove_patterns,
    )
    print(f"Filtered rows: {len(filtered_rows)}")
    if not filtered_rows:
        raise SystemExit(
            f"No rows to plot for task '{task}'. "
            "Check --task, --include-pattern, and --remove-pattern."
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
            group_label = _group_label_for_row(row, explicit_group_patterns)
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

    all_bucket_names = {r["bucket"] for r in filtered_rows}
    x_tick_ends = sorted(
        {e for b in all_bucket_names if (e := _bucket_end(b)) is not None}
    )
    if not x_tick_ends:
        raise SystemExit(
            "No parseable bucket ranges in filtered rows (expected names like '0-50')."
        )

    def series_buckets(series_label: str) -> list[str]:
        return sorted(
            {b for (sl, b) in series_to_bucket_vals if sl == series_label},
            key=_bucket_sort_key_plot,
        )

    # One dotted max line per **series** (each group-pattern / each unmatched model|lr),
    # finest-bin + Pareto within that series, on the **shared** grid ``x_tick_ends`` so
    # all lines align on the same bin upper bounds (with per-series fallback for extent).
    max_series: dict[str, tuple[str, list[float], list[float], list[float]]] = {}
    if include_max or include_max_only:
        for series_label in series_labels:
            datapoints = series_to_datapoints.get(series_label, set())
            if not datapoints:
                continue
            pruned, _ = select_max_winners_for_series(
                datapoints,
                datapoint_bucket_vals,
                all_ends_override=x_tick_ends,
            )
            if not pruned:
                continue
            winner_models = {m for m, _ in pruned}
            max_suffix = _group_spec_from_models(winner_models) or series_label
            max_label = f"max:{max_suffix}"
            mx, mmean, mstd = max_line_xy_for_winners(
                pruned,
                x_tick_ends,
                datapoint_bucket_vals,
                fallback_dps=datapoints,
            )
            max_series[series_label] = (max_label, mx, mmean, mstd)

    # Axis extent: every bin end in the CSV, plus any series / max-line point (defensive).
    x_max_data = float(max(x_tick_ends))
    for sl in series_labels:
        for b in series_buckets(sl):
            if (e := _bucket_end(b)) is not None:
                x_max_data = max(x_max_data, float(e))
    for _lbl, mx_pts, _, _ in max_series.values():
        if mx_pts:
            x_max_data = max(x_max_data, max(mx_pts))

    fig, ax = plt.subplots(figsize=(12, 7))
    x_ticks = [float(e) for e in x_tick_ends]
    x_labels = [f"<{e}" for e in x_tick_ends]

    stem_colors: dict[str, str] = {}
    for series_label in series_labels:
        buckets_sl = series_buckets(series_label)
        xs_stem = [float(_bucket_end(b)) for b in buckets_sl if _bucket_end(b) is not None]
        if not include_max_only:
            means: list[float] = []
            stds: list[float] = []
            for b in buckets_sl:
                if _bucket_end(b) is None:
                    continue
                vals = series_to_bucket_vals.get((series_label, b), [])
                if vals:
                    means.append(mean(vals) * 100.0)
                    stds.append(_sample_std(vals) * 100.0)
                else:
                    means.append(float("nan"))
                    stds.append(float("nan"))

            stem_line = ax.errorbar(
                xs_stem,
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

    for series_label in series_labels:
        if series_label not in max_series:
            continue
        max_label, mx, max_means, max_stds = max_series[series_label]
        dotted_color = stem_colors.get(series_label)
        if dotted_color is None and include_max_only:
            ln = ax.plot([], [], linestyle="solid")[0]
            dotted_color = ln.get_color()
            ln.remove()
        ax.errorbar(
            mx,
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
    ax.set_xlabel("Validation length (upper bound)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels)
    ax.set_ylim(-10.0, 110.0)
    pad = max(x_max_data * 0.02, 1.0)
    ax.set_xlim(0.0, x_max_data + pad)
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
            "Create per-task plots from the unified CSV with optional "
            "--include-pattern / --remove-pattern filtering and grouping."
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
        "--remove-pattern",
        action="append",
        default=[],
        help=(
            "Pre-filter: drop rows matching any pattern before plotting.  Same notation "
            "as --include-pattern (comma = AND within a pattern)."
        ),
    )
    parser.add_argument(
        "--include-pattern",
        action="append",
        default=[],
        help=(
            "Keep rows matching any pattern (OR across flags).  Comma-separated tokens "
            "are AND-ed against CSV columns, e.g. 'hyb,1l,s4'.  No comma: structured "
            "token then legacy model-string match."
        ),
    )
    parser.add_argument(
        "--group-pattern",
        action="append",
        default=[],
        help=(
            "Aggregate rows matching each pattern into one series (same CSV short notation "
            "as --include-pattern).  Bin-range tokens like 0-50 are **ignored** for matching "
            "so every eval bucket for the same model still belongs to the same group.  "
            "Use '*' for the rest.  Example: --group-pattern ssm --group-pattern hyb,1l --group-pattern \\*"
        ),
    )
    max_group = parser.add_mutually_exclusive_group()
    max_group.add_argument(
        "--include-max",
        action="store_true",
        help=(
            "Per plotted series, add a dotted max:... line (same color as the solid line): "
            "shared bin-end grid across the figure, finest bin at each end, Pareto prune "
            "within that series."
        ),
    )
    max_group.add_argument(
        "--include-max-only",
        action="store_true",
        help=(
            "Plot only dotted max:... lines (one per series, omit solid lines). "
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

    plot_path = args.output_path or (default_plot_dir / f"{args.task}.png")
    title = args.title or args.task
    plot_task(
        rows=rows,
        task=args.task,
        output_path=plot_path,
        title=title,
        legend_loc=args.legend_loc,
        include_patterns=args.include_pattern,
        remove_patterns=args.remove_pattern,
        group_patterns=args.group_pattern,
        include_max=args.include_max,
        include_max_only=args.include_max_only,
    )
    print(f"Wrote plot: {plot_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
