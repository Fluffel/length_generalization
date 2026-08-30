#!/usr/bin/env python3
"""DataFrame-first plotting from summary CSV.

This script keeps plot aesthetics and legend naming consistent with
``generate_plot.py`` while making selection logic explicit via pandas filters:

* ``--keep column=v1,v2``          -> ``df[df[column].isin([...])]``
* ``--remove column=v1,v2``        -> ``df[~df[column].isin([...])]``
* ``--query "expr"``               -> ``df.query(expr)``
* ``--exclude-query "expr"``       -> ``df[~df.query(expr).index]``

Groups are built from ``--group-by`` columns and plotted via their max-winner
datapoints per validation bin (same winner logic as ``generate_plot.py``).

The summary CSV holds one row per run with its validation bins side by side
(``bin1_range``/``bin1_acc``, …); it is exploded into one row per (run, bin) on
load, which exposes the run-level columns ``num_bins``, ``bins``,
``train_range``, ``train_len_min/max`` and ``max_eval_len`` to all filters.
Runs with differing bin counts or training ranges therefore never get mixed into
one series, and can be selected with:

* ``--num-bins 3,6``               -> keep runs with these bin counts
* ``--min-bins`` / ``--max-bins``  -> keep runs within a bin-count range
* ``--train-range 0-49,0-24``      -> keep runs trained on these length ranges
* ``--exclude-train-range 0-9``    -> drop runs trained on these ranges
* ``--bins "0-49|50-99|100-149"``  -> keep runs with exactly this bin layout
* ``--first-bins 3``               -> plot only each run's first three bins
* ``--max-bin-len 149``            -> drop bins reaching beyond this length

Multitask grids (``--multitask``) place one task per axis with:

* ``--ncols``        axes per row
* ``--multititles``  per-axis titles (defaults to task names)
* ``--plot-size``    per-axis size in inches (``W,H`` / ``WxH``); figure is
                     ``ncols*W`` by ``nrows*H``
* ``--merge-bins``   merge runs by ordinal bin index, ignoring interval names
                     (requires ``--x-ticks bins``)
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

from plot_utils import (
    BinFilter,
    load_summary_dataframe,
    normalize_bin_signatures,
    normalize_int_tokens,
    normalize_range_tokens,
    _bucket_plot_x,
    _bucket_sort_key_plot,
    _bucket_upper_bound,
    _dedupe_legend_labels,
    _draw_x_shrink_marks,
    _ordinal_bin_tick_labels,
    _parse_x_axis_shrink,
    _sample_std,
    _signature_label,
    _signature_x_ends,
    _split_rows_by_bin_signature,
    _x_data_to_plot_shrink,
    legend_label_from_rows,
    signature_layout_label,
    max_line_xy_by_bin_index,
    max_line_xy_for_winners,
    select_max_winners_by_bin_index,
    select_max_winners_for_series,
    _remap_rows_to_ordinal_bins,
)
from dataframe_query_utils import (
    apply_keep_remove_filters,
    apply_query_filters,
    require_columns,
)


def _format_group_key(row: dict, group_by: list[str]) -> str:
    if not group_by:
        return f"dp:{row['model']}|{row['learning_rate']}"
    return "grp:" + "|".join(f"{c}={row.get(c)}" for c in group_by)


def _parse_custom_group_labels(raw: str) -> list[str]:
    """Parse comma-separated labels, allowing quoted CSV-style values."""
    if not raw.strip():
        return []
    parsed = next(csv.reader([raw], skipinitialspace=True), [])
    return [p.strip() for p in parsed if p.strip()]


def _group_label_from_sid(sid: str, fallback: str) -> str:
    if sid.startswith("grp:"):
        return sid[len("grp:") :]
    return fallback


def _parse_plot_size(raw: str | None) -> tuple[float, float] | None:
    """Parse per-axis plot size as ``W,H`` or ``WxH`` (inches)."""
    if raw is None or not str(raw).strip():
        return None
    s = str(raw).strip().lower().replace(" ", "")
    if "x" in s:
        parts = s.split("x", 1)
    else:
        parts = s.split(",", 1)
    if len(parts) != 2:
        raise SystemExit(f"Invalid --plot-size {raw!r}; expected W,H or WxH.")
    try:
        w, h = float(parts[0]), float(parts[1])
    except ValueError as e:
        raise SystemExit(f"Invalid --plot-size {raw!r}; expected numeric W,H.") from e
    if w <= 0 or h <= 0:
        raise SystemExit("--plot-size width and height must be > 0.")
    return (w, h)


def _prepare_task_plot(
    df,
    *,
    task: str,
    group_by: list[str],
    group_label_mode: str,
    group_custom_labels: list[str],
    max_aggregation: str,
    x_ticks_mode: str,
    x_tick_step: int,
    x_axis_break: str | None,
    num_bins: int | None,
    merge_bins: bool,
) -> dict:
    """Collapse/filter one task and compute max-series geometry for drawing."""
    df = df.copy()
    df["task"] = df["task"].astype(str)
    df = df[df["task"] == task]
    if df.empty:
        raise SystemExit(f"No rows left for task={task!r} after filtering.")

    df["bucket"] = df["bucket"].astype(str)
    df["bucket_upper"] = df["bucket"].map(_bucket_upper_bound)
    df["learning_rate"] = df["learning_rate"].astype(float)
    df["accuracy"] = df["accuracy"].astype(float)

    if num_bins is not None:
        if "num_bins" in df.columns:
            df = df[df["num_bins"].astype(int) == num_bins]
        else:
            cnt = (
                df.groupby(["model", "learning_rate"])["bucket"]
                .nunique()
                .rename("num_bins")
                .reset_index()
            )
            keep_dps = cnt[cnt["num_bins"] == num_bins][["model", "learning_rate"]]
            df = df.merge(keep_dps, on=["model", "learning_rate"], how="inner")
        if df.empty:
            raise SystemExit(f"No rows left after --num-bins={num_bins} for task={task!r}.")

    require_columns(df, group_by, "--group-by")

    # Keep one value per datapoint-bucket: max accuracy.
    # Important: when grouping is active, include group-by columns in the key so
    # runs from different groups (e.g. mkar_vocab_size=32 vs 128) never mix.
    # ``bins`` keeps runs of the same model with different bin layouts apart.
    collapse_keys: list[str] = []
    for col in ["model", "learning_rate", "bucket", "bins", *group_by]:
        if col not in collapse_keys and (col != "bins" or "bins" in df.columns):
            collapse_keys.append(col)
    dfc = (
        df.groupby(collapse_keys, as_index=False)["accuracy"]
        .max()
        .copy()
    )

    filtered_rows = dfc.to_dict(orient="records")
    if not filtered_rows:
        raise SystemExit(f"No rows to plot after collapse/filtering for task={task!r}.")

    if merge_bins:
        if x_ticks_mode != "bins":
            raise SystemExit("--merge-bins requires --x-ticks bins.")
        filtered_rows = _remap_rows_to_ordinal_bins(filtered_rows)

    if x_ticks_mode == "bins" and x_axis_break is not None and str(x_axis_break).strip():
        raise SystemExit("--x-axis-break is incompatible with --x-ticks bins.")

    # Group-by series + split by identical bucket signatures.
    series_id_to_rows: dict[str, list[dict]] = defaultdict(list)
    ordered_sids: list[str] = []
    seen_sids: set[str] = set()
    for row in filtered_rows:
        sid = _format_group_key(row, group_by)
        if sid not in seen_sids:
            seen_sids.add(sid)
            ordered_sids.append(sid)
        series_id_to_rows[sid].append(row)

    if group_label_mode == "custom":
        if not group_by:
            raise SystemExit("--group-label-mode custom requires at least one --group-by column.")
        if not group_custom_labels:
            raise SystemExit("--group-label-mode custom requires --group-custom-labels.")
        if len(group_custom_labels) != len(ordered_sids):
            raise SystemExit(
                f"--group-custom-labels count ({len(group_custom_labels)}) must match "
                f"number of groups ({len(ordered_sids)}) for task={task!r}, "
                "in first-appearance order."
            )
        sid_to_custom_label = {
            sid: group_custom_labels[i] for i, sid in enumerate(ordered_sids)
        }
    else:
        sid_to_custom_label = {}

    sub_series_rows: dict[tuple[str, frozenset[str]], list[dict]] = {}
    empty_sig = frozenset()
    for sid, sid_rows in series_id_to_rows.items():
        if merge_bins:
            sub_series_rows[(sid, empty_sig)] = sid_rows
        else:
            for sig, sig_rows in _split_rows_by_bin_signature(sid_rows).items():
                sub_series_rows[(sid, sig)] = sig_rows

    def sub_key_sort(k: tuple[str, frozenset[str]]) -> tuple:
        sid, sig = k
        return (legend_label_from_rows(sub_series_rows[k]), sid, _signature_label(sig))

    sub_keys = sorted(sub_series_rows.keys(), key=sub_key_sort)
    if not sub_keys:
        raise SystemExit(f"No sub-series to plot for task={task!r}.")

    base_labels = {sk: legend_label_from_rows(sub_series_rows[sk]) for sk in sub_keys}
    sig_count_by_sid: dict[str, int] = defaultdict(int)
    for sid, _sig in sub_keys:
        sig_count_by_sid[sid] += 1

    display_labels: dict[tuple[str, frozenset[str]], str] = {}
    for sk in sub_keys:
        sid, sig = sk
        if group_label_mode == "model":
            base = base_labels[sk]
        elif group_label_mode == "group":
            base = _group_label_from_sid(sid, base_labels[sk])
        else:  # custom
            base = sid_to_custom_label.get(sid, base_labels[sk])
        if merge_bins or sig_count_by_sid[sid] <= 1:
            display_labels[sk] = base
        else:
            display_labels[sk] = f"{base} [{signature_layout_label(sig)}]"

    id_strs = [f"{sid}::{_signature_label(sig)}" for sid, sig in sub_keys]
    deduped = _dedupe_legend_labels(
        id_strs, {id_strs[i]: display_labels[sub_keys[i]] for i in range(len(sub_keys))}
    )
    sub_key_to_display = {sub_keys[i]: deduped[id_strs[i]] for i in range(len(sub_keys))}

    series_to_bucket_vals: dict[tuple[tuple[str, frozenset[str]], str], list[float]] = defaultdict(list)
    series_to_datapoints: dict[tuple[str, frozenset[str]], set[tuple[str, float]]] = defaultdict(set)
    series_to_dcv: dict[
        tuple[str, frozenset[str]],
        dict[tuple[str, float, str], list[float]],
    ] = {}

    for sk in sub_keys:
        local_dcv: dict[tuple[str, float, str], list[float]] = defaultdict(list)
        for row in sub_series_rows[sk]:
            dp = (str(row["model"]), float(row["learning_rate"]))
            b = str(row["bucket"])
            local_dcv[(dp[0], dp[1], b)].append(float(row["accuracy"]))
            series_to_datapoints[sk].add(dp)
        # Match legacy behavior: one value per datapoint-bucket, keep max.
        for key in list(local_dcv.keys()):
            vals = local_dcv[key]
            local_dcv[key] = [max(vals)] if vals else []
        series_to_dcv[sk] = local_dcv
        for (_m, _lr, b), vals in local_dcv.items():
            if vals:
                series_to_bucket_vals[(sk, b)].append(vals[0])

    all_bucket_names = {str(r["bucket"]) for r in filtered_rows}
    x_tick_ends = sorted({int(x) for b in all_bucket_names if (x := _bucket_plot_x(b)) is not None})
    if not x_tick_ends:
        raise SystemExit(
            f"No parseable bucket ranges for task={task!r} "
            "(expected bucket names like '0-50')."
        )

    def series_buckets(sk: tuple[str, frozenset[str]]) -> list[str]:
        return sorted({b for (k, b) in series_to_bucket_vals if k == sk}, key=_bucket_sort_key_plot)

    use_bins = x_ticks_mode == "bins"

    def bucket_x(sk: tuple[str, frozenset[str]], b: str) -> float | None:
        if use_bins:
            return float(series_buckets(sk).index(b))
        return _bucket_plot_x(b)

    if use_bins:
        ordinal_tick_labels = _ordinal_bin_tick_labels(sub_keys, series_buckets)
        num_ordinal_bins = len(ordinal_tick_labels)
    else:
        ordinal_tick_labels = []
        num_ordinal_bins = 0

    # Modes:
    # - pareto_mean: keep Pareto max-contributor datapoints, then mean+std per bin.
    # - bin_max: simple maximum per bin across all datapoints in the group/sub-series.
    max_series: dict[tuple[str, frozenset[str]], tuple[str, list[float], list[float], list[float]]] = {}
    for sk in sub_keys:
        max_label = f"{sub_key_to_display[sk]}"
        datapoints = series_to_datapoints.get(sk, set())
        if not datapoints:
            continue

        mx: list[float]
        mmean: list[float]
        mstd: list[float]
        if max_aggregation == "pareto_mean":
            if merge_bins:
                buckets_sl = series_buckets(sk)
                all_bin_indices = list(range(len(buckets_sl)))
                local_dcv = series_to_dcv.get(sk, {})
                pruned, _ = select_max_winners_by_bin_index(
                    datapoints,
                    local_dcv,
                    all_bin_indices=all_bin_indices,
                )
                if not pruned:
                    continue
                mx, mmean, mstd = max_line_xy_by_bin_index(
                    pruned,
                    all_bin_indices,
                    local_dcv,
                    fallback_dps=datapoints,
                )
            else:
                sig = sk[1]
                ends_for_sig = _signature_x_ends(sig)
                local_dcv = series_to_dcv.get(sk, {})
                pruned, _ = select_max_winners_for_series(
                    datapoints,
                    local_dcv,
                    all_ends_override=ends_for_sig or x_tick_ends,
                )
                if not pruned:
                    continue
                mx, mmean, mstd = max_line_xy_for_winners(
                    pruned,
                    ends_for_sig or x_tick_ends,
                    local_dcv,
                    fallback_dps=datapoints,
                )
        else:  # bin_max
            buckets_sl = series_buckets(sk)
            if not buckets_sl:
                continue
            mx, mmean, mstd = [], [], []
            for b in buckets_sl:
                x = bucket_x(sk, b)
                if x is None:
                    continue
                vals = series_to_bucket_vals.get((sk, b), [])
                if not vals:
                    continue
                mx.append(float(x))
                mmean.append(max(vals) * 100.0)
                mstd.append(0.0)

        if use_bins:
            mx = [float(i) for i in range(len(mx))]
        max_series[sk] = (max_label, mx, mmean, mstd)

    x_max_data = float(max(x_tick_ends))
    if use_bins:
        x_max_data = float(max(0, num_ordinal_bins - 1))
    else:
        for sk in sub_keys:
            for b in series_buckets(sk):
                if (x_b := _bucket_plot_x(b)) is not None:
                    x_max_data = max(x_max_data, float(x_b))
        for _lbl, mx_pts, _, _ in max_series.values():
            if mx_pts:
                x_max_data = max(x_max_data, max(mx_pts))

    all_plotted_x: list[float] = []
    for sk in sub_keys:
        if sk not in max_series:
            continue
        _lbl, mx, _m, _s = max_series[sk]
        all_plotted_x.extend(float(x) for x in mx)
    min_plotted_x = min(all_plotted_x) if all_plotted_x else 0.0
    shrink_to = _parse_x_axis_shrink(x_axis_break, min_plotted_x)

    if use_bins:
        pad = max(0.08 * max(x_max_data + 1.0, 1.0), 0.42)
        x_hi_data = x_max_data + pad
    else:
        pad = max(x_max_data * 0.02, 1.0)
        x_hi_data = x_max_data + pad

    return {
        "task": task,
        "sub_keys": sub_keys,
        "max_series": max_series,
        "use_bins": use_bins,
        "x_ticks_mode": x_ticks_mode,
        "x_tick_step": x_tick_step,
        "x_tick_ends": x_tick_ends,
        "ordinal_tick_labels": ordinal_tick_labels,
        "num_ordinal_bins": num_ordinal_bins,
        "all_plotted_x": all_plotted_x,
        "min_plotted_x": min_plotted_x,
        "shrink_to": shrink_to,
        "x_hi_data": x_hi_data,
    }


def _draw_task_on_ax(ax, prepared: dict, *, title: str, legend_loc: str) -> None:
    """Render one prepared task onto an existing matplotlib Axes."""
    sub_keys = prepared["sub_keys"]
    max_series = prepared["max_series"]
    use_bins = prepared["use_bins"]
    x_ticks_mode = prepared["x_ticks_mode"]
    x_tick_step = prepared["x_tick_step"]
    x_tick_ends = prepared["x_tick_ends"]
    ordinal_tick_labels = prepared["ordinal_tick_labels"]
    num_ordinal_bins = prepared["num_ordinal_bins"]
    all_plotted_x = prepared["all_plotted_x"]
    min_plotted_x = prepared["min_plotted_x"]
    shrink_to = prepared["shrink_to"]
    x_hi_data = prepared["x_hi_data"]

    def xplt(x: float) -> float:
        if shrink_to is None:
            return x
        return _x_data_to_plot_shrink(x, min_plotted_x, shrink_to)

    x_hi_plot = xplt(x_hi_data)

    ax.set_title(title)
    ax.tick_params(axis="both", which="major", width=2.0, length=8)
    for spine in ax.spines.values():
        spine.set_linewidth(1.8)

    for sk in sub_keys:
        if sk not in max_series:
            continue
        label, mx, max_means, max_stds = max_series[sk]
        mx_plot = [xplt(x) for x in mx]
        ax.errorbar(
            mx_plot,
            max_means,
            yerr=max_stds,
            marker="o",
            linestyle="-",
            linewidth=3.2,
            markersize=10.0,
            markeredgewidth=1.8,
            capsize=7,
            elinewidth=2.4,
            capthick=2.4,
            label=label,
        )

    if use_bins:
        lx = float(min(all_plotted_x)) if all_plotted_x else 0.0
        rx = float(max(all_plotted_x)) if all_plotted_x else float(x_hi_plot)
        ax.set_xlim(lx - 0.55, max(float(x_hi_plot), rx + 0.55))
    else:
        ax.set_xlim(0.0, x_hi_plot)

    if shrink_to is None:
        if use_bins:
            ax.set_xticks([float(i) for i in range(num_ordinal_bins)])
            ax.set_xticklabels(ordinal_tick_labels if ordinal_tick_labels else [""])
        elif x_ticks_mode == "ends":
            ax.set_xticks([float(e) for e in x_tick_ends])
            ax.set_xticklabels([f"<{e}" for e in x_tick_ends])
        elif x_ticks_mode == "regular":
            step = max(int(x_tick_step), 1)
            hi = int(math.ceil(x_hi_data / step) * step)
            reg_ticks = [float(x) for x in range(0, hi + 1, step)]
            ax.set_xticks(reg_ticks)
            ax.set_xticklabels([str(int(t)) if t == int(t) else str(t) for t in reg_ticks])
    else:
        if use_bins:
            ax.set_xticks([float(i) for i in range(num_ordinal_bins)])
            ax.set_xticklabels(ordinal_tick_labels if ordinal_tick_labels else [""])
        elif x_ticks_mode == "ends":
            tick_data = sorted({float(e) for e in x_tick_ends})
            if not tick_data:
                tick_data = [0.0]
            elif tick_data[0] > 0:
                tick_data = [0.0, *tick_data]
            tick_plot = [xplt(t) for t in tick_data]
            labels = ["0" if t <= 0 else f"<{int(t)}" for t in tick_data]
            ax.set_xticks(tick_plot)
            ax.set_xticklabels(labels)
        elif x_ticks_mode == "regular":
            step = max(int(x_tick_step), 1)
            hi_d = int(math.ceil(x_hi_data / step) * step)
            reg_data = [float(x) for x in range(0, hi_d + 1, step)]
            tick_plot = [xplt(t) for t in reg_data]
            ax.set_xticks(tick_plot)
            ax.set_xticklabels([str(int(t)) if t == int(t) else str(t) for t in reg_data])
        _draw_x_shrink_marks(ax, shrink_to, x_hi_plot)

    ax.set_xlabel("Validation bin" if use_bins else "Validation length (upper bound)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(-10.0, 110.0)
    ax.grid(alpha=0.35, linewidth=1.2)
    if legend_loc != "none":
        ax.legend(loc=legend_loc, fontsize=16, frameon=False, markerscale=1.2)


def plot_tasks_df(
    df,
    *,
    tasks: list[str],
    titles: list[str],
    output_path: Path,
    legend_loc: str,
    group_by: list[str],
    group_label_mode: str,
    group_custom_labels: list[str],
    max_aggregation: str,
    x_ticks_mode: str,
    x_tick_step: int,
    x_axis_break: str | None,
    num_bins: int | None,
    merge_bins: bool,
    ncols: int,
    plot_size: tuple[float, float],
) -> None:
    """Plot one or more tasks; multitask uses a grid of axes sized by ``plot_size``."""
    try:
        import matplotlib as mpl
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ModuleNotFoundError as e:
        raise SystemExit(
            "Plotting requires pandas, matplotlib, and seaborn. Install them first."
        ) from e

    if not tasks:
        raise SystemExit("At least one task is required.")
    if len(titles) != len(tasks):
        raise SystemExit(
            f"Number of titles ({len(titles)}) must match number of tasks ({len(tasks)})."
        )
    if ncols < 1:
        raise SystemExit("--ncols must be >= 1.")

    prepared = [
        _prepare_task_plot(
            df,
            task=task,
            group_by=group_by,
            group_label_mode=group_label_mode,
            group_custom_labels=group_custom_labels,
            max_aggregation=max_aggregation,
            x_ticks_mode=x_ticks_mode,
            x_tick_step=x_tick_step,
            x_axis_break=x_axis_break,
            num_bins=num_bins,
            merge_bins=merge_bins,
        )
        for task in tasks
    ]

    n = len(tasks)
    ncols_eff = min(ncols, n)
    nrows = math.ceil(n / ncols_eff)
    fig_w = plot_size[0] * ncols_eff
    fig_h = plot_size[1] * nrows

    mpl.rcParams["axes.titleweight"] = "bold"
    sns.set_theme(style="whitegrid", palette="dark6", context="talk", font_scale=1.4)
    fig, axes = plt.subplots(
        nrows,
        ncols_eff,
        figsize=(fig_w, fig_h),
        squeeze=False,
    )

    for i, prep in enumerate(prepared):
        r, c = divmod(i, ncols_eff)
        _draw_task_on_ax(axes[r][c], prep, title=titles[i], legend_loc=legend_loc)

    for j in range(n, nrows * ncols_eff):
        r, c = divmod(j, ncols_eff)
        axes[r][c].set_visible(False)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_task_df(
    df,
    *,
    task: str,
    output_path: Path,
    title: str,
    legend_loc: str,
    group_by: list[str],
    group_label_mode: str,
    group_custom_labels: list[str],
    max_aggregation: str,
    x_ticks_mode: str,
    x_tick_step: int,
    x_axis_break: str | None,
    num_bins: int | None,
    merge_bins: bool,
    plot_size: tuple[float, float] | None = None,
) -> None:
    """Single-task wrapper around ``plot_tasks_df`` (default size 12x7)."""
    size = plot_size if plot_size is not None else (12.0, 7.0)
    plot_tasks_df(
        df,
        tasks=[task],
        titles=[title],
        output_path=output_path,
        legend_loc=legend_loc,
        group_by=group_by,
        group_label_mode=group_label_mode,
        group_custom_labels=group_custom_labels,
        max_aggregation=max_aggregation,
        x_ticks_mode=x_ticks_mode,
        x_tick_step=x_tick_step,
        x_axis_break=x_axis_break,
        num_bins=num_bins,
        merge_bins=merge_bins,
        ncols=1,
        plot_size=size,
    )


def _dump_selected_rows(df, *, group_by: list[str], selected_cols: list[str]) -> None:
    default_cols = [
        "task",
        *group_by,
        "model",
        "learning_rate",
        "num_bins",
        "train_range",
        "bucket",
        "accuracy",
    ]
    cols = selected_cols or [c for c in default_cols if c in df.columns]
    require_columns(df, cols, "--selected-cols")
    out_df = df[cols].drop_duplicates()
    sort_cols = [c for c in ("task", "bucket", "model", "learning_rate", "accuracy") if c in cols]
    if sort_cols:
        out_df = out_df.sort_values(sort_cols)
    print(out_df.to_string(index=False))
    print(f"\nSelected rows: {len(out_df)}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Create plots from CSV via pandas DataFrame filtering. "
            "Use --keep/--remove for column value filters and --query/--exclude-query "
            "for standard pandas expressions."
        )
    )
    repo_root = Path(__file__).resolve().parents[2]
    default_csv = repo_root / "exports" / "summary_results_9bins.csv"
    default_plot_dir = repo_root / "exports" / "plots"

    parser.add_argument("--input-csv", "--csv", dest="input_csv", type=Path, default=default_csv)
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help="Task value from CSV task column (required unless --multitask is set).",
    )
    parser.add_argument(
        "--multitask",
        type=str,
        default="",
        help=(
            "CSV-style comma-separated task names to plot in a grid. "
            'Example: --multitask "012_star_0_2_star,aa_star,ab_star_d_bc_star". '
            "Mutually exclusive with --task."
        ),
    )
    parser.add_argument(
        "--ncols",
        type=int,
        default=2,
        help="Number of subplot axes per row when using --multitask (default: 2).",
    )
    parser.add_argument(
        "--multititles",
        type=str,
        default="",
        help=(
            "CSV-style comma-separated titles for --multitask panels, in task order. "
            "Defaults to the task names when omitted."
        ),
    )
    parser.add_argument(
        "--plot-size",
        type=str,
        default=None,
        metavar="WxH",
        help=(
            "Size of each individual axis plot in inches as W,H or WxH. "
            "Single-task default: 12x7. Multitask default: 6x5. "
            "Figure size is ncols*W by nrows*H."
        ),
    )
    parser.add_argument(
        "--output",
        "--plot-path",
        dest="output_path",
        type=Path,
        default=None,
        help="Output path (default: exports/plots/<task>_csv_df.png).",
    )
    parser.add_argument("--title", type=str, default=None, help="Plot title (default: task).")
    parser.add_argument(
        "--legend-loc",
        type=str,
        default="best",
        choices=[
            "best", "upper right", "upper left", "lower left", "lower right",
            "right", "center left", "center right", "lower center",
            "upper center", "center", "none",
        ],
    )
    parser.add_argument(
        "--keep",
        action="append",
        default=[],
        help="Keep rows by exact column values: column=v1,v2 (repeatable).",
    )
    parser.add_argument(
        "--remove",
        action="append",
        default=[],
        help="Drop rows by exact column values: column=v1,v2 (repeatable).",
    )
    parser.add_argument(
        "--query",
        action="append",
        default=[],
        help="Keep rows matching pandas query expression (repeatable, AND across uses).",
    )
    parser.add_argument(
        "--exclude-query",
        action="append",
        default=[],
        help="Drop rows matching pandas query expression (repeatable).",
    )
    parser.add_argument(
        "--group-by",
        action="append",
        default=[],
        help=(
            "Column to group datapoints by (repeatable). "
            "Examples: --group-by arch --group-by pe. "
            "If omitted, each (model, learning_rate) is its own group."
        ),
    )
    parser.add_argument(
        "--dump-selected",
        action="store_true",
        help=(
            "Print selected rows/columns (query_summary_df-style) after filtering "
            "and before plotting."
        ),
    )
    parser.add_argument(
        "--selected-cols",
        action="append",
        default=[],
        help=(
            "Columns to print with --dump-selected (repeatable). "
            "Default: task, --group-by cols, model,learning_rate,bucket,accuracy."
        ),
    )
    parser.add_argument(
        "--dump-selected-only",
        action="store_true",
        help="Print selected rows and exit without generating a plot.",
    )
    parser.add_argument(
        "--group-label-mode",
        choices=("model", "group", "custom"),
        default="model",
        help=(
            "Legend base label mode per group: "
            "model (existing compact model label), "
            "group (column=value pairs from --group-by), "
            "custom (labels from --group-custom-labels)."
        ),
    )
    parser.add_argument(
        "--group-custom-labels",
        type=str,
        default="",
        help=(
            'CSV-style comma-separated custom labels for groups in first-appearance order, '
            'used when --group-label-mode=custom. Example: '
            '--group-custom-labels "Hybrid,SSM,Transformer".'
        ),
    )
    parser.add_argument(
        "--max-aggregation",
        choices=("pareto_mean", "bin_max", "mean", "max"),
        default="pareto_mean",
        help=(
            "How to compute grouped max series: "
            "pareto_mean = Pareto winner selection, then mean+std over winner runs; "
            "bin_max = direct maximum within each bin across all grouped datapoints. "
            "Aliases: mean->pareto_mean, max->bin_max."
        ),
    )
    parser.add_argument(
        "--num-bins",
        action="append",
        default=[],
        metavar="N",
        help=(
            "Only plot runs with exactly N validation bins. Repeatable / "
            "comma-separated to allow several counts, e.g. --num-bins 3,6."
        ),
    )
    parser.add_argument(
        "--min-bins",
        type=int,
        default=None,
        metavar="N",
        help="Only plot runs with at least N validation bins.",
    )
    parser.add_argument(
        "--max-bins",
        type=int,
        default=None,
        metavar="N",
        help="Only plot runs with at most N validation bins.",
    )
    parser.add_argument(
        "--train-range",
        action="append",
        default=[],
        metavar="LO-HI",
        help=(
            "Only plot runs whose training bin (first bin) is one of these ranges, "
            "e.g. --train-range 0-49,0-24 (repeatable)."
        ),
    )
    parser.add_argument(
        "--exclude-train-range",
        action="append",
        default=[],
        metavar="LO-HI",
        help="Drop runs whose training bin is one of these ranges (repeatable).",
    )
    parser.add_argument(
        "--bins",
        action="append",
        default=[],
        metavar="LO-HI|LO-HI|...",
        help=(
            "Only plot runs with exactly this bin layout, e.g. "
            '--bins "0-49|50-99|100-149" (repeat the flag to allow several layouts).'
        ),
    )
    parser.add_argument(
        "--first-bins",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Plot only the first N bins of each run, dropping longer ones. Useful "
            "for comparing runs whose bin counts differ."
        ),
    )
    parser.add_argument(
        "--max-bin-len",
        type=int,
        default=None,
        metavar="LEN",
        help="Drop bins whose upper bound exceeds LEN (e.g. 149).",
    )
    parser.add_argument(
        "--x-ticks",
        dest="x_ticks_mode",
        choices=("ends", "regular", "bins"),
        default="bins",
    )
    parser.add_argument("--x-tick-step", type=int, default=10)
    parser.add_argument("--x-axis-break", type=str, default=None, metavar="POS")
    parser.add_argument(
        "--merge-bins",
        action="store_true",
        help=(
            "Merge datapoints by ordinal validation bin (bin0, bin1, ...) instead of "
            "splitting series when bucket interval names differ. Requires --x-ticks bins."
        ),
    )
    args = parser.parse_args()

    if args.x_tick_step < 1:
        raise SystemExit("--x-tick-step must be >= 1.")
    if args.min_bins is not None and args.min_bins < 1:
        raise SystemExit("--min-bins must be >= 1.")
    if args.max_bins is not None and args.max_bins < 1:
        raise SystemExit("--max-bins must be >= 1.")
    if (
        args.min_bins is not None
        and args.max_bins is not None
        and args.min_bins > args.max_bins
    ):
        raise SystemExit("--min-bins must not exceed --max-bins.")
    if args.first_bins is not None and args.first_bins < 1:
        raise SystemExit("--first-bins must be >= 1.")
    if args.ncols < 1:
        raise SystemExit("--ncols must be >= 1.")
    custom_labels = _parse_custom_group_labels(args.group_custom_labels)
    if args.group_label_mode != "custom" and custom_labels:
        raise SystemExit("--group-custom-labels is only valid with --group-label-mode custom.")
    if args.max_aggregation == "mean":
        args.max_aggregation = "pareto_mean"
    elif args.max_aggregation == "max":
        args.max_aggregation = "bin_max"

    multitask = _parse_custom_group_labels(args.multitask)
    if multitask and args.task:
        raise SystemExit("Use either --task or --multitask, not both.")
    if not multitask and not args.task:
        raise SystemExit("Provide --task or --multitask.")

    multititles = _parse_custom_group_labels(args.multititles)
    plot_size = _parse_plot_size(args.plot_size)

    if multitask:
        tasks = multitask
        if multititles:
            if len(multititles) != len(tasks):
                raise SystemExit(
                    f"--multititles count ({len(multititles)}) must match "
                    f"--multitask count ({len(tasks)})."
                )
            titles = multititles
        else:
            titles = list(tasks)
        if args.title is not None:
            raise SystemExit("Use --multititles with --multitask, not --title.")
        size = plot_size if plot_size is not None else (6.0, 5.0)
        default_name = "multitask_csv_df.png" if len(tasks) > 1 else f"{tasks[0]}_csv_df.png"
    else:
        tasks = [args.task]
        titles = [args.title or args.task]
        if multititles:
            raise SystemExit("--multititles requires --multitask.")
        size = plot_size if plot_size is not None else (12.0, 7.0)
        default_name = f"{args.task}_csv_df.png"

    bin_filter = BinFilter(
        bin_counts=normalize_int_tokens(args.num_bins, "--num-bins"),
        min_bins=args.min_bins,
        max_bins=args.max_bins,
        train_ranges=normalize_range_tokens(args.train_range, "--train-range"),
        exclude_train_ranges=normalize_range_tokens(
            args.exclude_train_range, "--exclude-train-range"
        ),
        bin_signatures=normalize_bin_signatures(args.bins, "--bins"),
        first_bins=args.first_bins,
        max_bin_upper=args.max_bin_len,
    )

    df = load_summary_dataframe(args.input_csv)
    df = apply_keep_remove_filters(df, args.keep, args.remove)
    df = apply_query_filters(df, args.query, args.exclude_query)
    if df.empty:
        raise SystemExit("No rows left after DataFrame filtering.")
    df = bin_filter.apply(df)
    if df.empty:
        raise SystemExit(f"No runs left after bin filtering ({bin_filter.describe()}).")
    if args.dump_selected or args.dump_selected_only:
        _dump_selected_rows(df, group_by=args.group_by, selected_cols=args.selected_cols)
        if args.dump_selected_only:
            return 0

    plot_path = args.output_path or (default_plot_dir / default_name)
    plot_tasks_df(
        df,
        tasks=tasks,
        titles=titles,
        output_path=plot_path,
        legend_loc=args.legend_loc,
        group_by=args.group_by,
        group_label_mode=args.group_label_mode,
        group_custom_labels=custom_labels,
        max_aggregation=args.max_aggregation,
        x_ticks_mode=args.x_ticks_mode,
        x_tick_step=args.x_tick_step,
        x_axis_break=args.x_axis_break,
        num_bins=None,  # already applied via BinFilter on the whole frame
        merge_bins=args.merge_bins,
        ncols=args.ncols if multitask else 1,
        plot_size=size,
    )
    print(f"Wrote plot: {plot_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
