#!/usr/bin/env python3
"""Plot accuracy vs. validation bucket from a summary CSV (e.g. 9-bin exports).

Filtering uses :func:`generate_summary_csv.filter_rows` (same CLI semantics as
:mod:`generate_summary_plots`).  Bucket x-coordinates, max-envelope logic, and
legend compaction from parsed model specs are defined in this module.

**Bin-homogeneous groups:** rows matched by the same ``--group-pattern`` are
split into separate plotted series whenever their ``(model, learning_rate)``
datapoints disagree on the *set* of bucket columns they cover.  That way a
group never mixes configurations that were evaluated on different bins; each
line only pools datapoints that share identical bin coverage.
"""
from __future__ import annotations

import argparse
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean

from generate_summary_csv import (
    filter_rows,
    load_csv_rows,
    parse_model_spec,
    row_matches_spec_pattern,
)


# ---------------------------------------------------------------------------
# Stats + bucket parsing (same semantics as legacy plot helpers)
# ---------------------------------------------------------------------------


def _sample_std(values: list[float]) -> float:
    n = len(values)
    if n <= 1:
        return 0.0
    mu = mean(values)
    var = sum((v - mu) ** 2 for v in values) / (n - 1)
    return math.sqrt(var)


_BUCKET_BOUNDS_RE = re.compile(r"^(\d+)\s*-\s*(\d+)$")


def _parse_bucket_bounds(bucket: str) -> tuple[int, int] | None:
    m = _BUCKET_BOUNDS_RE.match(bucket.strip())
    if not m:
        return None
    lo, hi = int(m.group(1)), int(m.group(2))
    if lo > hi:
        return None
    return (lo, hi)


def _bucket_upper_bound(bucket: str) -> int | None:
    bounds = _parse_bucket_bounds(bucket)
    return bounds[1] if bounds else None


def _bucket_width(bucket: str) -> int | None:
    bounds = _parse_bucket_bounds(bucket)
    if not bounds:
        return None
    lo, hi = bounds
    return hi - lo + 1


def _bucket_sort_key_plot(bucket: str) -> tuple[int, str]:
    ub = _bucket_upper_bound(bucket)
    return (ub if ub is not None else -1, bucket)


def _bucket_plot_x(bucket: str) -> float | None:
    u = _bucket_upper_bound(bucket)
    return float(u) if u is not None else None


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
    cands = [
        b
        for b in _buckets_for_datapoint(dp, dcv)
        if _bucket_upper_bound(b) == end
    ]
    if not cands:
        return []
    w_min = min(w for b in cands if (w := _bucket_width(b)) is not None)
    return [b for b in cands if _bucket_width(b) == w_min]


def _value_at_end_for_dp(
    dp: tuple[str, float], end: int, dcv: dict[tuple[str, float, str], list[float]]
) -> float:
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
    if all_ends_override is not None:
        all_ends = sorted(set(all_ends_override))
    else:
        ends_set: set[int] = set()
        for dp in datapoints:
            for b in _buckets_for_datapoint(dp, dcv):
                e = _bucket_upper_bound(b)
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
                if _bucket_upper_bound(b) != e:
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


# ---------------------------------------------------------------------------
# Legend: compact specs + merge identical prefixes across pooled models
# ---------------------------------------------------------------------------

_NUM_SUFFIX_RE = re.compile(r"^(.+)(l|h|d|dr|mlp)$")


def _spec_keyed_parts(row: dict[str, str]) -> list[tuple[str, str]]:
    """Ordered (key, fragment) pairs matching :func:`compact_spec_label_from_row` order."""
    spec = parse_model_spec(row.get("model", ""))
    out: list[tuple[str, str]] = []
    arch = spec.get("arch", "-")
    if arch != "-":
        out.append(("arch", arch))
    if arch == "hyb":
        lo = spec.get("layer_order", "-")
        if lo not in ("-", ""):
            out.append(("layer_order", lo))
    kern = spec.get("kernel", "-")
    if kern not in ("-", ""):
        out.append(("kernel", kern))
    ly = spec.get("layers", "-")
    if ly not in ("-", ""):
        out.append(("layers", f"{ly}l"))
    h = spec.get("heads", "-")
    if h not in ("-", ""):
        out.append(("heads", f"{h}h"))
    d = spec.get("d_model", "-")
    if d not in ("-", ""):
        out.append(("d_model", f"{d}d"))
    dr = spec.get("dropout", "-")
    if dr not in ("-", ""):
        out.append(("dropout", f"{dr}dr"))
    mlp = spec.get("mlp_size", "-")
    if mlp not in ("-", ""):
        out.append(("mlp_size", f"{mlp}mlp"))
    pe = spec.get("pe", "-")
    if pe == "True":
        out.append(("pe", "pe"))
    elif pe == "False":
        out.append(("pe", "nope"))
    ln = spec.get("ln", "-")
    if ln == "True":
        out.append(("ln", "ln"))
    elif ln == "False":
        out.append(("ln", "noln"))
    stp = spec.get("train_steps_k", "-")
    if stp not in ("-", ""):
        out.append(("train_steps_k", f"stp{stp}k"))
    try:
        lr = float(row.get("learning_rate", "nan"))
        out.append(("lr", f"{lr:g}lr"))
    except (ValueError, TypeError):
        out.append(("lr", f"{row.get('learning_rate', '')}lr"))
    return out


def _merge_fragments_for_key(key: str, frags: set[str]) -> str:
    if len(frags) == 1:
        return next(iter(frags))
    if key == "lr":
        bodies: list[str] = []
        suffix_ok = True
        for f in frags:
            if f.endswith("lr") and len(f) > 2:
                bodies.append(f[: -len("lr")])
            else:
                suffix_ok = False
                break
        if suffix_ok and bodies:
            return "/".join(sorted(bodies, key=lambda s: float(s) if _is_float_str(s) else s)) + "lr"
    m_groups: dict[str, list[str]] = defaultdict(list)
    unmerged: list[str] = []
    for f in frags:
        m = _NUM_SUFFIX_RE.match(f)
        if m:
            m_groups[m.group(2)].append(m.group(1))
        else:
            unmerged.append(f)
    if len(m_groups) == 1 and not unmerged:
        sfx, nums = next(iter(m_groups.items()))
        return "/".join(sorted(nums, key=_numeric_sort_key)) + sfx
    return "/".join(sorted(frags))


def _is_float_str(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


def _numeric_sort_key(s: str) -> tuple[int, float | str]:
    try:
        return (0, float(s))
    except ValueError:
        return (1, s)


def _legend_label_merged(rows: list[dict[str, str]]) -> str:
    """One legend entry per series: shared spec prefix once, varying fields like ``1/4l``."""
    if not rows:
        return ""
    reps: dict[tuple[str, float], dict[str, str]] = {}
    for r in rows:
        dp = (r["model"], float(r["learning_rate"]))
        reps[dp] = r
    keyed_lists = [_spec_keyed_parts(r) for r in reps.values()]
    key_sets = {frozenset(k for k, _ in kl) for kl in keyed_lists}
    if len(key_sets) > 1:
        labels = sorted({compact_spec_label_from_row(r) for r in reps.values()})
        return " | ".join(labels)
    key_order = [k for k, _ in keyed_lists[0]]
    by_key: dict[str, set[str]] = defaultdict(set)
    for kl in keyed_lists:
        for k, frag in kl:
            by_key[k].add(frag)
    out_parts: list[str] = []
    for k in key_order:
        frags = by_key[k]
        if not frags:
            continue
        out_parts.append(_merge_fragments_for_key(k, frags))
    return "".join(out_parts)


def compact_spec_label_from_row(row: dict[str, str]) -> str:
    spec = parse_model_spec(row.get("model", ""))
    parts: list[str] = []
    arch = spec.get("arch", "-")
    if arch != "-":
        parts.append(arch)
    if arch == "hyb":
        lo = spec.get("layer_order", "-")
        if lo not in ("-", ""):
            parts.append(lo)
    kern = spec.get("kernel", "-")
    if kern not in ("-", ""):
        parts.append(kern)
    ly = spec.get("layers", "-")
    if ly not in ("-", ""):
        parts.append(f"{ly}l")
    h = spec.get("heads", "-")
    if h not in ("-", ""):
        parts.append(f"{h}h")
    d = spec.get("d_model", "-")
    if d not in ("-", ""):
        parts.append(f"{d}d")
    dr = spec.get("dropout", "-")
    if dr not in ("-", ""):
        parts.append(f"{dr}dr")
    mlp = spec.get("mlp_size", "-")
    if mlp not in ("-", ""):
        parts.append(f"{mlp}mlp")
    pe = spec.get("pe", "-")
    if pe == "True":
        parts.append("pe")
    elif pe == "False":
        parts.append("nope")
    ln = spec.get("ln", "-")
    if ln == "True":
        parts.append("ln")
    elif ln == "False":
        parts.append("noln")
    stp = spec.get("train_steps_k", "-")
    if stp not in ("-", ""):
        parts.append("stp")
        parts.append(f"{stp}k")
    try:
        lr = float(row.get("learning_rate", "nan"))
        parts.append(f"{lr:g}lr")
    except (ValueError, TypeError):
        parts.append(f"{row.get('learning_rate', '')}lr")
    return "".join(parts)


def legend_label_from_rows(rows: list[dict[str, str]]) -> str:
    """Compact label for a pooled series (varying hyperparams folded, e.g. ``1/4l``)."""
    if not rows:
        return ""
    dps = {(r["model"], float(r["learning_rate"])) for r in rows}
    if len(dps) == 1:
        r0 = next(r for r in rows if (r["model"], float(r["learning_rate"])) in dps)
        return compact_spec_label_from_row(r0)
    return _legend_label_merged(rows)


def _assign_series_id(
    row: dict[str, str],
    explicit_patterns: list[str],
    wildcard_rest: bool,
) -> str:
    for pattern in explicit_patterns:
        if row_matches_spec_pattern(row, pattern):
            return f"p:{pattern}"
    if wildcard_rest:
        return "p:*"
    m = row["model"]
    lr = row["learning_rate"]
    return f"solo:{m}|{lr}"


def _dedupe_legend_labels(ids: list[str], id_to_label: dict[str, str]) -> dict[str, str]:
    seen: dict[str, int] = {}
    out: dict[str, str] = {}
    for sid in ids:
        base = id_to_label[sid]
        n = seen.get(base, 0)
        seen[base] = n + 1
        out[sid] = base if n == 0 else f"{base} ({n + 1})"
    return out


def _datapoint_bucket_set(
    rows_for_sid: list[dict[str, str]],
) -> dict[tuple[str, float], frozenset[str]]:
    """For each (model, lr), the set of *bucket* names present in *rows_for_sid*."""
    acc: dict[tuple[str, float], set[str]] = defaultdict(set)
    for row in rows_for_sid:
        dp = (row["model"], float(row["learning_rate"]))
        acc[dp].add(row["bucket"])
    return {dp: frozenset(bs) for dp, bs in acc.items()}


def _split_rows_by_bin_signature(
    rows_for_sid: list[dict[str, str]],
) -> dict[frozenset[str], list[dict[str, str]]]:
    """Partition *rows_for_sid* by the bucket-set signature shared by each datapoint."""
    dp_to_bins = _datapoint_bucket_set(rows_for_sid)
    by_sig: dict[frozenset[str], list[dict[str, str]]] = defaultdict(list)
    for row in rows_for_sid:
        dp = (row["model"], float(row["learning_rate"]))
        sig = dp_to_bins[dp]
        by_sig[sig].append(row)
    return dict(by_sig)


def _signature_x_ends(signature: frozenset[str]) -> list[int]:
    ends: list[int] = []
    for b in signature:
        u = _bucket_upper_bound(b)
        if u is not None:
            ends.append(u)
    return sorted(set(ends))


def _signature_label(signature: frozenset[str]) -> str:
    """Short legend suffix: upper bounds at each bin, in bucket order."""
    ordered = sorted(signature, key=_bucket_sort_key_plot)
    parts: list[str] = []
    for b in ordered:
        u = _bucket_upper_bound(b)
        parts.append(str(u) if u is not None else b)
    return ",".join(parts)


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
    *,
    x_ticks_mode: str,
    x_tick_step: int,
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

    datapoint_bucket_vals: dict[tuple[str, float, str], list[float]] = defaultdict(list)
    for row in filtered_rows:
        key = (row["model"], float(row["learning_rate"]), row["bucket"])
        datapoint_bucket_vals[key].append(float(row["accuracy"]))

    explicit_patterns = [p for p in group_patterns if p.strip() and p.strip() != "*"]
    wildcard_rest = any(p.strip() == "*" for p in group_patterns)

    series_id_to_rows: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in filtered_rows:
        sid = _assign_series_id(row, explicit_patterns, wildcard_rest)
        series_id_to_rows[sid].append(row)

    # Sub-series: (base_sid, bin_signature) so each line only pools same-bin datapoints.
    sub_series_rows: dict[tuple[str, frozenset[str]], list[dict[str, str]]] = {}
    for sid, sid_rows in series_id_to_rows.items():
        for sig, sig_rows in _split_rows_by_bin_signature(sid_rows).items():
            sub_series_rows[(sid, sig)] = sig_rows

    def sub_key_sort(k: tuple[str, frozenset[str]]) -> tuple:
        sid, sig = k
        return (legend_label_from_rows(sub_series_rows[k]), sid, _signature_label(sig))

    sub_keys = sorted(sub_series_rows.keys(), key=sub_key_sort)
    base_labels = {
        sk: legend_label_from_rows(sub_series_rows[sk])
        for sk in sub_keys
    }
    sig_count_by_sid: dict[str, int] = defaultdict(int)
    for sid, _sig in sub_keys:
        sig_count_by_sid[sid] += 1
    display_labels: dict[tuple[str, frozenset[str]], str] = {}
    for sk in sub_keys:
        sid, sig = sk
        base = base_labels[sk]
        if sig_count_by_sid[sid] > 1:
            display_labels[sk] = f"{base} [ends {_signature_label(sig)}]"
        else:
            display_labels[sk] = base

    id_strs = [f"{sid}::{_signature_label(sig)}" for sid, sig in sub_keys]
    deduped = _dedupe_legend_labels(id_strs, {id_strs[i]: display_labels[sub_keys[i]] for i in range(len(sub_keys))})
    sub_key_to_display = {sub_keys[i]: deduped[id_strs[i]] for i in range(len(sub_keys))}

    series_to_bucket_vals: dict[tuple[tuple[str, frozenset[str]], str], list[float]] = defaultdict(list)
    series_to_datapoints: dict[tuple[str, frozenset[str]], set[tuple[str, float]]] = defaultdict(set)

    for sk in sub_keys:
        for row in sub_series_rows[sk]:
            dp = (row["model"], float(row["learning_rate"]))
            series_to_datapoints[sk].add(dp)
            series_to_bucket_vals[(sk, row["bucket"])].append(float(row["accuracy"]))

    all_bucket_names = {r["bucket"] for r in filtered_rows}
    x_tick_ends = sorted(
        {int(x) for b in all_bucket_names if (x := _bucket_plot_x(b)) is not None}
    )
    if not x_tick_ends:
        raise SystemExit(
            "No parseable bucket ranges in filtered rows (expected names like '0-50')."
        )

    def series_buckets(sk: tuple[str, frozenset[str]]) -> list[str]:
        return sorted(
            {b for (k, b) in series_to_bucket_vals if k == sk},
            key=_bucket_sort_key_plot,
        )

    max_series: dict[tuple[str, frozenset[str]], tuple[str, list[float], list[float], list[float]]] = {}
    if include_max or include_max_only:
        for sk in sub_keys:
            datapoints = series_to_datapoints.get(sk, set())
            if not datapoints:
                continue
            sig = sk[1]
            ends_for_sig = _signature_x_ends(sig)
            pruned, _ = select_max_winners_for_series(
                datapoints,
                datapoint_bucket_vals,
                all_ends_override=ends_for_sig or x_tick_ends,
            )
            if not pruned:
                continue
            win_rows = [
                r
                for r in sub_series_rows[sk]
                if (r["model"], float(r["learning_rate"])) in pruned
            ]
            max_body = legend_label_from_rows(win_rows) or sub_key_to_display[sk]
            max_label = f"max:{max_body}"
            mx, mmean, mstd = max_line_xy_for_winners(
                pruned,
                ends_for_sig or x_tick_ends,
                datapoint_bucket_vals,
                fallback_dps=datapoints,
            )
            max_series[sk] = (max_label, mx, mmean, mstd)

    x_max_data = float(max(x_tick_ends))
    for sk in sub_keys:
        for b in series_buckets(sk):
            if (x_b := _bucket_plot_x(b)) is not None:
                x_max_data = max(x_max_data, x_b)
    for _lbl, mx_pts, _, _ in max_series.values():
        if mx_pts:
            x_max_data = max(x_max_data, max(mx_pts))

    fig, ax = plt.subplots(figsize=(12, 7))

    if x_ticks_mode == "ends":
        x_ticks = [float(e) for e in x_tick_ends]
        x_labels = [f"<{e}" for e in x_tick_ends]
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_labels)
    else:
        step = max(int(x_tick_step), 1)
        hi = int(math.ceil(x_max_data / step) * step)
        reg_ticks = [float(x) for x in range(0, hi + 1, step)]
        ax.set_xticks(reg_ticks)
        ax.set_xticklabels([str(int(t)) if t == int(t) else str(t) for t in reg_ticks])

    stem_colors: dict[tuple[str, frozenset[str]], str] = {}
    for sk in sub_keys:
        label = sub_key_to_display[sk]
        buckets_sl = series_buckets(sk)
        if not include_max_only:
            xs_stem: list[float] = []
            means: list[float] = []
            stds: list[float] = []
            for b in buckets_sl:
                x = _bucket_plot_x(b)
                if x is None:
                    continue
                vals = series_to_bucket_vals.get((sk, b), [])
                if vals:
                    xs_stem.append(x)
                    means.append(mean(vals) * 100.0)
                    stds.append(_sample_std(vals) * 100.0)
                else:
                    xs_stem.append(x)
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
                label=label,
            )
            stem_colors[sk] = stem_line.lines[0].get_color()

    for sk in sub_keys:
        if sk not in max_series:
            continue
        max_label, mx, max_means, max_stds = max_series[sk]
        dotted_color = stem_colors.get(sk)
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
            "Create per-task plots from a summary CSV. "
            "Filtering matches generate_summary_csv / generate_summary_plots. "
            "Each --group-pattern series is split by identical bucket coverage."
        )
    )
    repo_root = Path(__file__).resolve().parents[2]
    default_csv = repo_root / "exports" / "summary_results_9bins.csv"
    default_plot_dir = repo_root / "exports" / "plots"

    parser.add_argument(
        "--input-csv",
        "--csv",
        dest="input_csv",
        type=Path,
        default=default_csv,
        help="Input CSV path. Default: exports/summary_results_9bins.csv",
    )
    parser.add_argument(
        "--task",
        type=str,
        help="Task name in the CSV ``task`` column (required).",
    )
    parser.add_argument(
        "--output",
        "--plot-path",
        dest="output_path",
        type=Path,
        default=None,
        help="Output plot file. Default: exports/plots/<task>_csv.png",
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
        help="Legend placement (or 'none' to hide).",
    )
    parser.add_argument(
        "--remove-pattern",
        action="append",
        default=[],
        help="Drop rows matching any pattern (first filtering stage).",
    )
    parser.add_argument(
        "--include-pattern",
        action="append",
        default=[],
        help="After removes, keep rows matching any pattern.",
    )
    parser.add_argument(
        "--group-pattern",
        action="append",
        default=[],
        help=(
            "Aggregate matching rows into one series; split again if bin coverage differs. "
            "First match wins; use '*' for the rest."
        ),
    )
    parser.add_argument(
        "--x-ticks",
        dest="x_ticks_mode",
        choices=("ends", "regular"),
        default="ends",
        help="ends: tick at each bin upper bound with <N> labels (default).",
    )
    parser.add_argument(
        "--x-tick-step",
        type=int,
        default=10,
        help="Spacing for --x-ticks=regular.",
    )
    max_group = parser.add_mutually_exclusive_group()
    max_group.add_argument(
        "--include-max",
        action="store_true",
        help="Per sub-series: dotted max:… line (same color as stem).",
    )
    max_group.add_argument(
        "--include-max-only",
        action="store_true",
        help="Plot only dotted max lines.",
    )
    args = parser.parse_args()

    if not args.task:
        raise SystemExit("--task is required.")
    if args.x_tick_step < 1:
        raise SystemExit("--x-tick-step must be >= 1.")

    rows = load_csv_rows(args.input_csv)
    if not rows:
        raise SystemExit(
            f"No CSV rows found at {args.input_csv}. "
            "Point --input-csv at a merged summary file."
        )

    plot_path = args.output_path or (default_plot_dir / f"{args.task}_csv.png")
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
        x_ticks_mode=args.x_ticks_mode,
        x_tick_step=args.x_tick_step,
    )
    print(f"Wrote plot: {plot_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
