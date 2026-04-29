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

If the CSV lists the same ``(model, learning_rate, bucket)`` more than once
(e.g. repeated summary lines or runs), those rows are collapsed to a single
value: the **maximum** accuracy, before any plotting or max-envelope logic.

With ``--x-ticks bins``, x is ordinal (shortest train span first); ticks are labeled ``bin1``, ``bin2``, … .
"""
from __future__ import annotations

import argparse
import math
import re
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from statistics import mean

from generate_summary_csv import (
    MODEL_SPEC_COLUMNS,
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


def _collapse_identical_datapoint_buckets(
    dcv: dict[tuple[str, float, str], list[float]],
) -> None:
    """In-place: one value per ``(model, learning_rate, bucket)`` — keep max accuracy."""
    for key in list(dcv.keys()):
        vals = dcv[key]
        if not vals:
            del dcv[key]
        else:
            m = max(vals)
            dcv[key] = [m]


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


def _ordinal_bin_tick_labels(
    sub_keys: list[tuple[str, frozenset[str]]],
    series_buckets: Callable[[tuple[str, frozenset[str]]], list[str]],
) -> list[str]:
    """``bin1``, ``bin2``, … for each ordinal bin (shortest train span first per series)."""
    if not sub_keys:
        return []
    max_k = max(len(series_buckets(sk)) for sk in sub_keys)
    return [f"bin{i + 1}" for i in range(max_k)]


def _filter_rows_by_num_bins(
    rows: list[dict[str, str]], num_bins: int
) -> list[dict[str, str]]:
    """Keep rows whose (model, learning_rate) pair has exactly ``num_bins`` distinct buckets."""
    buckets_by_dp: dict[tuple[str, float], set[str]] = defaultdict(set)
    for r in rows:
        buckets_by_dp[(r["model"], float(r["learning_rate"]))].add(r["bucket"])
    keep = {dp for dp, bs in buckets_by_dp.items() if len(bs) == num_bins}
    return [r for r in rows if (r["model"], float(r["learning_rate"])) in keep]


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


def _spec_for_legend(row: dict[str, str]) -> dict[str, str]:
    """Model spec from the model string, with CSV cells filling any ``-`` holes."""
    spec: dict[str, str] = dict(parse_model_spec(row.get("model", "")))
    for key in MODEL_SPEC_COLUMNS:
        v = spec.get(key, "-")
        if v not in ("-", "", None):
            continue
        rv = row.get(key)
        if rv is None or rv == "" or rv == "-":
            continue
        spec[key] = str(rv).strip()
    return spec


def _spec_keyed_parts(row: dict[str, str]) -> list[tuple[str, str]]:
    """Ordered (key, fragment) pairs matching :func:`compact_spec_label_from_row` order."""
    spec = _spec_for_legend(row)
    out: list[tuple[str, str]] = []
    arch = spec.get("arch", "-")
    if arch != "-":
        out.append(("arch", arch))
    if "hyb" in arch:
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
    ne = spec.get("ne", "-")
    if ne == "True":
        out.append(("ne", "ne"))
    elif ne == "False":
        out.append(("ne", "none"))
    stp = spec.get("train_steps_k", "-")
    if stp not in ("-", ""):
        out.append(("train_steps_k", f"stp{stp}k"))
    try:
        lr = float(row.get("learning_rate", "nan"))
        out.append(("lr", f"{lr:g}lr"))
    except (ValueError, TypeError):
        out.append(("lr", f"{row.get('learning_rate', '')}lr"))
    return out


# Order matches :func:`_spec_keyed_parts` emission order.
_KEY_ORDER_LEGEND: tuple[str, ...] = (
    "arch",
    "layer_order",
    "kernel",
    "layers",
    "heads",
    "d_model",
    "dropout",
    "mlp_size",
    "pe",
    "ln",
    "ne",
    "train_steps_k",
    "lr",
)


def _merge_fragments_for_key(key: str, frags: set[str], *, alt_sep: str = "/") -> str:
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
    return alt_sep.join(sorted(frags))


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
    """One legend entry: same layout as a single spec; ``|`` only joins differing fragments per field."""
    if not rows:
        return ""
    reps: dict[tuple[str, float], dict[str, str]] = {}
    for r in rows:
        dp = (r["model"], float(r["learning_rate"]))
        reps[dp] = r
    keyed_dicts = [dict(_spec_keyed_parts(r)) for r in reps.values()]
    all_keys: set[str] = set()
    for d in keyed_dicts:
        all_keys |= d.keys()
    key_order = [k for k in _KEY_ORDER_LEGEND if k in all_keys]
    key_order.extend(sorted(all_keys - set(key_order)))
    by_key: dict[str, set[str]] = defaultdict(set)
    for d in keyed_dicts:
        for k, frag in d.items():
            by_key[k].add(frag)
    out_parts: list[str] = []
    for k in key_order:
        frags = by_key[k]
        if not frags:
            continue
        out_parts.append(_merge_fragments_for_key(k, frags, alt_sep="|"))
    return "".join(out_parts)


def compact_spec_label_from_row(row: dict[str, str]) -> str:
    spec = _spec_for_legend(row)
    parts: list[str] = []
    arch = spec.get("arch", "-")
    if arch != "-":
        parts.append(arch)
    if "hyb" in arch:
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
    ne = spec.get("ne", "-")
    if ne == "True":
        parts.append("ne")
    elif ne == "False":
        parts.append("none")
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


def _parse_x_axis_shrink(
    raw: str | None,
    x_min_data: float,
) -> float | None:
    """Return plot-space length for compressing [0, x_min_data); first point lands here.

    ``--x-axis-break 5`` maps data ``0 .. x_min_data`` linearly onto plot ``0 .. 5``,
    then continues with unit slope. ``auto`` uses 5.
    """
    if raw is None or not str(raw).strip():
        return None
    s = str(raw).strip().lower()
    if s == "auto":
        val = 5.0
    else:
        try:
            val = float(s)
        except ValueError:
            raise SystemExit(
                f"--x-axis-break: invalid value {raw!r} (use a positive number or 'auto')."
            ) from None
    if val <= 0 or not math.isfinite(val):
        return None
    if x_min_data <= 0 or not math.isfinite(x_min_data):
        return None
    return val


def _x_data_to_plot_shrink(x: float, x_min_data: float, shrink: float) -> float:
    if x <= x_min_data:
        return (x / x_min_data) * shrink
    return shrink + (x - x_min_data)


def _draw_x_shrink_marks(ax, br_plot: float, x_extent: float) -> None:
    """Draw two tight // pairs near the middle of ``[0, br_plot]`` (compressed region before first point)."""
    from matplotlib.transforms import blended_transform_factory

    if br_plot <= 0:
        return
    trans = blended_transform_factory(ax.transData, ax.transAxes)
    d = 0.015
    # Slash width from compressed segment, not full axis (avoids huge diagonals on long x-axes).
    dx_slash = max(br_plot * 0.10, min(0.45, x_extent * 0.004))
    kwargs: dict = dict(transform=trans, color="k", clip_on=False, linewidth=0.9, zorder=10)

    center = 0.5 * br_plot
    pair_gap = max(br_plot * 0.06, 0.12)
    cx_lo = center - 0.5 * pair_gap
    cx_hi = center + 0.5 * pair_gap
    margin = dx_slash * 1.15
    if cx_lo < margin:
        shift = margin - cx_lo
        cx_lo += shift
        cx_hi += shift
    if cx_hi > br_plot - margin:
        shift = cx_hi - (br_plot - margin)
        cx_lo -= shift
        cx_hi -= shift
    cx_lo = max(cx_lo, margin)
    cx_hi = min(cx_hi, br_plot - margin)
    if cx_hi - cx_lo < 0.25 * pair_gap:
        cx_lo = max(margin, center - 0.4 * pair_gap)
        cx_hi = min(br_plot - margin, center + 0.4 * pair_gap)

    for cx in (cx_lo, cx_hi):
        ax.plot((cx - dx_slash, cx + dx_slash), (-d, +d), **kwargs)
        ax.plot((cx - dx_slash, cx + dx_slash), (1 - d, 1 + d), **kwargs)


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
    x_axis_break: str | None = None,
    num_bins: int | None = None,
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
    if num_bins is not None:
        before_nb = len(filtered_rows)
        filtered_rows = _filter_rows_by_num_bins(filtered_rows, num_bins)
        print(
            f"After --num-bins={num_bins}: {len(filtered_rows)} rows "
            f"(dropped configurations with other distinct-bucket counts; had {before_nb})."
        )
    if not filtered_rows:
        raise SystemExit(
            f"No rows to plot for task '{task}'. "
            "Check --task, --include-pattern, --remove-pattern, and --num-bins."
        )

    if x_ticks_mode == "bins" and x_axis_break is not None and str(x_axis_break).strip():
        raise SystemExit("--x-axis-break is incompatible with --x-ticks bins.")

    datapoint_bucket_vals: dict[tuple[str, float, str], list[float]] = defaultdict(list)
    for row in filtered_rows:
        key = (row["model"], float(row["learning_rate"]), row["bucket"])
        datapoint_bucket_vals[key].append(float(row["accuracy"]))
    _collapse_identical_datapoint_buckets(datapoint_bucket_vals)

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
        seen_dp_bucket: set[tuple[tuple[str, float], str]] = set()
        for row in sub_series_rows[sk]:
            dp = (row["model"], float(row["learning_rate"]))
            series_to_datapoints[sk].add(dp)
            b = row["bucket"]
            if (dp, b) in seen_dp_bucket:
                continue
            seen_dp_bucket.add((dp, b))
            collapsed = datapoint_bucket_vals.get((dp[0], dp[1], b), [])
            if collapsed:
                series_to_bucket_vals[(sk, b)].append(collapsed[0])

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

    use_bins = x_ticks_mode == "bins"

    def bucket_x(sk: tuple[str, frozenset[str]], b: str) -> float | None:
        if use_bins:
            return float(series_buckets(sk).index(b))
        return _bucket_plot_x(b)

    ordinal_tick_labels: list[str]
    num_ordinal_bins: int
    if use_bins:
        ordinal_tick_labels = _ordinal_bin_tick_labels(sub_keys, series_buckets)
        num_ordinal_bins = len(ordinal_tick_labels)
    else:
        ordinal_tick_labels = []
        num_ordinal_bins = 0

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
        for b in series_buckets(sk):
            if (xb := bucket_x(sk, b)) is not None:
                all_plotted_x.append(float(xb))
    for _lbl, mx, _, _ in max_series.values():
        all_plotted_x.extend(float(x) for x in mx)

    min_plotted_x = min(all_plotted_x) if all_plotted_x else 0.0

    shrink_to = _parse_x_axis_shrink(x_axis_break, min_plotted_x)

    def xplt(x: float) -> float:
        if shrink_to is None:
            return x
        return _x_data_to_plot_shrink(x, min_plotted_x, shrink_to)

    if use_bins:
        pad = max(0.08 * max(x_max_data + 1.0, 1.0), 0.42)
        x_hi_data = x_max_data + pad
    else:
        pad = max(x_max_data * 0.02, 1.0)
        x_hi_data = x_max_data + pad
    x_hi_plot = xplt(x_hi_data)

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_title(title)

    stem_colors: dict[tuple[str, frozenset[str]], str] = {}
    for sk in sub_keys:
        label = sub_key_to_display[sk]
        buckets_sl = series_buckets(sk)
        if not include_max_only:
            xs_stem: list[float] = []
            means: list[float] = []
            stds: list[float] = []
            for b in buckets_sl:
                x = bucket_x(sk, b)
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

            xs_plot = [xplt(x) for x in xs_stem]
            stem_line = ax.errorbar(
                xs_plot,
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
        if sk in max_series and sk not in stem_colors and include_max_only:
            ln = ax.plot([], [], linestyle="solid")[0]
            stem_colors[sk] = ln.get_color()
            ln.remove()

    for sk in sub_keys:
        if sk not in max_series:
            continue
        max_label, mx, max_means, max_stds = max_series[sk]
        dotted_color = stem_colors[sk]
        mx_plot = [xplt(x) for x in mx]
        ax.errorbar(
            mx_plot,
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
            ax.set_xticklabels(
                [str(int(t)) if t == int(t) else str(t) for t in reg_ticks]
            )
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
            ax.set_xticklabels(
                [str(int(t)) if t == int(t) else str(t) for t in reg_data]
            )

        _draw_x_shrink_marks(ax, shrink_to, x_hi_plot)

    ax.set_xlabel(
        "Validation bin"
        if use_bins
        else "Validation length (upper bound)"
    )
    ax.set_ylabel("Accuracy (%)")
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
        "--num-bins",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Keep only rows whose model+learning-rate has exactly N distinct buckets; "
            "drop rows for configurations with any other bucket count."
        ),
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
        choices=("ends", "regular", "bins"),
        default="ends",
        help=(
            "ends: tick at each bin upper bound with <N> labels (default). "
            "regular: numeric axis with --x-tick-step. "
            "bins: ordinal bins (0,1,…); models with the same number of bins align even if "
            "numeric ranges differ slightly."
        ),
    )
    parser.add_argument(
        "--x-tick-step",
        type=int,
        default=10,
        help="Spacing for --x-ticks=regular.",
    )
    parser.add_argument(
        "--x-axis-break",
        type=str,
        default=None,
        metavar="POS",
        help=(
            "Compress [0, first bucket] onto plot [0, POS]: first datapoint is drawn at x=POS, "
            "then the axis continues with the same numeric spacing as the data. "
            "POS is a positive number (plot units, e.g. 5) or 'auto' (uses 5). "
            "// marks mark the join. Omit for an ordinary linear axis."
        ),
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
    if args.num_bins is not None and args.num_bins < 1:
        raise SystemExit("--num-bins must be >= 1.")

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
        x_axis_break=args.x_axis_break,
        num_bins=args.num_bins,
    )
    print(f"Wrote plot: {plot_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
