#!/usr/bin/env python3
"""Shared plotting helpers used by convenience scripts."""

from __future__ import annotations

import math
import re
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from statistics import mean

from generate_summary_csv import MODEL_SPEC_COLUMNS, parse_model_spec


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


def _ordinal_bin_index(bucket: str) -> int | None:
    s = bucket.strip()
    if s.startswith("bin") and s[3:].isdigit():
        return int(s[3:])
    return None


def _bucket_sort_key_plot(bucket: str) -> tuple[int, str]:
    oi = _ordinal_bin_index(bucket)
    if oi is not None:
        return (oi, bucket)
    ub = _bucket_upper_bound(bucket)
    return (ub if ub is not None else -1, bucket)


def _bucket_plot_x(bucket: str) -> float | None:
    oi = _ordinal_bin_index(bucket)
    if oi is not None:
        return float(oi)
    u = _bucket_upper_bound(bucket)
    return float(u) if u is not None else None


BIN_SIGNATURE_SEP = "|"

_BIN_ACC_COL_RE = re.compile(r"^bin(\d+)_acc$")
_BIN_RANGE_COL_RE = re.compile(r"^bin(\d+)_range$")

# Run-level columns that describe the bin layout.  They are recomputed from the
# rows that survive filtering, so plot filters can rely on them being accurate.
RUN_BIN_COLUMNS = (
    "num_bins",
    "bins",
    "train_range",
    "train_len_min",
    "train_len_max",
    "max_eval_len",
    "bin_index",
)


def _normalize_bucket_name(bucket: str) -> str:
    b = str(bucket).strip()
    return b[len("eval_len"):] if b.startswith("eval_len") else b


def load_summary_dataframe(csv_path):
    """Read a summary CSV and return one row per (run, bin).

    Accepts the current one-row-per-run layout (``bin1_range``/``bin1_acc``, …)
    and the legacy one-row-per-bin layout (``bucket``/``accuracy``).  In both
    cases the result carries a ``run_id`` plus the ``RUN_BIN_COLUMNS`` describing
    each run's bin layout.
    """
    try:
        import pandas as pd
    except ModuleNotFoundError as e:  # pragma: no cover - dependency guard
        raise SystemExit("Reading summary CSVs requires pandas.") from e

    df = pd.read_csv(csv_path)
    if df.empty:
        raise SystemExit(f"No CSV rows found at {csv_path}.")
    return summary_to_long_bins(df)


def summary_to_long_bins(df):
    """Explode per-run bin columns into one row per (run, bin)."""
    import pandas as pd

    df = df.copy()
    bin_indices = sorted(
        int(m.group(1)) for c in df.columns if (m := _BIN_ACC_COL_RE.match(str(c)))
    )

    if not bin_indices:
        if "bucket" not in df.columns or "accuracy" not in df.columns:
            raise SystemExit(
                "Unrecognised summary CSV: expected bin1_range/bin1_acc columns "
                "(one row per run) or bucket/accuracy columns (legacy layout)."
            )
        # Legacy layout: runs were not distinguishable, so all rows of one
        # (task, model, learning_rate) triple are treated as a single run.
        # Grouping on string casts keeps rows with missing model names.
        if "run_id" not in df.columns:
            keys = [
                df[c].astype(str).fillna("")
                for c in ("task", "model", "learning_rate")
            ]
            df["run_id"] = df.groupby(keys, sort=False, dropna=False).ngroup()
        return recompute_run_bin_metadata(df)

    if "run_id" not in df.columns:
        df["run_id"] = range(len(df))
    id_cols = [
        c
        for c in df.columns
        if not (_BIN_ACC_COL_RE.match(str(c)) or _BIN_RANGE_COL_RE.match(str(c)))
        and c not in RUN_BIN_COLUMNS
    ]

    frames = []
    for i in bin_indices:
        range_col, acc_col = f"bin{i}_range", f"bin{i}_acc"
        cols = [*id_cols, acc_col] + ([range_col] if range_col in df.columns else [])
        sub = df[cols].copy()
        sub = sub[sub[acc_col].notna()]
        if sub.empty:
            continue
        if range_col in sub.columns:
            sub = sub[sub[range_col].notna()]
            sub["bucket"] = sub[range_col].astype(str)
            sub = sub.drop(columns=[range_col])
        else:
            sub["bucket"] = f"bin{i}"
        sub["accuracy"] = sub[acc_col].astype(float)
        sub = sub.drop(columns=[acc_col])
        frames.append(sub)

    if not frames:
        raise SystemExit("Summary CSV has bin columns but no accuracies.")
    return recompute_run_bin_metadata(pd.concat(frames, ignore_index=True))


def recompute_run_bin_metadata(df):
    """(Re-)derive ``RUN_BIN_COLUMNS`` from the (run, bin) rows present in *df*."""
    df = df.copy()
    if "run_id" not in df.columns:
        raise SystemExit("recompute_run_bin_metadata: missing run_id column.")
    df["bucket"] = df["bucket"].astype(str).map(_normalize_bucket_name)
    df["learning_rate"] = df["learning_rate"].astype(float)
    df["accuracy"] = df["accuracy"].astype(float)

    ordered: dict[object, list[str]] = {}
    for run_id, grp in df.groupby("run_id", sort=False):
        ordered[run_id] = sorted(set(grp["bucket"]), key=_bucket_sort_key_plot)

    df["bins"] = df["run_id"].map(lambda r: BIN_SIGNATURE_SEP.join(ordered[r]))
    df["num_bins"] = df["run_id"].map(lambda r: len(ordered[r]))
    df["train_range"] = df["run_id"].map(lambda r: ordered[r][0] if ordered[r] else "")
    df["bin_index"] = [
        ordered[r].index(b) for r, b in zip(df["run_id"], df["bucket"], strict=True)
    ]
    train_bounds = df["train_range"].map(_parse_bucket_bounds)
    df["train_len_min"] = train_bounds.map(lambda b: b[0] if b else None)
    df["train_len_max"] = train_bounds.map(lambda b: b[1] if b else None)
    df["max_eval_len"] = df["run_id"].map(
        lambda r: _bucket_upper_bound(ordered[r][-1]) if ordered[r] else None
    )
    return df


@dataclass(frozen=True, kw_only=True)
class BinFilter:
    """Run/bin selection for plotting.

    Run-level criteria (``bin_counts``, ``min_bins``, ``max_bins``,
    ``train_ranges``, ``exclude_train_ranges``, ``bin_signatures``) drop whole
    runs and are applied to each run's original bin layout.  Bin-level criteria
    (``first_bins``, ``max_bin_upper``) then trim bins off the surviving runs,
    after which the run metadata is recomputed.
    """

    bin_counts: frozenset[int] = frozenset()
    min_bins: int | None = None
    max_bins: int | None = None
    train_ranges: frozenset[str] = frozenset()
    exclude_train_ranges: frozenset[str] = frozenset()
    bin_signatures: frozenset[str] = frozenset()
    first_bins: int | None = None
    max_bin_upper: int | None = None

    def is_empty(self) -> bool:
        return not any(
            (
                self.bin_counts,
                self.min_bins is not None,
                self.max_bins is not None,
                self.train_ranges,
                self.exclude_train_ranges,
                self.bin_signatures,
                self.first_bins is not None,
                self.max_bin_upper is not None,
            )
        )

    def describe(self) -> str:
        parts: list[str] = []
        if self.bin_counts:
            parts.append(f"num_bins in {sorted(self.bin_counts)}")
        if self.min_bins is not None:
            parts.append(f"num_bins >= {self.min_bins}")
        if self.max_bins is not None:
            parts.append(f"num_bins <= {self.max_bins}")
        if self.train_ranges:
            parts.append(f"train_range in {sorted(self.train_ranges)}")
        if self.exclude_train_ranges:
            parts.append(f"train_range not in {sorted(self.exclude_train_ranges)}")
        if self.bin_signatures:
            parts.append(f"bins in {sorted(self.bin_signatures)}")
        if self.first_bins is not None:
            parts.append(f"first {self.first_bins} bins")
        if self.max_bin_upper is not None:
            parts.append(f"bin upper bound <= {self.max_bin_upper}")
        return "; ".join(parts) if parts else "no bin filters"

    def apply(self, df):
        if self.is_empty():
            return df
        if "num_bins" not in df.columns or "bin_index" not in df.columns:
            df = recompute_run_bin_metadata(df)

        if self.bin_counts:
            df = df[df["num_bins"].astype(int).isin(self.bin_counts)]
        if self.min_bins is not None:
            df = df[df["num_bins"].astype(int) >= self.min_bins]
        if self.max_bins is not None:
            df = df[df["num_bins"].astype(int) <= self.max_bins]
        if self.train_ranges:
            df = df[df["train_range"].astype(str).isin(self.train_ranges)]
        if self.exclude_train_ranges:
            df = df[~df["train_range"].astype(str).isin(self.exclude_train_ranges)]
        if self.bin_signatures:
            df = df[df["bins"].astype(str).isin(self.bin_signatures)]

        trimmed = False
        if self.first_bins is not None:
            df = df[df["bin_index"].astype(int) < self.first_bins]
            trimmed = True
        if self.max_bin_upper is not None:
            uppers = df["bucket"].astype(str).map(_bucket_upper_bound)
            df = df[uppers.notna() & (uppers <= self.max_bin_upper)]
            trimmed = True

        if df.empty:
            return df
        return recompute_run_bin_metadata(df) if trimmed else df


def normalize_range_tokens(values: list[str], flag: str) -> frozenset[str]:
    """Parse repeated/comma-separated ``lo-hi`` CLI values into a set of ranges."""
    out: set[str] = set()
    for raw in values:
        for tok in str(raw).split(","):
            tok = _normalize_bucket_name(tok)
            if not tok:
                continue
            if _parse_bucket_bounds(tok) is None:
                raise SystemExit(f"{flag}: expected ranges like 0-49, got {tok!r}.")
            out.add(tok)
    return frozenset(out)


def normalize_int_tokens(values: list[str], flag: str, *, minimum: int = 1) -> frozenset[int]:
    """Parse repeated/comma-separated integer CLI values."""
    out: set[int] = set()
    for raw in values:
        for tok in str(raw).split(","):
            tok = tok.strip()
            if not tok:
                continue
            try:
                n = int(tok)
            except ValueError:
                raise SystemExit(f"{flag}: expected integers, got {tok!r}.") from None
            if n < minimum:
                raise SystemExit(f"{flag}: values must be >= {minimum}.")
            out.add(n)
    return frozenset(out)


def normalize_bin_signatures(values: list[str], flag: str) -> frozenset[str]:
    """Parse ``lo-hi|lo-hi|...`` layout specs into canonical signatures."""
    out: set[str] = set()
    for raw in values:
        ranges = normalize_range_tokens([str(raw).replace(BIN_SIGNATURE_SEP, ",")], flag)
        if not ranges:
            continue
        out.add(
            BIN_SIGNATURE_SEP.join(
                sorted(ranges, key=lambda r: _parse_bucket_bounds(r) or (0, 0))
            )
        )
    return frozenset(out)


def _ordinal_bin_tick_labels(
    sub_keys: list[tuple[str, frozenset[str]]],
    series_buckets: Callable[[tuple[str, frozenset[str]]], list[str]],
) -> list[str]:
    if not sub_keys:
        return []
    max_k = max(len(series_buckets(sk)) for sk in sub_keys)
    return [f"bin{i + 1}" for i in range(max_k)]


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
    cands = [b for b in _buckets_for_datapoint(dp, dcv) if _bucket_upper_bound(b) == end]
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


_NUM_SUFFIX_RE = re.compile(r"^(.+)(l|h|d|dr|mlp)$")


def _spec_for_legend(row: dict[str, str]) -> dict[str, str]:
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
            return "/".join(
                sorted(bodies, key=lambda s: float(s) if _is_float_str(s) else s)
            ) + "lr"
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


def _legend_label_merged(rows: list[dict[str, str]]) -> str:
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
    if not rows:
        return ""
    dps = {(r["model"], float(r["learning_rate"])) for r in rows}
    if len(dps) == 1:
        r0 = next(r for r in rows if (r["model"], float(r["learning_rate"])) in dps)
        return compact_spec_label_from_row(r0)
    return _legend_label_merged(rows)


def _dedupe_legend_labels(ids: list[str], id_to_label: dict[str, str]) -> dict[str, str]:
    seen: dict[str, int] = {}
    out: dict[str, str] = {}
    for sid in ids:
        base = id_to_label[sid]
        n = seen.get(base, 0)
        seen[base] = n + 1
        out[sid] = base if n == 0 else f"{base} ({n + 1})"
    return out


def _row_bin_signature(row: dict) -> frozenset[str] | None:
    """Bin layout recorded on the row itself (``bins`` column), if available."""
    raw = row.get("bins")
    if raw is None:
        return None
    s = str(raw).strip()
    if not s or s == "-" or s.lower() == "nan":
        return None
    parts = {p.strip() for p in s.split(BIN_SIGNATURE_SEP) if p.strip()}
    return frozenset(parts) or None


def _datapoint_bucket_set(
    rows_for_sid: list[dict[str, str]],
) -> dict[tuple[str, float], frozenset[str]]:
    acc: dict[tuple[str, float], set[str]] = defaultdict(set)
    for row in rows_for_sid:
        dp = (row["model"], float(row["learning_rate"]))
        acc[dp].add(row["bucket"])
    return {dp: frozenset(bs) for dp, bs in acc.items()}


def _run_group_key(
    row: dict, dp_to_bins: dict[tuple[str, float], frozenset[str]]
) -> tuple[str, float, frozenset[str]]:
    """Key that keeps runs with different bin layouts apart."""
    dp = (str(row["model"]), float(row["learning_rate"]))
    sig = _row_bin_signature(row) or dp_to_bins.get(dp, frozenset())
    return (dp[0], dp[1], sig)


def _remap_rows_to_ordinal_bins(rows: list[dict]) -> list[dict]:
    """Replace interval bucket names with bin0, bin1, ... per run layout."""
    dp_to_bins = _datapoint_bucket_set(rows)
    group_buckets: dict[tuple[str, float, frozenset[str]], set[str]] = defaultdict(set)
    for row in rows:
        group_buckets[_run_group_key(row, dp_to_bins)].add(str(row["bucket"]))
    group_sorted = {
        key: sorted(bs, key=_bucket_sort_key_plot) for key, bs in group_buckets.items()
    }
    out: list[dict] = []
    for row in rows:
        idx = group_sorted[_run_group_key(row, dp_to_bins)].index(str(row["bucket"]))
        new_row = dict(row)
        new_row["bucket"] = f"bin{idx}"
        out.append(new_row)
    return out


def _value_at_bin_index_for_dp(
    dp: tuple[str, float], bin_idx: int, dcv: dict[tuple[str, float, str], list[float]]
) -> float:
    buckets = _buckets_for_datapoint(dp, dcv)
    if bin_idx >= len(buckets):
        return float("-inf")
    m = _mean_for_dp_bucket(dp, buckets[bin_idx], dcv)
    return m if m is not None else float("-inf")


def _pool_raw_at_bin_index_for_dps(
    dps: set[tuple[str, float]],
    bin_idx: int,
    dcv: dict[tuple[str, float, str], list[float]],
) -> list[float]:
    chunk: list[float] = []
    for dp in dps:
        buckets = _buckets_for_datapoint(dp, dcv)
        if bin_idx < len(buckets):
            chunk.extend(dcv.get((dp[0], dp[1], buckets[bin_idx]), []))
    return chunk


def select_max_winners_by_bin_index(
    datapoints: set[tuple[str, float]],
    dcv: dict[tuple[str, float, str], list[float]],
    *,
    all_bin_indices: list[int] | None = None,
) -> tuple[set[tuple[str, float]], list[int]]:
    if all_bin_indices is not None:
        all_bins = sorted(set(all_bin_indices))
    else:
        max_k = 0
        for dp in datapoints:
            max_k = max(max_k, len(_buckets_for_datapoint(dp, dcv)))
        all_bins = list(range(max_k))
    if not all_bins:
        return set(), []

    winner_dps: set[tuple[str, float]] = set()
    for k in all_bins:
        scored: list[tuple[tuple[str, float], float]] = []
        for dp in datapoints:
            buckets = _buckets_for_datapoint(dp, dcv)
            if k >= len(buckets):
                continue
            m = _mean_for_dp_bucket(dp, buckets[k], dcv)
            if m is not None:
                scored.append((dp, m))
        if not scored:
            continue
        best = max(t[1] for t in scored)
        for dp, m in scored:
            if m == best:
                winner_dps.add(dp)

    if not winner_dps:
        return set(), all_bins

    point_vec: dict[tuple[str, float], list[float]] = {}
    for dp in winner_dps:
        point_vec[dp] = [_value_at_bin_index_for_dp(dp, k, dcv) for k in all_bins]

    pruned: set[tuple[str, float]] = set()
    for p in winner_dps:
        pv = point_vec[p]
        dominated = False
        for q in winner_dps:
            if q == p:
                continue
            qv = point_vec[q]
            if all(qv[i] >= pv[i] for i in range(len(all_bins))) and any(
                qv[i] > pv[i] for i in range(len(all_bins))
            ):
                dominated = True
                break
        if not dominated:
            pruned.add(p)

    return pruned, all_bins


def max_line_xy_by_bin_index(
    pruned: set[tuple[str, float]],
    all_bin_indices: list[int],
    dcv: dict[tuple[str, float, str], list[float]],
    *,
    fallback_dps: set[tuple[str, float]] | None = None,
) -> tuple[list[float], list[float], list[float]]:
    fallback = fallback_dps if fallback_dps is not None else set()
    xs: list[float] = []
    means: list[float] = []
    stds: list[float] = []
    for k in all_bin_indices:
        chunk = _pool_raw_at_bin_index_for_dps(pruned, k, dcv)
        if not chunk and fallback:
            chunk = _pool_raw_at_bin_index_for_dps(fallback, k, dcv)
        if chunk:
            xs.append(float(k))
            means.append(mean(chunk) * 100.0)
            stds.append(_sample_std(chunk) * 100.0)
    return xs, means, stds


def _split_rows_by_bin_signature(
    rows_for_sid: list[dict[str, str]],
) -> dict[frozenset[str], list[dict[str, str]]]:
    """Split a series into sub-series that share the same bin layout.

    The per-run ``bins`` column is authoritative; without it (legacy CSVs) the
    layout is approximated by the union of a datapoint's buckets, which merges
    runs of the same model trained on different length ranges.
    """
    dp_to_bins = _datapoint_bucket_set(rows_for_sid)
    by_sig: dict[frozenset[str], list[dict[str, str]]] = defaultdict(list)
    for row in rows_for_sid:
        _model, _lr, sig = _run_group_key(row, dp_to_bins)
        by_sig[sig].append(row)
    return dict(by_sig)


def _signature_x_ends(signature: frozenset[str]) -> list[int]:
    ends: list[int] = []
    for b in signature:
        u = _bucket_upper_bound(b)
        if u is not None:
            ends.append(u)
    return sorted(set(ends))


def signature_layout_label(signature: frozenset[str]) -> str:
    """Readable description of a bin layout, e.g. ``train 0-24, 6 bins``."""
    ordered = sorted(signature, key=_bucket_sort_key_plot)
    if not ordered:
        return ""
    return f"train {ordered[0]}, {len(ordered)} bins"


def _signature_label(signature: frozenset[str]) -> str:
    ordered = sorted(signature, key=_bucket_sort_key_plot)
    parts: list[str] = []
    for b in ordered:
        u = _bucket_upper_bound(b)
        parts.append(str(u) if u is not None else b)
    return ",".join(parts)


def _parse_x_axis_shrink(raw: str | None, x_min_data: float) -> float | None:
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
    from matplotlib.transforms import blended_transform_factory

    if br_plot <= 0:
        return
    trans = blended_transform_factory(ax.transData, ax.transAxes)
    d = 0.015
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
