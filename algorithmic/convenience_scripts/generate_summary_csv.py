#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean

# Extract the model name (first whitespace-delimited token on the line).
_LINE_MODEL_RE = re.compile(r"^(\S+)")
# Extract every eval bucket — captures (range, accuracy) pairs in order.
# Bucket names are just the numeric range, e.g. "0-50", "51-100".
_LINE_BUCKET_RE = re.compile(
    r"eval_len([0-9]+-[0-9]+)_acc:\s*([0-9]*\.?[0-9]+)"
)
# Extract the learning rate.
_LINE_LR_RE = re.compile(r"\blr:\s*([0-9]*\.?[0-9]+(?:e-?[0-9]+)?)")

# Known SSM kernel identifiers.  Extend this set to add new kernels; they are
# pre-extracted from the model string before the tokeniser runs so that the
# [sa]+ layer-ordering rule can remain a simple (?:a|s)+ without any lookaheads.
KNOWN_KERNELS: frozenset[str] = frozenset({"s4", "s6", "mamba"})

# Kernel placeholders use \x01N\x01 (SOH byte as delimiter) so they cannot be
# split by any letter or digit pattern in the tokeniser regex.
_KERNEL_SLOT_RE = re.compile(r"\x01(\d+)\x01")

# Tokenizer for model strings.  Order matters: more specific / longer patterns
# must come before more general ones.
#
# Known kernels are pre-replaced with \x01N\x01 placeholders in tokenize_model
# before this regex runs, so [sa]+ can simply be (?:a|s)+ — it stops naturally
# at digits and the placeholder SOH byte acts as a hard boundary.
_MODEL_TOKEN_RE = re.compile(
    r"\x01\d+\x01"                              # kernel slot — restore in tokenize_model
    r"|nope"                                    # NoPE flag — must precede "pe"
    r"|noln"                                    # no-LayerNorm flag — must precede "ln"
    r"|hyb"                                     # hybrid architecture
    r"|ssm"                                     # SSM architecture
    r"|lm(?![a-z])"                             # LM/transformer architecture
    r"|mlp"                                     # MLP-layers descriptor
    r"|stp"                                     # step-count prefix (stp{N}k)
    r"|dr"                                      # dropout suffix (pure-alpha form)
    r"|pe"                                      # positional-encoding flag
    r"|ln"                                      # LayerNorm flag
    r"|lr"                                      # learning-rate suffix (pure-alpha)
    r"|(?:a|s)+"                                # layer-ordering: sa, as, sas, …
    r"|[0-9]+(?:\.[0-9]+)?(?:mlp|dr|lr|[lhdk])"  # num+known-suffix: 1l, 4d, 0dr, 30k, 0.001lr
    r"|[a-z]+[0-9]+(?:\.[0-9]+)?"               # alpha+num fallback
    r"|[0-9]+(?:\.[0-9]+)?"                     # standalone number
    r"|[a-z]+"                                   # remaining alpha fallback
)

# Columns derived by parsing the model specification string.
MODEL_SPEC_COLUMNS = [
    "arch",          # lm | hyb | ssm
    "layers",        # number of layers (l)
    "heads",         # number of attention heads (h)
    "d_model",       # embedding dimension (d)
    "dropout",       # dropout rate (dr)
    "mlp_size",      # MLP layer size multiplier (mlp)
    "kernel",        # SSM kernel type (s4, s6, …); "-" for lm
    "pe",            # positional encoding: True / False / -
    "ln",            # layer norm: True / False / -
    "train_steps_k", # training steps in thousands (stp…k)
    "layer_order",   # hybrid layer ordering, e.g. sa, sas, as; "-" for lm/ssm
]

CSV_COLUMNS = [
    "task",
    # "source_file",
    # "source_line",
    "model",
    "learning_rate",
    "bucket",
    "accuracy",
] + MODEL_SPEC_COLUMNS


def parse_model_spec(model: str) -> dict[str, str]:
    """Parse a model specification string into structured feature columns.

    Strategy: tokenise with *_MODEL_TOKEN_RE* (which handles all ordering
    ambiguities, including ``sas4`` → ``sa`` + ``s4``), then assign each
    token to the appropriate column.  Pure-alpha flags are detected first;
    letter+number pairs are assigned to numeric columns afterwards.
    Any unrecognised tokens are silently discarded.
    """
    # Guard against null-byte pollution that can appear in legacy CSV rows.
    model_clean = model.strip().lstrip("\x00")
    tokens = tokenize_model(model_clean)

    pure_alpha: set[str] = set()
    features: dict[str, str] = {}  # alpha-key → numeric-value (last wins)
    for tok in tokens:
        feat = feature_from_token(tok)
        if feat is not None:
            key, val = feat
            features[key] = val
        else:
            pure_alpha.add(tok)

    # ── Architecture ──────────────────────────────────────────────────────────
    if "hyb" in pure_alpha:
        arch = "hyb"
    elif "ssm" in pure_alpha:
        arch = "ssm"
    else:
        arch = "lm"

    # ── SSM kernel ───────────────────────────────────────────────────────────
    # Kernels are matched as whole tokens (KNOWN_KERNELS), so we just look for
    # the first kernel token in the stream; default to "s4" if absent.
    if arch in ("hyb", "ssm"):
        kernel = next((tok for tok in tokens if tok in KNOWN_KERNELS), "s4")
    else:
        kernel = "-"

    # ── Positional encoding ──────────────────────────────────────────────────
    if "nope" in pure_alpha:
        pe = "False"
    elif "pe" in pure_alpha:
        pe = "True"
    else:
        pe = "-"

    # ── Layer norm ───────────────────────────────────────────────────────────
    if "noln" in pure_alpha:
        ln = "False"
    elif "ln" in pure_alpha:
        ln = "True"
    else:
        ln = "-"

    # ── Training steps (thousands) ───────────────────────────────────────────
    # Written as stp{N}k; tokeniser yields pure-alpha "stp" + feature ("k", N)
    if "stp" in pure_alpha and "k" in features:
        train_steps_k = features["k"]
    else:
        train_steps_k = "-"

    # ── Hybrid layer ordering ([sa]+) ─────────────────────────────────────────
    # We take the first token matching [sa]+ in the original token stream.
    layer_order = "-"
    if arch == "hyb":
        for tok in tokens:
            if re.fullmatch(r"[sa]+", tok):
                layer_order = tok
                break

    return {
        "arch": arch,
        "layers": features.get("l", "-"),
        "heads": features.get("h", "-"),
        "d_model": features.get("d", "-"),
        "dropout": features.get("dr", "-"),
        "mlp_size": features.get("mlp", "-"),
        "kernel": kernel,
        "pe": pe,
        "ln": ln,
        "train_steps_k": train_steps_k,
        "layer_order": layer_order,
    }


def parse_summary_line(line: str, task: str) -> list[dict[str, str | int | float]]:
    line = line.strip()
    m_model = _LINE_MODEL_RE.match(line)
    m_lr = _LINE_LR_RE.search(line)
    buckets = _LINE_BUCKET_RE.findall(line)  # list of (range_str, acc_str)
    if not m_model or not m_lr or not buckets:
        return []

    model = m_model.group(1)
    lr = float(m_lr.group(1))
    spec = parse_model_spec(model)

    rows: list[dict[str, str | int | float]] = []
    for range_str, acc_str in buckets:
        row: dict[str, str | int | float] = {
            "task": task,
            "model": model,
            "learning_rate": lr,
            "bucket": range_str,   # e.g. "0-50", "51-100"
            "accuracy": float(acc_str),
        }
        row.update(spec)  # type: ignore[arg-type]
        rows.append(row)
    return rows


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
        bucket = _normalize_bucket(row["bucket"])
        key = (
            bucket,
            row["model"],
            float(row["learning_rate"]),
            row["task"],
        )
        if key in seen_keys:
            continue
        seen_keys.add(key)
        # Build a base row with defaults for any columns absent in older CSV
        # files, then overwrite spec columns by re-parsing the model string so
        # that spec columns are always up-to-date even when loading a legacy CSV.
        base: dict[str, str | int | float] = {col: row.get(col, "-") for col in CSV_COLUMNS}
        base["bucket"] = bucket
        base.update(parse_model_spec(row["model"]))  # type: ignore[arg-type]
        merged_rows.append(base)

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


def feature_from_token(token: str) -> tuple[str, str] | None:
    """Return ``(key, value)`` if *token* encodes a named numeric feature, else ``None``.

    Both ``num+alpha`` tokens (e.g. ``"1l"`` → ``("l","1")``) and ``alpha+num``
    tokens (e.g. ``"s4"`` → ``("s","4")``) are normalised to ``(alpha_key, num_val)``.
    Pure-alpha tokens (``"nope"``, ``"hyb"``, …) return ``None``.
    """
    m = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)([a-z]+)", token)
    if m:
        return (m.group(2), m.group(1))
    m = re.fullmatch(r"([a-z]+)([0-9]+(?:\.[0-9]+)?)", token)
    if m:
        return (m.group(1), m.group(2))
    return None


def tokenize_model(model: str) -> list[str]:
    """Split a model string into its semantic tokens.

    Known kernels (``KNOWN_KERNELS``) are pre-replaced with ``\\x01N\\x01``
    placeholders so the ``(?:a|s)+`` layer-ordering rule never sees them and
    can remain a simple greedy match without lookaheads.
    """
    s = model.strip().lstrip("\x00").lower()
    # Pre-extract kernels longest-first to avoid partial matches.
    slots: list[str] = []
    for kernel in sorted(KNOWN_KERNELS, key=len, reverse=True):
        i = 0
        while (pos := s.find(kernel, i)) != -1:
            placeholder = f"\x01{len(slots)}\x01"
            slots.append(kernel)
            s = s[:pos] + placeholder + s[pos + len(kernel):]
            i = pos + len(placeholder)
    raw = _MODEL_TOKEN_RE.findall(s)
    # Restore placeholders to their original kernel strings.
    return [
        slots[int(m.group(1))] if (m := _KERNEL_SLOT_RE.fullmatch(tok)) else tok
        for tok in raw
    ]


def _bucket_sort_key(bucket: str) -> int:
    """Numeric sort key for bucket range strings like ``'0-50'``, ``'101-150'``."""
    return int(bucket.split("-")[0]) if "-" in bucket else 0


def _normalize_bucket(bucket: str) -> str:
    """Strip legacy ``eval_len`` prefix from bucket names."""
    return bucket[len("eval_len"):] if bucket.startswith("eval_len") else bucket


def split_spec_pattern_tokens(pattern: str) -> list[str]:
    """Split a CSV-row spec pattern into trimmed lowercase tokens (comma = AND)."""
    return [p.strip().lower() for p in pattern.split(",") if p.strip()]


def _token_matches_row(row: dict[str, str], token: str) -> bool:
    """Return whether a single spec token matches structured *row* columns."""
    t = token.strip().lower()
    if not t:
        return True

    if t in KNOWN_KERNELS:
        return row.get("kernel", "-") == t

    if t in ("lm", "hyb", "ssm"):
        return row.get("arch", "-") == t

    if t == "nope":
        return row.get("pe", "-") == "False"
    if t == "noln":
        return row.get("ln", "-") == "False"
    if t == "pe":
        return row.get("pe", "-") == "True"
    if t == "ln":
        return row.get("ln", "-") == "True"

    if re.fullmatch(r"[sa]+", t):
        return row.get("layer_order", "-") == t

    if re.fullmatch(r"\d+-\d+", t):
        return row.get("bucket", "") == t

    if t == "stp":
        return row.get("train_steps_k", "-") != "-"

    if t == "mlp":
        return row.get("mlp_size", "-") != "-"

    if t == "dr":
        return row.get("dropout", "-") != "-"

    feat = feature_from_token(t)
    if feat is not None:
        key, val = feat
        if key == "s":
            return row.get("kernel", "-") == f"s{val}"
        if key == "lr":
            try:
                return abs(float(row["learning_rate"]) - float(val)) < 1e-12
            except (KeyError, ValueError, TypeError):
                return False
        col_map = {
            "l": "layers",
            "h": "heads",
            "d": "d_model",
            "dr": "dropout",
            "mlp": "mlp_size",
            "k": "train_steps_k",
        }
        col = col_map.get(key)
        if not col:
            return False
        got = row.get(col, "-")
        if got == val:
            return True
        if key == "dr":
            try:
                return abs(float(got) - float(val)) < 1e-9
            except ValueError:
                return False
        return False

    return False


def row_matches_spec_pattern(row: dict[str, str], pattern: str) -> bool:
    """Return whether *row* satisfies the spec *pattern*.

    * Comma-separated clauses are **AND**-ed.  Each clause uses short notation
      against structured CSV columns (``arch``, ``layers``, ``kernel``, …)::

          1l,lm,nope   →  layers==1 AND arch==lm AND pe==False

    * If there is **no** comma, try a single structured token first, then fall
      back to :func:`matches_pattern` on the raw ``model`` string (legacy).
    """
    pattern_stripped = pattern.strip()
    if not pattern_stripped:
        return False
    low = pattern_stripped.lower()
    model = row.get("model", "")

    if "," in pattern_stripped:
        tokens = split_spec_pattern_tokens(pattern_stripped)
        if not tokens:
            return False
        return all(_token_matches_row(row, tok) for tok in tokens)

    if _token_matches_row(row, low):
        return True
    return matches_pattern(model, low)


def _parse_model_tokens(model: str) -> tuple[frozenset[str], frozenset[tuple[str, str]]]:
    """Return ``(pure_alpha_flags, feature_pairs)`` for *model*."""
    pure_alpha: set[str] = set()
    features: set[tuple[str, str]] = set()
    for tok in tokenize_model(model):
        feat = feature_from_token(tok)
        if feat is not None:
            features.add(feat)
        else:
            pure_alpha.add(tok)
    return frozenset(pure_alpha), frozenset(features)


def extract_feature_tokens(value: str) -> list[tuple[str, str]]:
    """Extract ``(key, value)`` feature pairs from a model/pattern string."""
    return [f for tok in tokenize_model(value) if (f := feature_from_token(tok)) is not None]


def matches_pattern(model: str, pattern: str) -> bool:
    """Return ``True`` if *model* satisfies all constraints expressed in *pattern*.

    Matching is **token-level**, not substring-level:

    * Pure-alpha flag tokens (``nope``, ``pe``, ``noln``, ``ln``, ``hyb``, …) must
      appear verbatim in the model's token set, **or** must match the alpha key of
      a numeric feature (so ``"dr"`` matches ``"0dr"`` or ``"0.1dr"``).
    * Numeric feature tokens (``"1l"``, ``"s4"``, ``"2mlp"``, ``"stp30k"`` parsed
      as ``stp`` + ``30k``) must appear as exact feature pairs in the model.

    This avoids the ``"pe"`` ⊆ ``"nope"`` false-positive of the old substring
    approach while remaining order-insensitive across feature tokens.
    """
    pattern_l = pattern.lower().strip()
    if not pattern_l:
        return False

    model_alpha, model_features = _parse_model_tokens(model)
    pattern_alpha, pattern_features = _parse_model_tokens(pattern_l)

    # Every numeric feature in the pattern must be present in the model.
    for feat in pattern_features:
        if feat not in model_features:
            return False

    # Every pure-alpha flag in the pattern must appear as a flag token OR as the
    # alpha key of any model feature (e.g. pattern "dr" matches model token "0dr").
    model_feature_keys = frozenset(key for key, _ in model_features)
    for flag in pattern_alpha:
        if flag not in model_alpha and flag not in model_feature_keys:
            return False

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
    remove_patterns: list[str] | None = None,
) -> list[dict[str, str]]:
    """Filter CSV *rows* by task, optional remove patterns, then optional includes.

    * ``remove_patterns``: row is dropped if it matches **any** pattern (pre-filter).
    * ``include_patterns``: if non-empty, row is kept only if it matches **any**
      pattern **or** satisfies an ``--include`` exact model/lr rule.
    * Within a single pattern string, comma-separated tokens are **AND**-ed
      (see :func:`row_matches_spec_pattern`).
    """
    remove_patterns = [p for p in (remove_patterns or []) if p.strip()]

    if tasks:
        task_set = set(tasks)
        rows = [r for r in rows if r["task"] in task_set]

    if remove_patterns:
        rows = [
            r
            for r in rows
            if not any(row_matches_spec_pattern(r, p) for p in remove_patterns)
        ]

    include_patterns = [p for p in include_patterns if p.strip()]
    if not include_rules and not include_patterns:
        return rows

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

        matched_pattern = any(row_matches_spec_pattern(row, p) for p in include_patterns)
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

    all_buckets = sorted({k[3] for k in grouped}, key=_bucket_sort_key)

    by_task: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for task, model, lr in series_keys:
        by_task[task].append((model, lr))

    for task in sorted(by_task.keys()):
        print(f"\nTask: {task}")
        for model, lr in sorted(by_task[task], key=lambda x: (x[0], x[1])):
            parts: list[str] = []
            for bucket in all_buckets:
                vals = grouped.get((task, model, lr, bucket), [])
                acc = f"{mean(vals):.4f}" if vals else "nan"
                parts.append(f"len{bucket}={acc} (n={len(vals)})")
            print(f"- {model} | lr={lr:g} | " + " | ".join(parts))


def _select_max_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """
    Keep only rows belonging to max-model datapoints per task.
    A datapoint is (model, learning_rate). We select datapoints that:
      1) achieve the per-bin maximum mean for at least one bin, and
      2) are not dominated in all bins by another datapoint.
    """
    rows_by_task: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        rows_by_task[row["task"]].append(row)

    selected_keys: set[tuple[str, str, float]] = set()
    for task, task_rows in rows_by_task.items():
        datapoint_bucket_vals: dict[tuple[str, float, str], list[float]] = defaultdict(list)
        datapoints: set[tuple[str, float]] = set()
        for row in task_rows:
            model = row["model"]
            lr = float(row["learning_rate"])
            bucket = row["bucket"]
            datapoints.add((model, lr))
            datapoint_bucket_vals[(model, lr, bucket)].append(float(row["accuracy"]))

        if not datapoints:
            continue

        task_buckets = sorted(
            {b for _, _, b in datapoint_bucket_vals}, key=_bucket_sort_key
        )

        point_vec: dict[tuple[str, float], list[float]] = {}
        for dp in datapoints:
            vec: list[float] = []
            for bucket in task_buckets:
                vals = datapoint_bucket_vals.get((dp[0], dp[1], bucket), [])
                vec.append(mean(vals) if vals else float("-inf"))
            point_vec[dp] = vec

        winner_points: set[tuple[str, float]] = set()
        for bucket in task_buckets:
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

        # Remove winners dominated by another datapoint in all bins.
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

        for model, lr in pruned_winners:
            selected_keys.add((task, model, lr))

    return [
        row
        for row in rows
        if (row["task"], row["model"], float(row["learning_rate"])) in selected_keys
    ]


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
    parser.add_argument("--create", action="store_true")
    
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
        "--remove-pattern",
        action="append",
        default=[],
        help=(
            "Pre-filter: drop rows that match any pattern.  Same short notation as "
            "--include-pattern (comma = AND within one pattern).  Repeat flag for "
            "multiple remove patterns (row removed if it matches any)."
        ),
    )
    parser.add_argument(
        "--include-pattern",
        action="append",
        default=[],
        help=(
            "Keep rows that match any pattern (OR across repeated flags).  Each pattern "
            "uses comma-separated short tokens AND-ed against CSV columns, e.g. "
            "'1l,lm,nope' (layers=1, arch=lm, NoPE).  No comma: structured token "
            "then legacy model-string matching."
        ),
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="Print matching models and per-bin mean results from the CSV.",
    )
    parser.add_argument(
        "--include-max-only",
        action="store_true",
        help=(
            "When used with --list-models, keep only models that are max contributors "
            "per task across bins (with dominated models removed)."
        ),
    )
    args = parser.parse_args()

    if args.create:
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
        remove_patterns=args.remove_pattern,
    )
    if args.include_max_only:
        filtered_rows = _select_max_rows(filtered_rows)
    print_models(filtered_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
