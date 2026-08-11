#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

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
KNOWN_KERNELS: frozenset[str] = frozenset({"s4", "s6", "mamba", "mamba2", "mamba3", "gdn"})

# Kernel placeholders use \x01N\x01 (SOH byte as delimiter) so they cannot be
# split by any letter or digit pattern in the tokeniser regex.
_KERNEL_SLOT_RE = re.compile(r"\x01(\d+)\x01")

# ``s4`` / ``s6`` appear as *substrings* of ``[as]+`` + ``<n>l`` (e.g. ``aaas4l``:
# last ``s`` of the motif plus the first digit of ``4l``).  Those must not be
# pre-extracted as SSM kernels (real modular / legacy names never place a
# kernel literal between ``hyb`` and ``<n>l``).
_S4_S6_FALSE_KERNEL_RE = re.compile(r"\d*l")


def _sdigit_kernel_is_layer_count_suffix(s: str, pos: int, kernel: str) -> bool:
    if kernel not in frozenset({"s4", "s6"}):
        return False
    if pos == 0 or s[pos - 1] not in "as":
        return False
    return _S4_S6_FALSE_KERNEL_RE.match(s[pos + len(kernel) :]) is not None


DEAFAULT_FEATURE_ORDER = ["l", "h", "d", "dr"]  # Some logs omit features; use position fallback.

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
    r"|none"                                    # no Nevative Eigenvalues for olmo
    r"|olmo"                                    # library package
    r"|hyb"                                     # hybrid architecture
    r"|ssm"                                     # SSM architecture
    r"|lm(?![a-z])"                             # LM/transformer architecture
    r"|mlp"                                     # MLP-layers descriptor
    r"|stp"                                     # step-count prefix (stp{N}k)
    r"|dr"                                      # dropout suffix (pure-alpha form)
    r"|pe"                                      # positional-encoding flag
    r"|ln"                                      # LayerNorm flag
    r"|ne"                                      # Negative Eigenvalues for olmo
    r"|lr"                                      # learning-rate suffix (pure-alpha)
    r"|(?:a|s)+"                                # layer-ordering: sa, as, sas, …
    r"|[0-9]+(?:\.[0-9]+)?(?:mlp|dr|lr|[lhdk])"  # num+known-suffix: 1l, 4d, 0dr, 30k, 0.001lr
    r"|[a-z]+[0-9]+(?:\.[0-9]+)?"               # alpha+num fallback
    r"|[0-9]+(?:\.[0-9]+)?reg"                   # e.g. 0.0reg — prefix before l/h/d sweep
    r"|[0-9]+(?:\.[0-9]+)?"                     # standalone number
    r"|[a-z]+"                                   # remaining alpha fallback
)

# Columns derived by parsing the model specification string.
MODEL_SPEC_COLUMNS = [
    "arch",          # lm | hyb | ssm | olmo
    "layers",        # number of layers (l)
    "heads",         # number of attention heads (h)
    "d_model",       # embedding dimension (d)
    "dropout",       # dropout rate (dr)
    "mlp_size",      # MLP layer size multiplier (mlp)
    "kernel",        # SSM kernel type (s4, s6, …); "-" for lm
    "pe",            # positional encoding: True / False / -
    "ln",            # layer norm: True / False / -
    "ne",            # negative eigenvalues: True / False / -
    "train_steps_k", # training steps in thousands (stp…k)
    "layer_order",   # hybrid layer ordering, e.g. sa, sas, as; "-" for lm/ssm
]

# Task-specific dataset parameters. Filled with defaults for legacy filenames.
TASK_PARAM_COLUMNS = [
    "mkar_key_len",
    "mkar_vocab_size",
    "mqar_query_fraction_lower",
    "mqar_query_fraction_upper",
    "mqar_monoid",
    "mqar_monoid_n",
    "mqar_key_size",
    "selective_copy_marker_vocab_size",
    "selective_copy_misc_vocab_size",
]

CSV_COLUMNS = [
    "task",
    # "source_file",
    # "source_line",
    "model",
    "learning_rate",
    "bucket",
    "accuracy",
] + MODEL_SPEC_COLUMNS + TASK_PARAM_COLUMNS

_FLOAT_TOKEN_RE = r"[0-9]+(?:\.[0-9]+)?"
_MKAR_KEY_LEN_RE = re.compile(r"key[_-]?len(?P<v>[0-9]+)")
_MKAR_VOCAB_RE = re.compile(r"(?:v[_-]?size|vocab[_-]?size)(?P<v>[0-9]+)")
_MQAR_FL_RE = re.compile(r"(?:fl|fraction[_-]?lower)(?P<v>" + _FLOAT_TOKEN_RE + r")")
_MQAR_FU_RE = re.compile(r"(?:fu|fraction[_-]?upper)(?P<v>" + _FLOAT_TOKEN_RE + r")")
_MQAR_KEY_SIZE_RE = re.compile(r"(?:key[_-]?size|ks)(?P<v>[0-9]+)")
_MQAR_MONOID_N_RE = re.compile(r"(?:monoid[_-]?n|mn)(?P<v>[0-9]+)")
_MQAR_MONOID_RE = re.compile(r"(?:monoid|mt)(?P<v>parity|cyclic)")
_SEL_MARKER_RE = re.compile(
    r"(?:marker[_-]?vocab[_-]?size|marker[_-]?size|mv|v[_-]?size)(?P<v>[0-9]+)"
)
_SEL_MISC_RE = re.compile(r"(?:misc[_-]?vocab[_-]?size|misc[_-]?size|ms)(?P<v>[0-9]+)")

TASK_PARAM_DEFAULTS: dict[str, dict[str, str]] = {
    "mkar": {
        "mkar_key_len": "4",
        "mkar_vocab_size": "128",
    },
    "mqar": {
        "mqar_query_fraction_lower": "0.2",
        "mqar_query_fraction_upper": "0.2",
        "mqar_monoid": "parity",
        "mqar_monoid_n": "2",
        "mqar_key_size": "32",
    },
    "selective_copy": {
        "selective_copy_marker_vocab_size": "16",
        "selective_copy_misc_vocab_size": "16",
    },
}


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
    ith_feature = 0
    for tok in tokens:
        feat = feature_from_token(tok, ith_feature)
        if feat is not None:
            key, val = feat
            features[key] = val
            ith_feature += 1
        else:
            pure_alpha.add(tok)

    # ── Architecture ──────────────────────────────────────────────────────────
    arch = ""
    if "olmo" in pure_alpha:
        arch += "olmo"
    if "hyb" in pure_alpha:
        arch += "hyb"
    elif "lm" in pure_alpha:
        arch += "lm"
    elif "ssm" in pure_alpha or "gdn" in pure_alpha:
        arch += "ssm"
    # elif arch.startswith("olmo"):
    #     # ``olmo`` alone (no explicit ``lm`` / ``hyb`` / ``ssm``): OLMo SSM-backed stack → gdn.
    #     arch += "ssm"
    else:
        # Structure-only specs like ``2l1h64d`` omit an arch keyword; default to LM.
        arch += "lm"

    # ── SSM kernel ───────────────────────────────────────────────────────────
    # Kernels are matched as whole tokens (KNOWN_KERNELS), so we just look for
    # the first kernel token in the stream; default to "s4" if absent.
    if "hyb" in arch or "ssm" in arch:
        if "olmo" in arch:
            kernel = "gdn"
        else:
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

    # ── Negative eigenvalues ───────────────────────────────────────────────────────────
    if "none" in pure_alpha:
        ne = "False"
    elif "ne" in pure_alpha:
        ne = "True"
    else:
        ne = "-"


    # ── Training steps (thousands) ───────────────────────────────────────────
    # Written as stp{N}k; tokeniser yields pure-alpha "stp" + feature ("k", N)
    if "stp" in pure_alpha and "k" in features:
        train_steps_k = features["k"]
    else:
        train_steps_k = "-"

    # ── Hybrid layer ordering ([sa]+) ─────────────────────────────────────────
    # We take the first token matching [sa]+ in the original token stream.
    layer_order = "-"
    if "hyb" in arch:
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
        "ne": ne,
        "train_steps_k": train_steps_k,
        "layer_order": layer_order,
    }


def parse_task_params(task: str, summary_file: Path) -> dict[str, str]:
    """Task-parameter columns extracted from summary filename, with defaults."""
    params = {col: "-" for col in TASK_PARAM_COLUMNS}
    defaults = TASK_PARAM_DEFAULTS.get(task, {})
    params.update(defaults)

    stem = summary_file.stem.lower()

    if task == "mkar":
        if (m := _MKAR_KEY_LEN_RE.search(stem)):
            params["mkar_key_len"] = m.group("v")
        if (m := _MKAR_VOCAB_RE.search(stem)):
            params["mkar_vocab_size"] = m.group("v")
        return params

    if task == "mqar":
        if (m := _MQAR_FL_RE.search(stem)):
            params["mqar_query_fraction_lower"] = m.group("v")
        if (m := _MQAR_FU_RE.search(stem)):
            params["mqar_query_fraction_upper"] = m.group("v")
        if (m := _MQAR_KEY_SIZE_RE.search(stem)):
            params["mqar_key_size"] = m.group("v")
        if (m := _MQAR_MONOID_N_RE.search(stem)):
            params["mqar_monoid_n"] = m.group("v")
        if (m := _MQAR_MONOID_RE.search(stem)):
            params["mqar_monoid"] = m.group("v")
        return params

    if task == "selective_copy":
        if (m := _SEL_MARKER_RE.search(stem)):
            params["selective_copy_marker_vocab_size"] = m.group("v")
        if (m := _SEL_MISC_RE.search(stem)):
            params["selective_copy_misc_vocab_size"] = m.group("v")
        return params

    return params


def parse_summary_line(
    line: str,
    task: str,
    task_params: dict[str, str] | None = None,
) -> list[dict[str, str | int | float]]:
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
        row.update(task_params or {})
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


def build_or_update_csv(
    logs_root: Path,
    csv_path: Path,
    *,
    bucket_end_digits: frozenset[int] | None = None,
) -> list[dict[str, str]]:
    existing_rows = load_csv_rows(csv_path)

    # seen_keys = set()
    merged_rows: list[dict[str, str | int | float]] = []

    for row in existing_rows:
        bucket = _normalize_bucket(row["bucket"])
        key = (
            bucket,
            row["model"],
            float(row["learning_rate"]),
            row["task"],
        )
        # if key in seen_keys:
        #     continue
        # seen_keys.add(key)
        # Build a base row with defaults for any columns absent in older CSV
        # files, then overwrite spec columns by re-parsing the model string so
        # that spec columns are always up-to-date even when loading a legacy CSV.
        base: dict[str, str | int | float] = {col: row.get(col, "-") for col in CSV_COLUMNS}
        base["bucket"] = bucket
        task_defaults = TASK_PARAM_DEFAULTS.get(row.get("task", ""), {})
        for col, default_val in task_defaults.items():
            if str(base.get(col, "-")).strip() in {"", "-"}:
                base[col] = default_val
        base.update(parse_model_spec(row["model"]))  # type: ignore[arg-type]
        merged_rows.append(base)

    summary_files = sorted(logs_root.glob("**/summary*.txt"))
    for summary_file in summary_files:
        task = summary_file.parent.name
        task_params = parse_task_params(task, summary_file)
        with summary_file.open("r") as f:
            for raw_line in f:
                parsed_rows = parse_summary_line(
                    raw_line, task=task, task_params=task_params
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
                    # if key in seen_keys:
                    #     continue
                    # seen_keys.add(key)
                    merged_rows.append(row)

    if bucket_end_digits:
        merged_rows = [
            r for r in merged_rows if row_matches_bucket_end_digit(r, bucket_end_digits)
        ]

    write_csv_rows(csv_path, merged_rows)
    return load_csv_rows(csv_path)


def feature_from_token(token: str, ith_feature=None) -> tuple[str, str] | None:
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
    m = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)", token)
    # __________ feature attribute forgotton in model spec. Fixing that using ith_feature.
    if m:
        if ith_feature and ith_feature < len(DEAFAULT_FEATURE_ORDER):
            return (DEAFAULT_FEATURE_ORDER[ith_feature], m.group(1))
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
            if _sdigit_kernel_is_layer_count_suffix(s, pos, kernel):
                i = pos + 1
                continue
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


def _normalize_bucket(bucket: str) -> str:
    """Strip legacy ``eval_len`` prefix from bucket names."""
    return bucket[len("eval_len"):] if bucket.startswith("eval_len") else bucket


_BUCKET_RANGE_INCLUSIVE_RE = re.compile(r"^(\d+)\s*-\s*(\d+)$")


def bucket_range_upper_bound(bucket: str) -> int | None:
    """Inclusive upper bound of ``lo-hi`` bucket names (second integer), or ``None``."""
    b = _normalize_bucket(bucket.strip())
    m = _BUCKET_RANGE_INCLUSIVE_RE.match(b)
    if not m:
        return None
    lo, hi = int(m.group(1)), int(m.group(2))
    if lo > hi:
        return None
    return hi


def row_matches_bucket_end_digit(
    row: dict[str, str | int | float],
    allowed: frozenset[int],
) -> bool:
    """``True`` if the bucket's upper bound's ones digit is in *allowed*."""
    ub = bucket_range_upper_bound(str(row.get("bucket", "")))
    if ub is None:
        return False
    return (ub % 10) in allowed


def _parse_cli_bucket_end_digit(s: str) -> int:
    v = int(s.strip())
    if v < 0 or v > 9:
        raise argparse.ArgumentTypeError(f"bucket end digit must be 0–9, got {v!r}")
    return v


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build/update summary CSV from logs/**/summary*.txt."
        )
    )
    repo_root = Path(__file__).resolve().parents[2]
    default_logs_root = repo_root / "logs"
    default_csv = repo_root / "exports" / "summary.csv"

    parser.add_argument("--logs-root", type=Path, default=default_logs_root)
    parser.add_argument("--csv", type=Path, default=default_csv)
    parser.add_argument(
        "--bucket-end-digit",
        action="append",
        dest="bucket_end_digits",
        type=_parse_cli_bucket_end_digit,
        default=None,
        help=(
            "Keep only rows whose bucket upper bound ends in this decimal digit "
            "(ones place), e.g. 0 keeps 0-50 (50) and 51-100 (100); 9 keeps 25-49 (49). "
            "Repeat flag for multiple digits (OR). Omit to keep all buckets."
        ),
    )
    args = parser.parse_args()

    be_digits = frozenset(args.bucket_end_digits) if args.bucket_end_digits else None

    rows = build_or_update_csv(
        logs_root=args.logs_root,
        csv_path=args.csv,
        bucket_end_digits=be_digits,
    )
    print(f"Wrote/updated CSV: {args.csv} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
