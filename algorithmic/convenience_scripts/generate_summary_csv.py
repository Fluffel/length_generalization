#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from dataclasses import dataclass
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
KNOWN_KERNELS: frozenset[str] = frozenset(
    {"s4", "s6", "mamba", "mamba2", "mamba3", "gdn", "gdn1", "gdn2"}
)

# Kernels that older logs wrote under a shorter, unversioned name.  ``gdn`` was
# the only OLMo GatedDeltaNet variant before GDN2 existed, so those runs are
# GDN1 and are reported as such.
KERNEL_ALIASES: dict[str, str] = {"gdn": "gdn1"}

# Kernel used by OLMo-backed specs that name no kernel at all (all such runs
# predate the gdn1/gdn2 split).
OLMO_DEFAULT_KERNEL = "gdn1"
DEFAULT_KERNEL = "s4"

# Kernel placeholders use \x01N\x01 (SOH byte as delimiter) so they cannot be
# split by any letter or digit pattern in the tokeniser regex.
_KERNEL_SLOT_RE = re.compile(r"\x01(\d+)\x01")

# Every kernel literal is immediately followed by the layer count (``<n>l``),
# optionally preceded by the hybrid layer-ordering motif in the legacy
# ``hyb<kernel><motif><n>l`` spelling.  Requiring that tail is what tells the
# real kernel apart from a coincidental substring: in ``hybaaas4l`` the ``s4``
# is the motif's last ``s`` plus the layer count's ``4``, and in ``ssmmamba2l``
# the ``2`` is the layer count rather than the ``mamba2`` version digit.
_KERNEL_TAIL_RE = re.compile(r"(?:a|s)*[0-9]+l")


def _kernel_match_is_real(s: str, pos: int, kernel: str) -> bool:
    """``True`` if the *kernel* literal found at *pos* is followed by a layer count."""
    return _KERNEL_TAIL_RE.match(s, pos + len(kernel)) is not None


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

# One CSV row = one summary line = one training run.  Its eval bins live in the
# ``bin{i}_range`` / ``bin{i}_acc`` column pairs, numbered from 1 in ascending
# length order, so ``bin1`` is always the training bin.  Runs with fewer bins
# than the widest run in the file leave the trailing pairs empty.
RUN_META_COLUMNS = [
    "task",
    "model",
    "learning_rate",
    "num_bins",
    "bins",           # bin ranges joined by "|", e.g. "0-49|50-99|100-149"
    "train_range",    # first (= training) bin, e.g. "0-49"
    "train_len_min",
    "train_len_max",
    "max_eval_len",   # upper bound of the last bin
    "source_file",
    "source_line",
]

BIN_SIGNATURE_SEP = "|"


def bin_column_names(index: int) -> tuple[str, str]:
    """Column names holding the range and accuracy of the *index*-th bin (1-based)."""
    return (f"bin{index}_range", f"bin{index}_acc")


def csv_columns(max_bins: int) -> list[str]:
    """Full column list for a CSV whose widest run has *max_bins* eval bins."""
    bin_cols = [
        col for i in range(1, max(max_bins, 1) + 1) for col in bin_column_names(i)
    ]
    return [*RUN_META_COLUMNS, *bin_cols, *MODEL_SPEC_COLUMNS, *TASK_PARAM_COLUMNS]

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

    kernels = [tok for tok in tokens if tok in KNOWN_KERNELS]

    pure_alpha: set[str] = set()
    features: dict[str, str] = {}  # alpha-key → numeric-value (last wins)
    ith_feature = 0
    for tok in tokens:
        # Kernel names are handled separately: versioned ones like "gdn1" would
        # otherwise be read as the numeric feature ("gdn", "1") and shift the
        # positional fallback used for specs that omit a feature suffix.
        if tok in KNOWN_KERNELS:
            continue
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
    elif "ssm" in pure_alpha or kernels:
        arch += "ssm"
    # elif arch.startswith("olmo"):
    #     # ``olmo`` alone (no explicit ``lm`` / ``hyb`` / ``ssm``): OLMo SSM-backed stack → gdn.
    #     arch += "ssm"
    else:
        # Structure-only specs like ``2l1h64d`` omit an arch keyword; default to LM.
        arch += "lm"

    # ── SSM kernel ───────────────────────────────────────────────────────────
    # Kernels are matched as whole tokens (KNOWN_KERNELS), so we just take the
    # first kernel token in the stream and fall back to the family's default
    # when the spec names none.
    if "hyb" in arch or "ssm" in arch:
        default = OLMO_DEFAULT_KERNEL if "olmo" in arch else DEFAULT_KERNEL
        kernel = kernels[0] if kernels else default
        kernel = KERNEL_ALIASES.get(kernel, kernel)
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


def parse_line_bins(line: str) -> list[tuple[str, float]]:
    """``(range, accuracy)`` pairs of one summary line, ascending by length.

    Ranges are de-duplicated (first occurrence wins) and sorted by their
    bounds, so index 0 is always the training bin regardless of the order in
    which the log wrote them.
    """
    seen: dict[str, float] = {}
    for range_str, acc_str in _LINE_BUCKET_RE.findall(line):
        rng = _normalize_bucket(range_str)
        if rng not in seen:
            seen[rng] = float(acc_str)
    return sorted(seen.items(), key=lambda kv: (bucket_range_bounds(kv[0]) or (0, 0)))


def bins_signature(bins: list[tuple[str, float]]) -> str:
    """Stable signature of a run's bin layout, e.g. ``0-49|50-99|100-149``."""
    return BIN_SIGNATURE_SEP.join(rng for rng, _ in bins)


def parse_summary_line(
    line: str,
    task: str,
    task_params: dict[str, str] | None = None,
    *,
    source_file: str = "",
    source_line: int = 0,
) -> dict[str, str | int | float] | None:
    """Turn one summary line into a single wide CSV row, or ``None`` if unparseable."""
    line = line.strip()
    m_model = _LINE_MODEL_RE.match(line)
    m_lr = _LINE_LR_RE.search(line)
    bins = parse_line_bins(line)
    if not m_model or not m_lr or not bins:
        return None

    model = m_model.group(1)
    train_bounds = bucket_range_bounds(bins[0][0])
    last_bounds = bucket_range_bounds(bins[-1][0])

    row: dict[str, str | int | float] = {
        "task": task,
        "model": model,
        "learning_rate": float(m_lr.group(1)),
        "num_bins": len(bins),
        "bins": bins_signature(bins),
        "train_range": bins[0][0],
        "train_len_min": train_bounds[0] if train_bounds else "",
        "train_len_max": train_bounds[1] if train_bounds else "",
        "max_eval_len": last_bounds[1] if last_bounds else "",
        "source_file": source_file,
        "source_line": source_line,
    }
    for i, (rng, acc) in enumerate(bins, start=1):
        range_col, acc_col = bin_column_names(i)
        row[range_col] = rng
        row[acc_col] = acc
    row.update(parse_model_spec(model))  # type: ignore[arg-type]
    row.update(task_params or {})
    return row


def load_csv_rows(csv_path: Path) -> list[dict[str, str]]:
    if not csv_path.exists():
        return []
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def is_wide_csv(rows: list[dict[str, str]]) -> bool:
    """``True`` if *rows* use the one-row-per-run layout (bin columns present)."""
    return bool(rows) and bin_column_names(1)[1] in rows[0]


def write_csv_rows(csv_path: Path, rows: list[dict[str, str | int | float]]) -> None:
    max_bins = max((int(r.get("num_bins") or 0) for r in rows), default=0)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=csv_columns(max_bins),
            extrasaction="ignore",
            restval="",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _run_key(row: dict[str, str | int | float]) -> tuple:
    """Identity of a run for de-duplication across log scans and merged CSVs."""
    accs = tuple(
        str(row.get(bin_column_names(i)[1], ""))
        for i in range(1, int(row.get("num_bins") or 0) + 1)
    )
    return (
        str(row.get("task", "")),
        str(row.get("model", "")),
        str(row.get("learning_rate", "")),
        str(row.get("bins", "")),
        accs,
    )


@dataclass(frozen=True, kw_only=True)
class RunFilter:
    """Which runs to keep when building the CSV.

    Every criterion is optional; a run must satisfy all of the ones that are
    set.  Ranges are compared as normalised ``lo-hi`` strings.
    """

    tasks: frozenset[str] = frozenset()
    bin_counts: frozenset[int] = frozenset()
    min_bins: int | None = None
    max_bins: int | None = None
    train_ranges: frozenset[str] = frozenset()
    exclude_train_ranges: frozenset[str] = frozenset()
    bin_signatures: frozenset[str] = frozenset()
    bucket_end_digits: frozenset[int] = frozenset()

    def matches(self, row: dict[str, str | int | float]) -> bool:
        if self.tasks and str(row.get("task", "")) not in self.tasks:
            return False
        n = int(row.get("num_bins") or 0)
        if self.bin_counts and n not in self.bin_counts:
            return False
        if self.min_bins is not None and n < self.min_bins:
            return False
        if self.max_bins is not None and n > self.max_bins:
            return False
        train_range = str(row.get("train_range", ""))
        if self.train_ranges and train_range not in self.train_ranges:
            return False
        if train_range in self.exclude_train_ranges:
            return False
        if self.bin_signatures and str(row.get("bins", "")) not in self.bin_signatures:
            return False
        if self.bucket_end_digits:
            ends = [
                bucket_range_upper_bound(rng)
                for rng in str(row.get("bins", "")).split(BIN_SIGNATURE_SEP)
                if rng
            ]
            if not ends or any(
                e is None or (e % 10) not in self.bucket_end_digits for e in ends
            ):
                return False
        return True


def read_summary_runs(
    logs_root: Path,
    run_filter: RunFilter | None = None,
) -> list[dict[str, str | int | float]]:
    """Parse every ``logs/**/summary*.txt`` line into one run row."""
    keep = run_filter or RunFilter()
    rows: list[dict[str, str | int | float]] = []
    for summary_file in sorted(logs_root.glob("**/summary*.txt")):
        task = summary_file.parent.name
        if keep.tasks and task not in keep.tasks:
            continue
        task_params = parse_task_params(task, summary_file)
        try:
            rel = str(summary_file.relative_to(logs_root))
        except ValueError:
            rel = str(summary_file)
        with summary_file.open("r", errors="replace") as f:
            for lineno, raw_line in enumerate(f, start=1):
                row = parse_summary_line(
                    raw_line,
                    task=task,
                    task_params=task_params,
                    source_file=rel,
                    source_line=lineno,
                )
                if row is not None and keep.matches(row):
                    rows.append(row)
    return rows


def refresh_existing_rows(
    rows: list[dict[str, str]],
) -> list[dict[str, str | int | float]]:
    """Re-derive spec/task columns on rows loaded from an existing wide CSV."""
    out: list[dict[str, str | int | float]] = []
    for row in rows:
        base: dict[str, str | int | float] = dict(row)
        task_defaults = TASK_PARAM_DEFAULTS.get(str(row.get("task", "")), {})
        for col, default_val in task_defaults.items():
            if str(base.get(col, "-")).strip() in {"", "-"}:
                base[col] = default_val
        base.update(parse_model_spec(str(row.get("model", ""))))  # type: ignore[arg-type]
        out.append(base)
    return out


def build_csv(
    logs_root: Path,
    csv_path: Path,
    *,
    run_filter: RunFilter | None = None,
    merge_existing: bool = False,
) -> list[dict[str, str | int | float]]:
    """Rebuild *csv_path* from the logs, one row per run.

    With *merge_existing*, runs already present in the CSV but no longer found
    in the logs are carried over instead of dropped.
    """
    rows = read_summary_runs(logs_root, run_filter)

    if merge_existing:
        existing = load_csv_rows(csv_path)
        if existing and not is_wide_csv(existing):
            print(
                f"Ignoring {csv_path}: legacy one-row-per-bin layout cannot be merged; "
                "rebuilding from logs instead."
            )
        elif existing:
            seen = {_run_key(r) for r in rows}
            carried = [
                r
                for r in refresh_existing_rows(existing)
                if _run_key(r) not in seen
                and (run_filter is None or run_filter.matches(r))
            ]
            rows.extend(carried)

    write_csv_rows(csv_path, rows)
    return rows


def summarize_runs(rows: list[dict[str, str | int | float]]) -> str:
    """Human-readable histogram of bin counts and train ranges."""
    by_bins: dict[int, int] = defaultdict(int)
    by_train: dict[str, int] = defaultdict(int)
    for row in rows:
        by_bins[int(row.get("num_bins") or 0)] += 1
        by_train[str(row.get("train_range", ""))] += 1
    bins_part = ", ".join(f"{n} bins: {c}" for n, c in sorted(by_bins.items()))
    train_part = ", ".join(
        f"{r}: {c}" for r, c in sorted(by_train.items(), key=lambda kv: -kv[1])
    )
    return f"  bin counts   -> {bins_part}\n  train ranges -> {train_part}"


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

    Kernels are tried longest-first, so a versioned name wins over its own
    prefix whenever both readings leave a valid layer count behind: ``gdn11l``
    is GDN1 with one layer rather than GDN with eleven.  New logs always write
    the version digit, and no legacy ``gdn`` run has a two-digit layer count,
    so that preference is unambiguous in practice.
    """
    s = model.strip().lstrip("\x00").lower()
    # Pre-extract kernels longest-first to avoid partial matches.
    slots: list[str] = []
    for kernel in sorted(KNOWN_KERNELS, key=len, reverse=True):
        i = 0
        while (pos := s.find(kernel, i)) != -1:
            if not _kernel_match_is_real(s, pos, kernel):
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


def bucket_range_bounds(bucket: str) -> tuple[int, int] | None:
    """Inclusive ``(lo, hi)`` bounds of a ``lo-hi`` bucket name, or ``None``."""
    m = _BUCKET_RANGE_INCLUSIVE_RE.match(_normalize_bucket(bucket.strip()))
    if not m:
        return None
    lo, hi = int(m.group(1)), int(m.group(2))
    if lo > hi:
        return None
    return (lo, hi)


def bucket_range_upper_bound(bucket: str) -> int | None:
    """Inclusive upper bound of ``lo-hi`` bucket names (second integer), or ``None``."""
    bounds = bucket_range_bounds(bucket)
    return bounds[1] if bounds else None


def _parse_cli_bucket_end_digit(s: str) -> int:
    v = int(s.strip())
    if v < 0 or v > 9:
        raise argparse.ArgumentTypeError(f"bucket end digit must be 0–9, got {v!r}")
    return v


def _split_cli_list(values: list[str]) -> list[str]:
    """Flatten repeated CLI flags that each may hold a comma-separated list."""
    out: list[str] = []
    for raw in values:
        out.extend(v.strip() for v in str(raw).split(",") if v.strip())
    return out


def _parse_cli_bin_counts(values: list[str]) -> frozenset[int]:
    counts: set[int] = set()
    for tok in _split_cli_list(values):
        try:
            n = int(tok)
        except ValueError:
            raise SystemExit(f"--num-bins: expected integers, got {tok!r}.") from None
        if n < 1:
            raise SystemExit("--num-bins values must be >= 1.")
        counts.add(n)
    return frozenset(counts)


def _parse_cli_ranges(values: list[str], flag: str) -> frozenset[str]:
    ranges: set[str] = set()
    for tok in _split_cli_list(values):
        norm = _normalize_bucket(tok)
        if bucket_range_bounds(norm) is None:
            raise SystemExit(f"{flag}: expected ranges like 0-49, got {tok!r}.")
        ranges.add(norm)
    return frozenset(ranges)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build summary CSV from logs/**/summary*.txt, one row per run. "
            "Each run's eval bins are written as bin1_range/bin1_acc, "
            "bin2_range/bin2_acc, ... in ascending length order, so bin1 is the "
            "training bin. Runs may have different bin counts; unused trailing "
            "bin columns are left empty and num_bins records the actual count."
        )
    )
    repo_root = Path(__file__).resolve().parents[2]
    default_logs_root = repo_root / "logs"
    default_csv = repo_root / "exports" / "summary.csv"

    parser.add_argument("--logs-root", type=Path, default=default_logs_root)
    parser.add_argument("--csv", type=Path, default=default_csv)
    parser.add_argument(
        "--task",
        action="append",
        default=[],
        help="Only include these tasks (repeatable, comma-separated lists allowed).",
    )
    parser.add_argument(
        "--num-bins",
        "--require-bins",
        dest="num_bins",
        action="append",
        default=[],
        metavar="N",
        help=(
            "Keep only runs with exactly N eval bins. Repeatable / comma-separated "
            "to allow several counts, e.g. --num-bins 3,6."
        ),
    )
    parser.add_argument("--min-bins", type=int, default=None, metavar="N")
    parser.add_argument("--max-bins", type=int, default=None, metavar="N")
    parser.add_argument(
        "--train-range",
        action="append",
        default=[],
        metavar="LO-HI",
        help=(
            "Keep only runs whose training bin (first bin) is one of these ranges, "
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
            "Keep only runs with exactly this bin layout, e.g. "
            '--bins "0-49|50-99|100-149" (repeat the flag to allow several layouts).'
        ),
    )
    parser.add_argument(
        "--bucket-end-digit",
        action="append",
        dest="bucket_end_digits",
        type=_parse_cli_bucket_end_digit,
        default=None,
        help=(
            "Keep only runs whose bin upper bounds all end in one of these decimal "
            "digits (ones place), e.g. 0 keeps 0-50/51-100; 9 keeps 0-49/50-99. "
            "Repeat flag for multiple digits (OR)."
        ),
    )
    parser.add_argument(
        "--merge-existing",
        action="store_true",
        help=(
            "Carry over runs that are already in the CSV but no longer present in "
            "the logs. Without this flag the CSV is rebuilt from the logs alone."
        ),
    )
    args = parser.parse_args()

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

    bin_signatures = {
        BIN_SIGNATURE_SEP.join(
            sorted(
                _parse_cli_ranges([raw.replace(BIN_SIGNATURE_SEP, ",")], "--bins"),
                key=lambda r: bucket_range_bounds(r) or (0, 0),
            )
        )
        for raw in args.bins
    }

    run_filter = RunFilter(
        tasks=frozenset(_split_cli_list(args.task)),
        bin_counts=_parse_cli_bin_counts(args.num_bins),
        min_bins=args.min_bins,
        max_bins=args.max_bins,
        train_ranges=_parse_cli_ranges(args.train_range, "--train-range"),
        exclude_train_ranges=_parse_cli_ranges(
            args.exclude_train_range, "--exclude-train-range"
        ),
        bin_signatures=frozenset(bin_signatures),
        bucket_end_digits=frozenset(args.bucket_end_digits or ()),
    )

    rows = build_csv(
        logs_root=args.logs_root,
        csv_path=args.csv,
        run_filter=run_filter,
        merge_existing=args.merge_existing,
    )
    print(f"Wrote CSV: {args.csv} ({len(rows)} runs)")
    if rows:
        print(summarize_runs(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
