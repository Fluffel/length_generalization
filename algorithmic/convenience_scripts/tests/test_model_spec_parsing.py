#!/usr/bin/env python3
"""Tests for the model-specification parser in ``generate_summary_csv.py``.

Run the tests::

    python algorithmic/convenience_scripts/tests/test_model_spec_parsing.py

Print how every model string found in ``logs/`` is parsed, which is the quickest
way to eyeball a new naming scheme::

    python algorithmic/convenience_scripts/tests/test_model_spec_parsing.py --dump

Three layers of checks:

1. ``ExplicitCaseTests``  — a hand-written table of real log strings and the
   spec each one must produce.  This is where the intended reading of an
   ambiguous name is pinned down.
2. ``RoundTripTests``     — renders model strings from known specs with a local
   copy of ``format_log_prefix``'s naming rules and asserts the parser recovers
   the spec.  Sweeps the whole kernel x layer-count space, so it catches
   version digits being swallowed by layer counts and vice versa.
3. ``LogCorpusTests``     — parses every distinct model string in ``logs/`` and
   asserts the result is plausible (known kernel, power-of-two layer count,
   d_model from the sweeps, ...).  Catches regressions on data no one wrote a
   case for.
"""

from __future__ import annotations

import argparse
import re
import sys
import unittest
from itertools import product
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from generate_summary_csv import (  # noqa: E402
    KNOWN_KERNELS,
    parse_model_spec,
    tokenize_model,
)

def _find_logs_root(start: Path) -> Path:
    """Nearest ancestor directory holding a ``logs/`` tree.

    Keeps the tests working from both copies of the script: the one under
    ``algorithmic/convenience_scripts`` and the one in the log-analysis repo.
    """
    for parent in start.parents:
        if (parent / "logs").is_dir():
            return parent / "logs"
    return start.parents[-1] / "logs"


LOGS_ROOT = _find_logs_root(Path(__file__).resolve())


# ── Reference renderer ───────────────────────────────────────────────────────
# Mirrors algorithmic/language_modeling_train.py::format_log_prefix.  Kept
# separate from the parser so a round trip through both actually proves
# something.

def render_model_spec(
    *,
    family: str,
    olmo: bool = False,
    kernel: str | None = None,
    motif: str | None = None,
    n_layer: int,
    n_head: int | None = None,
    d_model: int,
    dropout: float,
    mlp: int | None = None,
    pe: bool | None = None,
    ln: bool | None = None,
    ne: bool | None = None,
    steps_k: float | None = None,
    lr: float | None = None,
    kernel_first: bool = False,
) -> str:
    """Build a model string the way the training run does.

    ``kernel_first`` selects the legacy hybrid spelling that put the kernel
    before the layer-ordering motif (``hybs4aaas1l...``) instead of after it
    (``hybaass42l...``).
    """
    parts: list[str] = []
    if family == "transformer":
        parts.append("olmolm" if olmo else "lm")
        parts += [f"{n_layer}l", f"{n_head}h", f"{d_model}d", f"{dropout}dr"]
    elif family == "ssm":
        parts.append("olmo" if olmo else "ssm")
        if kernel:
            parts.append(kernel)
        parts += [f"{n_layer}l", f"{d_model}d", f"{dropout}dr"]
    else:
        parts.append("olmohyb" if olmo else "hyb")
        ordered = [kernel, motif] if kernel_first else [motif, kernel]
        parts += [p for p in ordered if p]
        parts += [f"{n_layer}l", f"{n_head}h", f"{d_model}d", f"{dropout}dr"]

    if mlp is not None:
        parts.append(f"{mlp}mlp")
    if ne is not None:
        parts.append("ne" if ne else "none")
    if pe is not None:
        parts.append("pe" if pe else "nope")
    if ln is not None:
        parts.append("ln" if ln else "noln")
    if steps_k is not None:
        parts.append(f"stp{steps_k:.3g}k")
    if lr is not None:
        parts.append(f"{lr}lr")
    return "".join(parts)


def _flag(value: bool | None) -> str:
    return "-" if value is None else str(value)


def expected_spec(
    *,
    family: str,
    olmo: bool = False,
    kernel: str | None = None,
    motif: str | None = None,
    n_layer: int,
    n_head: int | None = None,
    d_model: int,
    dropout: float,
    mlp: int | None = None,
    pe: bool | None = None,
    ln: bool | None = None,
    ne: bool | None = None,
    steps_k: float | None = None,
    **_ignored,
) -> dict[str, str]:
    """The columns ``parse_model_spec`` must produce for the same arguments."""
    if family == "transformer":
        arch, expected_kernel = "lm", "-"
    elif family == "ssm":
        arch = "ssm"
        expected_kernel = kernel or "s4"
    else:
        arch = "hyb"
        expected_kernel = kernel or "s4"
    if olmo:
        arch = "olmo" + arch
        if family != "transformer":
            expected_kernel = kernel or "gdn1"
    if expected_kernel == "gdn":
        expected_kernel = "gdn1"

    return {
        "arch": arch,
        "layers": str(n_layer),
        "heads": "-" if n_head is None else str(n_head),
        "d_model": str(d_model),
        "dropout": str(dropout),
        "mlp_size": "-" if mlp is None else str(mlp),
        "kernel": expected_kernel,
        "pe": _flag(pe),
        "ln": _flag(ln),
        "ne": _flag(ne),
        "train_steps_k": "-" if steps_k is None else f"{steps_k:.3g}",
        "layer_order": motif or "-",
    }


# ── 1. Explicit cases ────────────────────────────────────────────────────────

def _spec(arch, kernel, layers, heads, d_model, dropout, **rest):
    base = {
        "arch": arch,
        "kernel": kernel,
        "layers": layers,
        "heads": heads,
        "d_model": d_model,
        "dropout": dropout,
        "mlp_size": "-",
        "pe": "-",
        "ln": "-",
        "ne": "-",
        "train_steps_k": "-",
        "layer_order": "-",
    }
    base.update(rest)
    return base


# (model string, expected columns).  Every entry is a string that occurs in
# logs/, except where marked as a synthetic ambiguity probe.
EXPLICIT_CASES: list[tuple[str, dict[str, str]]] = [
    # OLMo GDN before the gdn1/gdn2 split: bare "gdn", so the digits after it
    # are the layer count.  Those runs used GDN1, hence the kernel column.
    (
        "olmogdn1l16d0drnestp30k0.001lr",
        _spec("olmossm", "gdn1", "1", "-", "16", "0", ne="True", train_steps_k="30"),
    ),
    (
        "olmogdn2l64d0.1drnestp30k0.0003lr",
        _spec("olmossm", "gdn1", "2", "-", "64", "0.1", ne="True", train_steps_k="30"),
    ),
    (
        "olmogdn4l16d0drnestp30k0.001lr",
        _spec("olmossm", "gdn1", "4", "-", "16", "0", ne="True", train_steps_k="30"),
    ),
    # Versioned kernels: "gdn1"/"gdn2" followed by the layer count.
    (
        "olmogdn11l16d0drnestp30k0.001lr",
        _spec("olmossm", "gdn1", "1", "-", "16", "0", ne="True", train_steps_k="30"),
    ),
    (
        "olmogdn11l8d0.1drnestp30k0.001lr",
        _spec("olmossm", "gdn1", "1", "-", "8", "0.1", ne="True", train_steps_k="30"),
    ),
    (
        "olmogdn21l16d0drnestp30k0.001lr",
        _spec("olmossm", "gdn2", "1", "-", "16", "0", ne="True", train_steps_k="30"),
    ),
    (
        "olmogdn21l8d0drnestp30k0.001lr",
        _spec("olmossm", "gdn2", "1", "-", "8", "0", ne="True", train_steps_k="30"),
    ),
    # Synthetic: gdn2 with more than one layer keeps both digits apart.
    (
        "olmogdn24l64d0drnestp30k0.001lr",
        _spec("olmossm", "gdn2", "4", "-", "64", "0", ne="True", train_steps_k="30"),
    ),
    # OLMo hybrid runs predate the split and name no kernel at all.
    (
        "olmohybas2l4h256d0drnestp30k0.001lr",
        _spec(
            "olmohyb", "gdn1", "2", "4", "256", "0",
            ne="True", train_steps_k="30", layer_order="as",
        ),
    ),
    (
        "olmohybsssa2l2h64d0.1drnestp30k0.001lr",
        _spec(
            "olmohyb", "gdn1", "2", "2", "64", "0.1",
            ne="True", train_steps_k="30", layer_order="sssa",
        ),
    ),
    # "mamba" followed by the layer count must not be read as the "mamba2"
    # kernel with no layer count.
    (
        "ssmmamba2l64d0dr1mlplnstp30k0.001lr",
        _spec(
            "ssm", "mamba", "2", "-", "64", "0",
            mlp_size="1", ln="True", train_steps_k="30",
        ),
    ),
    (
        "ssmmamba4l256d0.1dr1mlplnstp30k0.0001lr",
        _spec(
            "ssm", "mamba", "4", "-", "256", "0.1",
            mlp_size="1", ln="True", train_steps_k="30",
        ),
    ),
    # Synthetic: an actual mamba2 run spells the layer count out separately.
    (
        "ssmmamba22l64d0dr1mlplnstp30k0.001lr",
        _spec(
            "ssm", "mamba2", "2", "-", "64", "0",
            mlp_size="1", ln="True", train_steps_k="30",
        ),
    ),
    # s4, whose "4" is part of the kernel name rather than the layer count.
    (
        "ssms44l64d0dr2mlplnstp30k0.001lr",
        _spec(
            "ssm", "s4", "4", "-", "64", "0",
            mlp_size="2", ln="True", train_steps_k="30",
        ),
    ),
    (
        "ssms416l64d0dr2mlplnstp60k0.001lr",
        _spec(
            "ssm", "s4", "16", "-", "64", "0",
            mlp_size="2", ln="True", train_steps_k="60",
        ),
    ),
    # Both hybrid spellings: kernel before the motif (legacy) and after it.
    (
        "hybs4aaas4l4h64d0dr2mlppelnstp60k0.001lr",
        _spec(
            "hyb", "s4", "4", "4", "64", "0",
            mlp_size="2", pe="True", ln="True", train_steps_k="60", layer_order="aaas",
        ),
    ),
    (
        "hybaass42l1h16d0.1dr2mlppelnstp30k0.001lr",
        _spec(
            "hyb", "s4", "2", "1", "16", "0.1",
            mlp_size="2", pe="True", ln="True", train_steps_k="30", layer_order="aas",
        ),
    ),
    (
        "hybmambasa1l2h256d0dr1mlppelnstp30k0.0001lr",
        _spec(
            "hyb", "mamba", "1", "2", "256", "0",
            mlp_size="1", pe="True", ln="True", train_steps_k="30", layer_order="sa",
        ),
    ),
    # Transformers carry no kernel.
    (
        "olmolm2l4h64d0.1drstp30k0.001lr",
        _spec("olmolm", "-", "2", "4", "64", "0.1", train_steps_k="30"),
    ),
    (
        "lm0.0reg1l1h64d0dr2mlpnopelnstp30k0.001lr",
        _spec(
            "lm", "-", "1", "1", "64", "0",
            mlp_size="2", pe="False", ln="True", train_steps_k="30",
        ),
    ),
    # Legacy structure-only names.
    ("2l4h64d", _spec("lm", "-", "2", "4", "64", "-")),
    ("ssm16l256d0.1dr", _spec("ssm", "s4", "16", "-", "256", "0.1")),
]


class ExplicitCaseTests(unittest.TestCase):
    def test_expected_spec_per_model_string(self):
        for model, expected in EXPLICIT_CASES:
            with self.subTest(model=model):
                got = parse_model_spec(model)
                self.assertEqual({k: got[k] for k in expected}, expected)

    def test_tokens_cover_the_whole_string(self):
        """No character may be dropped, or some feature went silently missing."""
        for model, _ in EXPLICIT_CASES:
            with self.subTest(model=model):
                self.assertEqual("".join(tokenize_model(model)), model.lower())


# ── 2. Round trip against the naming rules ───────────────────────────────────

LAYER_COUNTS = (1, 2, 3, 4, 8, 16, 32)
D_MODELS = (4, 8, 16, 64, 256)
HEADS = (1, 2, 4)
DROPOUTS = (0, 0.1)
MOTIFS = ("sa", "as", "aas", "aaas", "sas", "sssa")


def is_ambiguous(kernel: str | None, n_layer: int) -> bool:
    """``True`` if ``<kernel><n_layer>l`` also reads as a versioned sibling kernel.

    ``gdn16l`` is either GDN with 16 layers or GDN1 with 6, and ``mamba32l`` is
    either Mamba with 32 layers or Mamba3 with 2.  No name carries enough
    information to decide, so these are excluded from the round trip and the
    parser's documented preference is pinned in ``AmbiguityTests`` instead.
    """
    if kernel is None or len(str(n_layer)) < 2:
        return False
    versions = [k[len(kernel):] for k in KNOWN_KERNELS if k != kernel and k.startswith(kernel)]
    return any(str(n_layer).startswith(v) for v in versions)


class RoundTripTests(unittest.TestCase):
    def _check(self, **kwargs):
        model = render_model_spec(**kwargs)
        expected = expected_spec(**kwargs)
        got = parse_model_spec(model)
        self.assertEqual(
            {k: got[k] for k in expected},
            expected,
            f"model={model!r} tokens={tokenize_model(model)}",
        )

    def test_ssm(self):
        kernels = sorted(KNOWN_KERNELS) + [None]
        for kernel, n_layer, d_model in product(kernels, LAYER_COUNTS, D_MODELS):
            if is_ambiguous(kernel, n_layer):
                continue
            olmo = kernel is not None and kernel.startswith("gdn")
            with self.subTest(kernel=kernel, n_layer=n_layer, d_model=d_model):
                self._check(
                    family="ssm",
                    olmo=olmo,
                    kernel=kernel,
                    n_layer=n_layer,
                    d_model=d_model,
                    dropout=0.1,
                    mlp=None if olmo else 2,
                    ne=True if olmo else None,
                    ln=None if olmo else True,
                    steps_k=30,
                    lr=0.001,
                )

    def test_hybrid_both_kernel_positions(self):
        kernels = sorted(KNOWN_KERNELS) + [None]
        for kernel, motif, kernel_first, n_layer in product(
            kernels, MOTIFS, (False, True), LAYER_COUNTS
        ):
            if is_ambiguous(kernel, n_layer):
                continue
            olmo = kernel is not None and kernel.startswith("gdn")
            with self.subTest(
                kernel=kernel, motif=motif, kernel_first=kernel_first, n_layer=n_layer
            ):
                self._check(
                    family="hybrid",
                    olmo=olmo,
                    kernel=kernel,
                    motif=motif,
                    kernel_first=kernel_first,
                    n_layer=n_layer,
                    n_head=2,
                    d_model=64,
                    dropout=0,
                    mlp=None if olmo else 2,
                    ne=True if olmo else None,
                    pe=None if olmo else True,
                    ln=None if olmo else True,
                    steps_k=30,
                    lr=0.001,
                )

    def test_transformer(self):
        for olmo, n_layer, n_head, d_model, dropout in product(
            (False, True), LAYER_COUNTS, HEADS, D_MODELS, DROPOUTS
        ):
            with self.subTest(olmo=olmo, n_layer=n_layer, d_model=d_model):
                self._check(
                    family="transformer",
                    olmo=olmo,
                    n_layer=n_layer,
                    n_head=n_head,
                    d_model=d_model,
                    dropout=dropout,
                    mlp=None if olmo else 2,
                    pe=None if olmo else False,
                    ln=None if olmo else True,
                    steps_k=30,
                    lr=0.001,
                )


class AmbiguityTests(unittest.TestCase):
    """Pins the reading chosen where a name admits more than one."""

    def test_version_digit_wins_over_two_digit_layer_count(self):
        # "gdn16l" is read as GDN1 with 6 layers, not GDN with 16.  Every log
        # written since GDN2 exists names the version, so the versioned reading
        # is the right default; the corpus test guards against a legacy name in
        # that range showing up later.
        spec = parse_model_spec("olmogdn16l64d0drnestp30k0.001lr")
        self.assertEqual(spec["kernel"], "gdn1")
        self.assertEqual(spec["layers"], "6")

        spec = parse_model_spec("ssmmamba32l64d0dr2mlplnstp30k0.001lr")
        self.assertEqual(spec["kernel"], "mamba3")
        self.assertEqual(spec["layers"], "2")

    def test_kernel_needs_a_layer_count_behind_it(self):
        # The "s4" here is the motif's last "s" plus the layer count's "4".
        spec = parse_model_spec("olmohybsas4l4h64d0drnestp30k0.001lr")
        self.assertEqual(spec["kernel"], "gdn1")
        self.assertEqual(spec["layer_order"], "sas")
        self.assertEqual(spec["layers"], "4")


# ── 3. Whole-corpus plausibility ─────────────────────────────────────────────

_MODEL_TOKEN_RE = re.compile(r"^(\S+)")
# Interleaved writes from parallel runs leave fragments and glued-together
# names in the summary files.  A well-formed name mentions each structural
# anchor at most once and holds nothing but lowercase letters, digits and dots.
_STRUCTURAL_ANCHORS = ("olmo", "ssm", "hyb", "stp")


def log_model_strings(logs_root: Path = LOGS_ROOT) -> list[str]:
    """Every distinct model string in ``logs/**/summary*.txt``."""
    models: set[str] = set()
    for summary_file in sorted(logs_root.glob("**/summary*.txt")):
        with summary_file.open("r", errors="replace") as f:
            for raw_line in f:
                line = raw_line.replace("\x00", "").strip()
                if (m := _MODEL_TOKEN_RE.match(line)) and "eval_len" in line:
                    models.add(m.group(1))
    return sorted(models)


def is_well_formed(model: str) -> bool:
    if not re.fullmatch(r"[a-z0-9.]+", model):
        return False
    if not re.search(r"[0-9]+l[0-9]+", model):
        return False
    return all(model.count(anchor) <= 1 for anchor in _STRUCTURAL_ANCHORS)


@unittest.skipUnless(LOGS_ROOT.is_dir(), f"no logs directory at {LOGS_ROOT}")
class LogCorpusTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.models = [m for m in log_model_strings() if is_well_formed(m)]

    def test_corpus_is_not_empty(self):
        self.assertGreater(len(self.models), 500)

    def test_every_model_parses_to_plausible_values(self):
        allowed_kernels = {"-"} | {k for k in KNOWN_KERNELS if k != "gdn"}
        for model in self.models:
            with self.subTest(model=model):
                spec = parse_model_spec(model)
                self.assertIn(spec["arch"], {"lm", "ssm", "hyb", "olmolm", "olmossm", "olmohyb"})
                self.assertIn(spec["kernel"], allowed_kernels)
                self.assertIn(int(spec["layers"]), LAYER_COUNTS)
                self.assertIn(int(spec["d_model"]), D_MODELS)
                if spec["dropout"] != "-":
                    self.assertIn(float(spec["dropout"]), DROPOUTS)
                if spec["heads"] != "-":
                    self.assertIn(int(spec["heads"]), HEADS)

    def test_arch_and_kernel_agree(self):
        for model in self.models:
            with self.subTest(model=model):
                spec = parse_model_spec(model)
                has_kernel = spec["kernel"] != "-"
                self.assertEqual(has_kernel, "ssm" in spec["arch"] or "hyb" in spec["arch"])
                if spec["arch"].startswith("olmo") and has_kernel:
                    self.assertTrue(spec["kernel"].startswith("gdn"))

    def test_hybrid_models_have_a_layer_order(self):
        for model in self.models:
            if "hyb" not in parse_model_spec(model)["arch"]:
                continue
            with self.subTest(model=model):
                self.assertRegex(parse_model_spec(model)["layer_order"], r"^[as]+$")

    def test_tokens_cover_the_whole_string(self):
        for model in self.models:
            with self.subTest(model=model):
                self.assertEqual("".join(tokenize_model(model)), model)


# ── Report mode ──────────────────────────────────────────────────────────────

DUMP_COLUMNS = (
    "arch", "kernel", "layers", "heads", "d_model", "dropout",
    "mlp_size", "pe", "ln", "ne", "train_steps_k", "layer_order",
)


def dump_log_model_specs(logs_root: Path) -> None:
    models = log_model_strings(logs_root)
    width = max((len(m) for m in models), default=10)
    header = f"{'model':<{width}}  " + "  ".join(f"{c:<11}" for c in DUMP_COLUMNS)
    print(header)
    print("-" * len(header))
    skipped: list[str] = []
    for model in models:
        if not is_well_formed(model):
            skipped.append(model)
            continue
        spec = parse_model_spec(model)
        print(f"{model:<{width}}  " + "  ".join(f"{spec[c]:<11}" for c in DUMP_COLUMNS))
    print(f"\n{len(models) - len(skipped)} model strings from {logs_root}")
    if skipped:
        print(f"{len(skipped)} skipped as malformed (interleaved log writes):")
        for model in skipped:
            print(f"  {model!r}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dump",
        action="store_true",
        help="Print the parsed spec of every model string in the logs and exit.",
    )
    parser.add_argument("--logs-root", type=Path, default=LOGS_ROOT)
    args, remaining = parser.parse_known_args()

    if args.dump:
        dump_log_model_specs(args.logs_root)
        return 0
    unittest.main(argv=[sys.argv[0], *remaining], verbosity=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
