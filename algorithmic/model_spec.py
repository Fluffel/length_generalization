"""Model specifications: YAML files that fully describe the model to train.

A spec fixes everything model-specific -- family, positional encoding, layer
norm, SSM kernel, hybrid layer pattern, OLMo mixer -- plus the architecture
slots the training sweep iterates over. Everything else (task, seeds, step
budgets, eval cadence, ...) stays on the command line, so one spec is reused
across tasks and schedules.

Specs live in ``model_specs/<family>/<variant>.yaml`` and are selected with
``language_modeling_train.py --model <family>/<variant>``. A spec may name
another one via ``extends:``; the child's keys override the parent's, and
``architectures`` is replaced wholesale rather than merged.
"""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import Any

import yaml

from utils import ArchSlot, RunConfig

MODEL_SPEC_ROOT = Path(__file__).resolve().parent / "model_specs"

MODEL_FAMILIES = ("transformer", "ssm", "hybrid")

# RunConfig fields a spec may set, i.e. everything that defines the model itself.
_SPEC_FIELDS: dict[str, type] = {
    "model_family": str,
    "use_nope": bool,
    "use_olmo_core": bool,
    "regularize": float,
    "ssm_kernel": str,
    "hybrid_layer_pattern": str,
    "olmo_gdn_variant": str,
    "olmo_gdn_allow_neg_eigval": bool,
    "olmo_gdn_expand_v": float,
    "olmo_gdn_head_dim_multiplier": float,
}

_ARCH_FIELDS: dict[str, type] = {
    "n_layer": int,
    "n_head": int,
    "d_model": int,
    "between_block_mlp_layers": int,
    "layer_norm": bool,
    "dropout": float,
    "lr": float,
}


def available_model_specs() -> list[str]:
    """Spec names accepted by ``--model``, relative to ``MODEL_SPEC_ROOT``."""
    names = {
        path.relative_to(MODEL_SPEC_ROOT).with_suffix("").as_posix()
        for pattern in ("*.yaml", "*.yml")
        for path in MODEL_SPEC_ROOT.rglob(pattern)
    }
    return sorted(names)


def resolve_model_spec_path(name: str) -> Path:
    """Locate ``name`` as a spec inside ``model_specs/`` or as a plain file path."""
    candidates = [
        MODEL_SPEC_ROOT / name,
        MODEL_SPEC_ROOT / f"{name}.yaml",
        MODEL_SPEC_ROOT / f"{name}.yml",
        Path(name),
        Path(f"{name}.yaml"),
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    available = "\n  ".join(available_model_specs())
    raise ValueError(f"Unknown model spec {name!r}. Available specs:\n  {available}")


def load_model_spec(name: str) -> RunConfig:
    """Build a ``RunConfig`` whose model fields and sweep come from spec ``name``.

    All other fields keep their ``RunConfig`` defaults, to be overridden by the
    caller (the CLI in ``language_modeling_train.py``).
    """
    path = resolve_model_spec_path(name)
    spec = _read_spec(path, ())
    where = path.as_posix()

    architectures = spec.pop("architectures", None)
    if not architectures:
        raise ValueError(f"{where}: 'architectures' must list at least one entry.")
    if not isinstance(architectures, list):
        raise ValueError(f"{where}: 'architectures' must be a list of mappings.")

    unknown = sorted(set(spec) - set(_SPEC_FIELDS))
    if unknown:
        allowed = ", ".join(sorted(_SPEC_FIELDS))
        raise ValueError(
            f"{where}: unknown key(s) {', '.join(unknown)}. A spec may set only "
            f"model fields ({allowed}) plus 'architectures' and 'extends'; other "
            "hyperparameters are command-line arguments."
        )
    if "model_family" not in spec:
        raise ValueError(f"{where}: 'model_family' is required.")

    fields = {key: _coerce(_SPEC_FIELDS[key], value, f"{where}: {key}") for key, value in spec.items()}
    if fields["model_family"] not in MODEL_FAMILIES:
        raise ValueError(
            f"{where}: model_family must be one of {', '.join(MODEL_FAMILIES)}, "
            f"got {fields['model_family']!r}."
        )

    slots: list[ArchSlot] = []
    for index, entry in enumerate(architectures):
        slots.extend(_expand_arch_entry(entry, f"{where}: architectures[{index}]"))

    return RunConfig(architectures=slots, model_spec=name, **fields)


def _read_spec(path: Path, seen: tuple[Path, ...]) -> dict[str, Any]:
    """Spec at ``path`` merged onto whatever it ``extends``, child keys winning."""
    path = path.resolve()
    if path in seen:
        chain = " -> ".join(p.as_posix() for p in (*seen, path))
        raise ValueError(f"Circular 'extends' in model specs: {chain}")

    raw = yaml.safe_load(path.read_text()) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{path.as_posix()}: expected a mapping at the top level.")

    parent = raw.pop("extends", None)
    if parent is None:
        return raw

    merged = _read_spec(_resolve_extends(parent, path), (*seen, path))
    merged.update(raw)
    return merged


def _resolve_extends(parent: Any, child_path: Path) -> Path:
    if not isinstance(parent, str):
        raise ValueError(f"{child_path.as_posix()}: 'extends' must be a spec name or path.")
    for candidate in (child_path.parent / parent, MODEL_SPEC_ROOT / parent):
        if candidate.is_file():
            return candidate
    raise ValueError(f"{child_path.as_posix()}: cannot find extended spec {parent!r}.")


def _expand_arch_entry(entry: Any, where: str) -> list[ArchSlot]:
    """One ``architectures`` entry as slots: list-valued fields form a cross product."""
    if not isinstance(entry, dict):
        raise ValueError(f"{where}: expected a mapping of architecture fields.")
    unknown = sorted(set(entry) - set(_ARCH_FIELDS))
    if unknown:
        allowed = ", ".join(_ARCH_FIELDS)
        raise ValueError(f"{where}: unknown field(s) {', '.join(unknown)}; allowed: {allowed}.")
    if "n_layer" not in entry:
        raise ValueError(f"{where}: 'n_layer' is required.")

    names = list(entry)
    value_lists = [_coerce_values(_ARCH_FIELDS[name], entry[name], f"{where}.{name}") for name in names]
    return [ArchSlot(**dict(zip(names, combo))) for combo in itertools.product(*value_lists)]


def _coerce_values(kind: type, raw: Any, where: str) -> list[Any]:
    values = raw if isinstance(raw, list) else [raw]
    if not values:
        raise ValueError(f"{where}: empty list of values.")
    return [_coerce(kind, value, where) for value in values]


def _coerce(kind: type, value: Any, where: str) -> Any:
    if kind is bool:
        if not isinstance(value, bool):
            raise ValueError(f"{where}: expected true or false, got {value!r}.")
        return value
    if isinstance(value, (list, dict)):
        raise ValueError(f"{where}: expected a single {kind.__name__} value, got {value!r}.")
    try:
        return kind(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{where}: expected {kind.__name__}, got {value!r}.") from exc
