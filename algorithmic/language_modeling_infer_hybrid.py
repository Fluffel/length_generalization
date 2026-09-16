"""
Load a trained hybrid / SSM / OLMo checkpoint and run inference.

Architecture is read from the weights filename, supporting both legacy and newer
modular names from language_modeling_train.py.
If the filename does not match known conventions, pass explicit architecture flags.

Use --eval-full to score the same eval bins materialised during training
(same --dataset-seed and --test-num) and print every success and failure.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import torch
import torch.nn as nn

from dataset_generators import ALL_TASKS, build_datasets
from transformers.trainer_utils import set_seed
from utils import ArchSlot, RunConfig

TASK_CHOICES = list(ALL_TASKS)

# Default from current train setup. Can be auto-adjusted from checkpoint shape.
TRAIN_LENGTH_RANGE = (0, 50)
MAX_TEST_LENGTH = 149

# Condor job ids look like "62555.9"; keep the suffix permissive.
_WEIGHTS_SUFFIX = r"_weights(?:_seed\d+_id.*)?\.pt$"

_WEIGHTS_NAME_RE_NEW = re.compile(
    rf"^hyb([as]+)_(\d+)l(\d+)h(\d+)d([0-9.]+)dr(smalllr)?{_WEIGHTS_SUFFIX}$"
)
_WEIGHTS_NAME_RE_LEGACY = re.compile(
    rf"^hyb(as|sa)(\d+)l(\d+)h(\d+)d([0-9.]+)dr(smalllr)?{_WEIGHTS_SUFFIX}$"
)
_WEIGHTS_NAME_RE_MODULAR = re.compile(
    r"^hyb([as]+)(mamba3|mamba2|mamba|s4)(\d+)l(\d+)h(\d+)d([0-9.]+)dr"
    r"(\d+)mlp(nope|pe)(noln|ln)"
    r"(?:frz(?:a|ssm)[0-9.]+)?"
    rf"stp([0-9.]+)k([0-9eE+\-\.]+)lr{_WEIGHTS_SUFFIX}$"
)
_WEIGHTS_NAME_RE_OLMO_HYB = re.compile(
    r"^olmohyb([as]+)(gdn[12])?(\d+)l(\d+)h(\d+)d([0-9.]+)dr"
    r"(ne|none)?(nope|pe)?(?:frz(?:a|ssm)[0-9.]+)?"
    rf"stp([0-9.]+)k([0-9eE+\-\.]+)lr{_WEIGHTS_SUFFIX}$"
)
# Versioned kernels first so "olmogdn22l..." is gdn2 with 2 layers, not gdn with 22.
_WEIGHTS_NAME_RE_OLMO_SSM = re.compile(
    r"^olmo(gdn[12])(\d+)l(\d+)d([0-9.]+)dr(ne|none)?"
    rf"stp([0-9.]+)k([0-9eE+\-\.]+)lr{_WEIGHTS_SUFFIX}$"
)
_WEIGHTS_NAME_RE_OLMO_SSM_LEGACY = re.compile(
    r"^olmogdn(\d+)l(\d+)d([0-9.]+)dr(ne|none)?"
    rf"stp([0-9.]+)k([0-9eE+\-\.]+)lr{_WEIGHTS_SUFFIX}$"
)
_WEIGHTS_NAME_RE_SSM = re.compile(
    r"^ssm(mamba3|mamba2|mamba|s4)(\d+)l(\d+)d([0-9.]+)dr"
    r"(?:(\d+)mlp)?(?:(noln|ln))?"
    rf"stp([0-9.]+)k([0-9eE+\-\.]+)lr{_WEIGHTS_SUFFIX}$"
)


def _neg_eigval(flag: str | None) -> bool | None:
    if flag == "ne":
        return True
    if flag == "none":
        return False
    return None


def parse_architecture_from_weights_path(path: str) -> dict:
    base = os.path.basename(path)
    m = _WEIGHTS_NAME_RE_OLMO_SSM.match(base)
    if m:
        gdn_variant, n_layer, nd, dr_s, neg_eig, _steps_k, _lr = m.groups()
        return {
            "model_family": "ssm",
            "layer_pattern": "s",
            "n_pattern_repeats": int(n_layer),
            "n_head": 1,
            "d_model": int(nd),
            "dropout": float(dr_s),
            "between_block_mlp_layers": 1,
            "layer_norm": True,
            "nope": None,
            "ssm_kernel": gdn_variant,
            "olmo_gdn_variant": gdn_variant,
            "olmo_gdn_allow_neg_eigval": _neg_eigval(neg_eig),
            "olmo": True,
        }
    m = _WEIGHTS_NAME_RE_OLMO_SSM_LEGACY.match(base)
    if m:
        n_layer, nd, dr_s, neg_eig, _steps_k, _lr = m.groups()
        return {
            "model_family": "ssm",
            "layer_pattern": "s",
            "n_pattern_repeats": int(n_layer),
            "n_head": 1,
            "d_model": int(nd),
            "dropout": float(dr_s),
            "between_block_mlp_layers": 1,
            "layer_norm": True,
            "nope": None,
            "ssm_kernel": "gdn1",
            "olmo_gdn_variant": "gdn1",
            "olmo_gdn_allow_neg_eigval": _neg_eigval(neg_eig),
            "olmo": True,
        }
    m = _WEIGHTS_NAME_RE_OLMO_HYB.match(base)
    if m:
        motif, gdn_variant, n_rep, nh, nd, dr_s, neg_eig, pe_mode, _steps_k, _lr = m.groups()
        return {
            "model_family": "hybrid",
            "layer_pattern": motif,
            "n_pattern_repeats": int(n_rep),
            "n_head": int(nh),
            "d_model": int(nd),
            "dropout": float(dr_s),
            "between_block_mlp_layers": 1,
            "layer_norm": True,
            "nope": None if pe_mode is None else pe_mode == "nope",
            "ssm_kernel": gdn_variant or "gdn1",
            "olmo_gdn_variant": gdn_variant or "gdn1",
            "olmo_gdn_allow_neg_eigval": _neg_eigval(neg_eig),
            "olmo": True,
        }
    m = _WEIGHTS_NAME_RE_MODULAR.match(base)
    if m:
        (
            layer_pattern,
            ssm_kernel,
            n_rep,
            nh,
            nd,
            dr_s,
            mlp_layers,
            pe_mode,
            ln_mode,
            _steps_k,
            _lr,
        ) = m.groups()
        return {
            "model_family": "hybrid",
            "layer_pattern": layer_pattern,
            "n_pattern_repeats": int(n_rep),
            "n_head": int(nh),
            "d_model": int(nd),
            "dropout": float(dr_s),
            "between_block_mlp_layers": int(mlp_layers),
            "layer_norm": ln_mode == "ln",
            "nope": pe_mode == "nope",
            "ssm_kernel": ssm_kernel,
            "olmo": False,
        }
    m = _WEIGHTS_NAME_RE_SSM.match(base)
    if m:
        ssm_kernel, n_layer, nd, dr_s, mlp_layers, ln_mode, _steps_k, _lr = m.groups()
        return {
            "model_family": "ssm",
            "layer_pattern": "s",
            "n_pattern_repeats": int(n_layer),
            "n_head": 1,
            "d_model": int(nd),
            "dropout": float(dr_s),
            "between_block_mlp_layers": int(mlp_layers) if mlp_layers is not None else 1,
            "layer_norm": ln_mode != "noln",
            "nope": None,
            "ssm_kernel": ssm_kernel,
            "olmo": False,
        }
    m = _WEIGHTS_NAME_RE_NEW.match(base)
    if m:
        motif, n_rep, nh, nd, dr_s, _smalllr = m.groups()
        return {
            "model_family": "hybrid",
            "layer_pattern": motif,
            "n_pattern_repeats": int(n_rep),
            "n_head": int(nh),
            "d_model": int(nd),
            "dropout": float(dr_s),
            "between_block_mlp_layers": 1,
            "layer_norm": True,
            "nope": None,
            "ssm_kernel": "s4",
            "olmo": False,
        }
    m = _WEIGHTS_NAME_RE_LEGACY.match(base)
    if m:
        variant, nl, nh, nd, dr_s, _smalllr = m.groups()
        return {
            "model_family": "hybrid",
            "layer_pattern": variant,
            "n_pattern_repeats": int(nl),
            "n_head": int(nh),
            "d_model": int(nd),
            "dropout": float(dr_s),
            "between_block_mlp_layers": 1,
            "layer_norm": True,
            "nope": None,
            "ssm_kernel": "s4",
            "olmo": False,
        }
    raise ValueError(
        f"Cannot parse architecture from filename {base!r}. "
        "Expected an OLMo SSM name (olmo<gdn1|gdn2><layers>l...), "
        "an OLMo hybrid name (olmohyb...), a modular hybrid name "
        "(hyb<pattern><kernel>...mlp<pe><ln>..._weights*.pt), "
        "the old underscore pattern (hyb<pattern>_<repeats>l..._weights*.pt), "
        "or legacy hyb<as|sa><repeats>l..._weights*.pt. "
        "Otherwise pass explicit architecture flags."
    )


def _parse_length_range(value: str) -> tuple[int, int]:
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("train length range must be 'min,max'")
    try:
        start, end = int(parts[0]), int(parts[1])
    except ValueError as exc:
        raise argparse.ArgumentTypeError("train length range values must be integers") from exc
    return (start, end)


def infer_max_test_length_from_state(task: str, state: dict, fallback: int) -> int:
    """
    Infer dataset max_test_length from checkpoint positional embedding size.
    This keeps n_positions consistent with training.
    OLMo checkpoints have no GPT-2 ``wpe``; callers should keep the training default.
    For sort, content vocabulary is independent of max_test_length when --sort-vocab-size is set.
    """
    wpe = state.get("wpe.weight")
    if wpe is None:
        return fallback
    n_positions = int(wpe.shape[0])
    if task in {"bin_majority", "majority", "bin_majority_interleave", "parity", "addition"}:
        return n_positions - 4
    if task in {"unique_copy", "repeat_copy", "sort"}:
        return (n_positions - 3) // 2
    if task in {"mqar", "selective_state_tracking"}:
        # bos + content (pairs + query) + sep + sep + answer + eos
        return n_positions - 5
    return fallback


def tokens_string_to_ids(tokenizer, text: str) -> list[int]:
    parts = text.split()
    out = []
    for p in parts:
        if p not in tokenizer.vocab:
            raise ValueError(f"Unknown token {p!r} for this task's vocabulary.")
        out.append(tokenizer.vocab[p])
    return out


def _answer_positions(label: list[int], pad_id: int) -> list[int]:
    return [i for i in range(1, len(label)) if label[i] != pad_id]


def _result_from_ids(
    tokenizer,
    instance: list[int],
    pred_answer: list[int],
    gold_answer: list[int],
    same_pos_pred_ids: list[int],
    shifted_pred_ids: list[int],
    same_pos_gold_ids: list[int],
    correct: bool,
    same_pos_acc: float,
    shifted_acc: float,
) -> dict:
    gold_str = " ".join(tokenizer.convert_ids_to_tokens(gold_answer, rm_special=True))
    pred_str = " ".join(tokenizer.convert_ids_to_tokens(pred_answer, rm_special=True))
    return {
        "sequence_tokens": tokenizer.convert_ids_to_tokens(instance, rm_special=False),
        "gold_answer_tokens": tokenizer.convert_ids_to_tokens(gold_answer, rm_special=True),
        "pred_answer_tokens": tokenizer.convert_ids_to_tokens(pred_answer, rm_special=True),
        "gold_answer_str": gold_str,
        "pred_answer_str": pred_str,
        "correct": correct,
        "same_pos_acc": same_pos_acc,
        "shifted_acc": shifted_acc,
        "same_pos_pred_tokens": tokenizer.convert_ids_to_tokens(same_pos_pred_ids, rm_special=False),
        "shifted_pred_tokens": tokenizer.convert_ids_to_tokens(shifted_pred_ids, rm_special=False),
        "target_tokens": tokenizer.convert_ids_to_tokens(same_pos_gold_ids, rm_special=False),
    }


def infer_one_sample(
    model: nn.Module,
    tokenizer,
    instance: list[int],
    pos_ids: list[int],
    label: list[int],
    device: torch.device,
):
    pad_id = tokenizer.pad_token_id
    answer_indices = _answer_positions(label, pad_id)
    gold_answer = [instance[i] for i in answer_indices]

    if answer_indices:
        prefix_len = answer_indices[0]
    else:
        prefix_len = len(instance)

    prompt_ids = list(instance[:prefix_len])
    prompt_pos_ids = list(pos_ids[:prefix_len])
    pred_answer = []

    model.eval()
    with torch.no_grad():
        # Teacher-forced forward pass on the full sequence for alignment diagnostics.
        full_input_ids = torch.tensor([instance], dtype=torch.long, device=device)
        full_position_ids = torch.tensor([pos_ids], dtype=torch.long, device=device)
        full_logits = model(full_input_ids, position_ids=full_position_ids).logits[0]

        for _ in range(len(answer_indices)):
            input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
            position_ids = torch.tensor([prompt_pos_ids], dtype=torch.long, device=device)
            logits = model(input_ids, position_ids=position_ids).logits
            next_token = logits[0, -1, :].argmax(dim=-1).item()
            pred_answer.append(next_token)
            prompt_ids.append(next_token)
            if prompt_pos_ids:
                prompt_pos_ids.append(prompt_pos_ids[-1] + 1)
            else:
                prompt_pos_ids.append(0)

    correct = pred_answer == gold_answer

    target_positions = [i for i in range(len(label)) if label[i] != pad_id]
    same_pos_pred_ids = [int(full_logits[i].argmax(dim=-1).item()) for i in target_positions]
    same_pos_gold_ids = [instance[i] for i in target_positions]
    same_pos_acc = (
        sum(int(p == g) for p, g in zip(same_pos_pred_ids, same_pos_gold_ids)) / len(target_positions)
        if target_positions else 0.0
    )

    shifted_positions = [i for i in target_positions if i > 0]
    shifted_pred_ids = [int(full_logits[i - 1].argmax(dim=-1).item()) for i in shifted_positions]
    shifted_gold_ids = [instance[i] for i in shifted_positions]
    shifted_acc = (
        sum(int(p == g) for p, g in zip(shifted_pred_ids, shifted_gold_ids)) / len(shifted_positions)
        if shifted_positions else 0.0
    )

    return _result_from_ids(
        tokenizer,
        instance,
        pred_answer,
        gold_answer,
        same_pos_pred_ids,
        shifted_pred_ids,
        same_pos_gold_ids,
        correct,
        same_pos_acc,
        shifted_acc,
    )


def collate_eval_batch(examples: list[tuple[list[int], list[int], list[int]]], pad_id: int) -> dict:
    """Pad a batch the same way training's ``customCollator`` does, without mutating inputs."""
    input_ids = [list(example[0]) for example in examples]
    pos_ids = [list(example[1]) for example in examples]
    labels = [list(example[2]) for example in examples]
    max_len = max(len(item) for item in input_ids)
    input_ids = torch.tensor([item + [pad_id] * (max_len - len(item)) for item in input_ids], dtype=torch.long)
    labels_t = torch.tensor([item + [pad_id] * (max_len - len(item)) for item in labels], dtype=torch.long)
    labels_t[labels_t == pad_id] = -100
    pos_ids_t = torch.tensor(
        [item + [item[-1]] * (max_len - len(item)) for item in pos_ids],
        dtype=torch.long,
    )
    return {"input_ids": input_ids, "position_ids": pos_ids_t, "labels": labels_t}


def teacher_forced_eval_examples(
    model: nn.Module,
    tokenizer,
    examples: list[tuple[list[int], list[int], list[int]]],
    device: torch.device,
    batch_size: int,
) -> list[dict]:
    """Score examples with the same shifted exact-match rule as training ``compute_metrics``."""
    pad_id = tokenizer.pad_token_id
    model.eval()
    results: list[dict] = []
    with torch.no_grad():
        for start in range(0, len(examples), batch_size):
            batch_examples = examples[start : start + batch_size]
            batch = collate_eval_batch(batch_examples, pad_id)
            logits = model(
                input_ids=batch["input_ids"].to(device),
                position_ids=batch["position_ids"].to(device),
            ).logits
            shift_logits = logits[:, :-1]
            shift_labels = batch["labels"][:, 1:].to(device)
            pred = shift_logits.argmax(dim=-1)
            row_correct = ((pred == shift_labels) | (shift_labels == -100)).all(dim=1)

            for i, (instance, _pos_ids, label) in enumerate(batch_examples):
                answer_indices = _answer_positions(label, pad_id)
                gold_answer = [instance[j] for j in answer_indices]
                pred_answer = [int(pred[i, j - 1].item()) for j in answer_indices]
                target_positions = [j for j in range(len(label)) if label[j] != pad_id]
                same_pos_gold_ids = [instance[j] for j in target_positions]
                same_pos_pred_ids = [int(logits[i, j].argmax(dim=-1).item()) for j in target_positions]
                shifted_positions = [j for j in target_positions if j > 0]
                shifted_pred_ids = [int(pred[i, j - 1].item()) for j in shifted_positions]
                shifted_gold_ids = [instance[j] for j in shifted_positions]
                same_pos_acc = (
                    sum(int(p == g) for p, g in zip(same_pos_pred_ids, same_pos_gold_ids)) / len(target_positions)
                    if target_positions else 0.0
                )
                shifted_acc = (
                    sum(int(p == g) for p, g in zip(shifted_pred_ids, shifted_gold_ids)) / len(shifted_positions)
                    if shifted_positions else 0.0
                )
                results.append(
                    _result_from_ids(
                        tokenizer,
                        instance,
                        pred_answer,
                        gold_answer,
                        same_pos_pred_ids,
                        shifted_pred_ids,
                        same_pos_gold_ids,
                        bool(row_correct[i].item()),
                        same_pos_acc,
                        shifted_acc,
                    )
                )
    return results


def select_eval_bins(test_dataset: dict, eval_bins: str) -> list[str]:
    keys = list(test_dataset)
    if not keys:
        raise ValueError("No eval bins were materialised.")
    requested = eval_bins.strip().lower()
    if requested == "all":
        return keys
    if requested in {"train", "in-dist", "id"}:
        return [keys[0]]
    chosen = [part.strip() for part in eval_bins.split(",") if part.strip()]
    missing = [key for key in chosen if key not in test_dataset]
    if missing:
        raise ValueError(
            f"Unknown eval bin(s) {missing}. Available: {keys}. "
            "Use 'train' for the in-distribution bin or 'all'."
        )
    return chosen


def _print_sample(index: int, out: dict) -> None:
    status = "CORRECT" if out["correct"] else "WRONG"
    print(f"[{index}] {status}")
    print("  Full sequence:", " ".join(out["sequence_tokens"]))
    print("  Gold answer:  ", " ".join(out["gold_answer_tokens"]), f"({out['gold_answer_str']!r})")
    print("  Pred answer:  ", " ".join(out["pred_answer_tokens"]), f"({out['pred_answer_str']!r})")
    print("  Teacher-forced shifted acc on supervised targets:", out["shifted_acc"])
    print("  Answer correct:", out["correct"])


def print_successes_and_failures(bin_name: str, results: list[dict]) -> None:
    failures = [(i, out) for i, out in enumerate(results) if not out["correct"]]
    successes = [(i, out) for i, out in enumerate(results) if out["correct"]]
    n = len(results)
    n_ok = len(successes)
    print(f"=== {bin_name}: {n_ok}/{n} correct (accuracy {n_ok / n if n else 0.0:.6f}) ===")
    print(f"--- failures ({len(failures)}) ---")
    if not failures:
        print("(none)")
    for index, out in failures:
        _print_sample(index, out)
    print(f"--- successes ({len(successes)}) ---")
    if not successes:
        print("(none)")
    for index, out in successes:
        _print_sample(index, out)


def _jsonable_result(out: dict) -> dict:
    return {
        "sequence": " ".join(out["sequence_tokens"]),
        "gold_answer": out["gold_answer_str"],
        "pred_answer": out["pred_answer_str"],
        "correct": out["correct"],
        "shifted_acc": out["shifted_acc"],
    }


def build_eval_run_config(args: argparse.Namespace, arch: dict, nope: bool, use_olmo: bool) -> RunConfig:
    olmo_gdn_variant = args.olmo_gdn_variant or arch.get("olmo_gdn_variant", "gdn1")
    allow_neg = arch.get("olmo_gdn_allow_neg_eigval")
    if allow_neg is None:
        allow_neg = True
    rc = RunConfig(
        model_family=arch.get("model_family", args.model_family),
        architectures=[],
        task=args.task,
        dataset_seed=args.dataset_seed,
        train_length_range=args.train_length_range,
        test_num=args.test_num,
        use_nope=nope,
        use_olmo_core=use_olmo,
        olmo_gdn_variant=olmo_gdn_variant,
        olmo_gdn_allow_neg_eigval=allow_neg,
        hybrid_layer_pattern=arch["layer_pattern"],
        ssm_kernel="s4" if use_olmo else arch["ssm_kernel"],
        formal_aligned_targets=not args.formal_packed_targets,
        monoid=args.monoid,
        monoid_n=args.monoid_n,
        query_fraction_upper=args.query_fraction,
        query_fraction_lower=args.query_fraction,
    )
    # These are class attributes on RunConfig, not dataclass fields.
    rc.key_len = args.key_len
    rc.mkar_vocab_size = args.mkar_vocab_size
    rc.marker_vocab_size = args.marker_vocab_size
    rc.marker_frequency = args.marker_frequency
    rc.sort_vocab_size = args.sort_vocab_size
    return rc


def main():
    parser = argparse.ArgumentParser(description="Inference for hybrid/SSM/OLMo LMs.")
    parser.add_argument("--weights", type=str, required=True, help="Path to *_weights.pt from training.")
    parser.add_argument("--task", type=str, choices=TASK_CHOICES, required=True)
    parser.add_argument("--seed", type=int, default=0, help="RNG seed when drawing a random sample (default: 0).")
    parser.add_argument(
        "--dataset-seed",
        type=int,
        default=42,
        help=(
            "Seed used to materialize eval bins. Pass the training run's "
            "dataset_seed (recorded in the JSON run record). Default: 42."
        ),
    )
    parser.add_argument(
        "--test-num",
        type=int,
        default=2000,
        help="Examples per eval length bin, matching training RunConfig.test_num (default: 2000).",
    )
    parser.add_argument(
        "--train-length-range",
        type=_parse_length_range,
        default=TRAIN_LENGTH_RANGE,
        help="Comma-separated min,max for the training length window (default: 0,50).",
    )
    parser.add_argument(
        "--eval-full",
        action="store_true",
        help=(
            "Score the full training eval set (same dataset seed and sizes as training) "
            "instead of drawing --num-samples random examples."
        ),
    )
    parser.add_argument(
        "--eval-bins",
        type=str,
        default="train",
        help=(
            "Which eval bins to score with --eval-full: 'train' (in-distribution / training-length "
            "bin, default), 'all', or a comma-separated list of keys such as len0-50,len51-100."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for --eval-full teacher-forced scoring (default: 64).",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Optional path to write successes/failures as JSON.",
    )
    parser.add_argument(
        "--tokens",
        type=str,
        default=None,
        help="Optional: whitespace-separated token strings (vocabulary keys, e.g. '<bos>' '0' '1' '<sep>' '0' '<eos>'). "
        "If omitted, random examples are sampled from the task generator unless --eval-full is set.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1,
        help="Number of random task samples to generate when --tokens and --eval-full are omitted (default: 1).",
    )
    parser.add_argument(
        "--nope",
        action="store_true",
        help="Force no positional embeddings for explicit/legacy configs. Modular filenames infer this automatically.",
    )
    parser.add_argument(
        "--use-olmo",
        "--use_olmo",
        action="store_true",
        help=(
            "Load the checkpoint with OLMo-core hybrid/SSM blocks. "
            "When enabled, --between-block-mlp-layers and --no-layer-norm are ignored "
            "(OLMo uses one FFN block and always keeps layer norm), and OLMo positional "
            "encoding is used instead of GPT-2 absolute embeddings."
        ),
    )
    parser.add_argument(
        "--olmo-gdn-variant",
        choices=["gdn1", "gdn2"],
        default=None,
        help=(
            "Override the OLMo GDN variant. New checkpoint filenames encode this; "
            "legacy OLMo checkpoints default to gdn1."
        ),
    )
    parser.add_argument("--model-family", type=str, default=None, choices=["hybrid", "transformer", "ssm"])
    parser.add_argument("--n-layer", type=int, default=None)
    parser.add_argument("--n-head", type=int, default=None)
    parser.add_argument("--d-model", type=int, default=None)
    parser.add_argument("--between-block-mlp-layers", type=int, default=1)
    parser.add_argument("--layer-norm", action="store_true", help="Enable layer norm when using explicit architecture flags (default).")
    parser.add_argument("--no-layer-norm", action="store_true", help="Disable layer norm when using explicit architecture flags.")
    parser.add_argument("--ssm-kernel", type=str, default="s4")
    parser.add_argument(
        "--start-with-attention",
        action="store_true",
        help="Only when passing explicit --n-layer/--n-head/--d-model: use alternating pattern starting with attention ('as').",
    )
    parser.add_argument(
        "--layer-pattern",
        type=str,
        default=None,
        help="When passing explicit layer dims: motif of 'a' (attention) and 's' (SSM) repeated --n-layer times (default: 'sa' or 'as' from --start-with-attention).",
    )
    parser.add_argument(
        "--monoid",
        type=str,
        default="parity",
        choices=["parity", "cyclic", "s5"],
        help="Monoid for MQAR and selective_state_tracking: parity (Z_2 XOR), cyclic (Z_n addition, MQAR only), or s5 (S_5 composition).",
    )
    parser.add_argument("--monoid_n", type=int, default=2)
    parser.add_argument("--query_fraction", type=float, default=0.2)
    parser.add_argument("--key-len", type=int, default=4)
    parser.add_argument("--mkar-vocab-size", type=int, default=128)
    parser.add_argument("--marker-vocab-size", type=int, default=16)
    parser.add_argument(
        "--marker-frequency",
        type=float,
        default=0.2,
        help=(
            "Selective copy only: lower bound on the fraction of content tokens "
            "that are numbered markers. Must match training. Per example the "
            "frequency is drawn uniformly from [this value, 1]. The count is "
            "ceil(length * frequency), including the last marker, clamped to "
            "[1, length]. Must be in [0, 1]."
        ),
    )
    parser.add_argument(
        "--sort-vocab-size",
        type=int,
        default=None,
        help=(
            "Sort task only: number of distinct content tokens. Must match training. "
            "If omitted, the vocabulary has max_test_length tokens (the current default). "
            "Raised to the maximum sequence length if smaller, so every example uses unique tokens."
        ),
    )
    parser.add_argument(
        "--formal-packed-targets",
        action="store_true",
        help="Formal-language tasks only: use packed '<bos> src <sep> tgt <eos>' instead of aligned targets.",
    )
    args = parser.parse_args()

    explicit = (
        args.n_layer is not None,
        args.n_head is not None,
        args.d_model is not None,
    )
    head_only_override = args.n_head is not None and args.n_layer is None and args.d_model is None
    if any(explicit) and not all(explicit) and not head_only_override:
        parser.error(
            "If you pass --n-layer or --d-model, pass --n-layer, --n-head, and --d-model. "
            "--n-head alone is allowed as an override for parsed SSM filenames, which omit heads."
        )
    if args.layer_norm and args.no_layer_norm:
        parser.error("Pass at most one of --layer-norm or --no-layer-norm.")
    if args.num_samples < 1:
        parser.error("--num-samples must be at least 1.")
    if args.test_num < 1:
        parser.error("--test-num must be at least 1.")
    if args.sort_vocab_size is not None and args.sort_vocab_size < 1:
        parser.error("--sort-vocab-size must be at least 1.")
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1.")
    if args.tokens is not None and args.num_samples != 1:
        parser.error("--num-samples can only be used when --tokens is omitted.")
    if args.eval_full and args.tokens is not None:
        parser.error("--eval-full cannot be combined with --tokens.")

    if all(explicit):
        if args.layer_pattern is not None:
            motif = args.layer_pattern.strip().lower()
        else:
            motif = "as" if args.start_with_attention else "sa"
        layer_norm = not args.no_layer_norm
        arch = {
            "model_family": args.model_family or "hybrid",
            "layer_pattern": motif,
            "n_pattern_repeats": args.n_layer,
            "n_head": args.n_head,
            "d_model": args.d_model,
            "dropout": 0.0,
            "between_block_mlp_layers": args.between_block_mlp_layers,
            "layer_norm": layer_norm,
            "nope": args.nope,
            "ssm_kernel": args.ssm_kernel,
            "olmo": args.use_olmo,
        }
    else:
        try:
            arch = parse_architecture_from_weights_path(args.weights)
        except ValueError as e:
            raise SystemExit(str(e)) from e
        if head_only_override:
            arch["n_head"] = args.n_head

    # Legacy filename formats do not encode NoPE; use explicit flag there.
    nope = arch["nope"] if arch["nope"] is not None else args.nope
    if arch["nope"] is not None and arch["nope"] != args.nope:
        print(f"[info] Ignoring --nope={args.nope}: filename implies nope={arch['nope']}.")
    inferred_olmo = bool(arch.get("olmo", False))
    use_olmo = args.use_olmo or inferred_olmo
    if inferred_olmo and not args.use_olmo:
        print("[info] Inferred OLMo checkpoint from filename; enabling --use-olmo.")
    olmo_gdn_variant = args.olmo_gdn_variant or arch.get("olmo_gdn_variant", "gdn1")
    model_family = args.model_family or arch.get("model_family", "hybrid")
    if args.model_family is None and arch.get("model_family"):
        print(f"[info] Inferred model-family={model_family} from filename.")

    state = torch.load(args.weights, map_location="cpu")
    eval_run_config = build_eval_run_config(args, arch, nope, use_olmo)
    eval_run_config.model_family = model_family
    fallback_max_test = eval_run_config.test_length_ranges[-1][1]
    max_test_length = infer_max_test_length_from_state(args.task, state, fallback_max_test)
    if max_test_length != fallback_max_test:
        print(
            f"[info] Inferred max_test_length={max_test_length} from checkpoint for task={args.task} "
            f"(default was {fallback_max_test}). "
            "Eval bins still follow --train-length-range and RunConfig.num_test_bins so the "
            "dataset matches training."
        )

    set_seed(args.dataset_seed)
    train_dataset, test_dataset, train_length_range, test_length_ranges = build_datasets(eval_run_config)
    tokenizer = train_dataset.tokenizer
    print(
        f"[info] Materialized eval bins with dataset_seed={args.dataset_seed}, "
        f"test_num={args.test_num}, train_length_range={train_length_range}, "
        f"test_length_ranges={test_length_ranges}."
    )
    for key, ds in test_dataset.items():
        print(f"[info]   {key}: {len(ds)} examples")

    if args.tokens is not None:
        instance = tokens_string_to_ids(tokenizer, args.tokens)
        start_of_inference = instance.index(tokenizer.sep_token_id)
        label = copy.deepcopy(instance)
        label[:start_of_inference + 1] = [tokenizer.pad_token_id] * (start_of_inference + 1)
        pos_ids = train_dataset.get_pos_ids(len(instance), max(0, train_dataset.n_positions - len(instance)))
        eval_examples = None
    elif args.eval_full:
        try:
            bin_keys = select_eval_bins(test_dataset, args.eval_bins)
        except ValueError as e:
            raise SystemExit(str(e)) from e
        eval_examples = {key: list(test_dataset[key].data) for key in bin_keys}
        instance = pos_ids = label = None
    else:
        # Random samples from the training-length generator, seeded independently of dataset_seed.
        set_seed(args.seed)
        from task_datasets import EvalDataset

        first_bin = next(iter(test_dataset.values()))
        eval_ds = EvalDataset(first_bin.source_dataset, args.num_samples)
        eval_examples = {"random": list(eval_ds.data)}
        instance = pos_ids = label = None

    layer_norm = arch["layer_norm"]
    if use_olmo and not layer_norm:
        print("[info] Ignoring no-layer-norm setting for OLMo; layer norm is always enabled.")
        layer_norm = True
    if use_olmo and arch["between_block_mlp_layers"] != 1:
        print(
            "[info] Ignoring --between-block-mlp-layers for OLMo; "
            "OLMo uses one FFN block per transformer block."
        )

    arch_slot = ArchSlot(
        n_layer=arch["n_pattern_repeats"],
        n_head=arch["n_head"],
        d_model=arch["d_model"],
        between_block_mlp_layers=arch["between_block_mlp_layers"],
        layer_norm=layer_norm,
        dropout=arch.get("dropout", 0.0),
        lr=1e-3,
    )
    allow_neg = arch.get("olmo_gdn_allow_neg_eigval")
    if allow_neg is None:
        allow_neg = True
    run_config = RunConfig(
        model_family=model_family,
        architectures=[arch_slot],
        use_nope=nope,
        use_olmo_core=use_olmo,
        olmo_gdn_variant=olmo_gdn_variant,
        olmo_gdn_allow_neg_eigval=allow_neg,
        hybrid_layer_pattern=arch["layer_pattern"],
        # The OLMo hybrid/SSM path maps SSM layers to the selected GatedDeltaNet variant.
        ssm_kernel="s4" if use_olmo else arch["ssm_kernel"],
    )
    from models import build_model

    model = build_model(run_config, arch_slot, tokenizer, train_dataset.n_positions, seed=args.seed)
    model.load_state_dict(state, strict=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    print("Task:", args.task)
    if args.tokens is not None:
        out = infer_one_sample(model, tokenizer, instance, pos_ids, label, device)
        _print_sample(0, out)
        return

    report = {
        "task": args.task,
        "weights": args.weights,
        "dataset_seed": args.dataset_seed,
        "test_num": args.test_num,
        "train_length_range": list(train_length_range),
        "test_length_ranges": [list(r) for r in test_length_ranges],
        "bins": {},
    }
    for bin_name, examples in eval_examples.items():
        if args.eval_full:
            results = teacher_forced_eval_examples(model, tokenizer, examples, device, args.batch_size)
        else:
            results = [
                infer_one_sample(model, tokenizer, inst, pids, lab, device)
                for inst, pids, lab in examples
            ]
        print_successes_and_failures(bin_name, results)
        failures = [out for out in results if not out["correct"]]
        successes = [out for out in results if out["correct"]]
        report["bins"][bin_name] = {
            "num_samples": len(results),
            "num_correct": len(successes),
            "num_wrong": len(failures),
            "accuracy": (len(successes) / len(results)) if results else 0.0,
            "failures": [_jsonable_result(out) for out in failures],
            "successes": [_jsonable_result(out) for out in successes],
        }

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(report, f, indent=2)
        print(f"[info] Wrote {args.output_json}")


if __name__ == "__main__":
    main()
