"""
Load a trained HybridGPT2S4 checkpoint and run inference on one or more task samples.

Architecture is read from the weights filename, supporting both legacy and newer
modular names from language_modeling_train.py.
If the filename does not match known conventions, pass explicit architecture flags.
"""

from __future__ import annotations

import argparse
import copy
import os
import random
import re
import torch

from dataset_generators import (
    AdditionDataset,
    BinaryMajorityDataset,
    BinaryMajorityInterleaveDataset,
    MajorityDataset,
    MQARWordProblemDataset,
    ParityDataset,
    RepeatCopyDataset,
    SortDataset,
    UniqueCopyDataset,
)
from dataset_generators import EvalDataset
from models import HybridConfig, HybridGPT2S4LMHeadModel

TASK_CHOICES = [
    "bin_majority",
    "majority",
    "bin_majority_interleave",
    "unique_copy",
    "repeat_copy",
    "sort",
    "parity",
    "addition",
    "mqar",
]

# Default from current train setup. Can be auto-adjusted from checkpoint shape.
TRAIN_LENGTH_RANGE = (0, 50)
MAX_TEST_LENGTH = 150


_WEIGHTS_NAME_RE_NEW = re.compile(
    r"^hyb([as]+)_(\d+)l(\d+)h(\d+)d([0-9.]+)dr(smalllr)?_weights(?:_seed\d+_id\d+)?\.pt$"
)
_WEIGHTS_NAME_RE_LEGACY = re.compile(
    r"^hyb(as|sa)(\d+)l(\d+)h(\d+)d([0-9.]+)dr(smalllr)?_weights(?:_seed\d+_id\d+)?\.pt$"
)
_WEIGHTS_NAME_RE_MODULAR = re.compile(
    r"^hyb(.+?)(\d+)l(\d+)h(\d+)d([0-9.]+)dr(\d+)mlp(nope|pe)(noln|ln)"
    r"stp([0-9.]+)k([0-9eE+\-\.]+)lr_weights(?:_seed\d+_id\d+)?\.pt$"
)


def parse_architecture_from_weights_path(path: str) -> dict:
    base = os.path.basename(path)
    m = _WEIGHTS_NAME_RE_MODULAR.match(base)
    if m:
        pattern_and_kernel, n_rep, nh, nd, _dr_s, mlp_layers, pe_mode, ln_mode, _steps_k, _lr = m.groups()
        i = 0
        while i < len(pattern_and_kernel) and pattern_and_kernel[i] in "as":
            i += 1
        layer_pattern = pattern_and_kernel[:i]
        ssm_kernel = pattern_and_kernel[i:] if i < len(pattern_and_kernel) else "s4"
        if not layer_pattern:
            raise ValueError(
                f"Could not parse layer_pattern from modular filename {base!r}. "
                "Expected a prefix like hybsa... or hybas..."
            )
        return {
            "layer_pattern": layer_pattern,
            "n_pattern_repeats": int(n_rep),
            "n_head": int(nh),
            "d_model": int(nd),
            "between_block_mlp_layers": int(mlp_layers),
            "layer_norm": ln_mode == "ln",
            "nope": pe_mode == "nope",
            "ssm_kernel": ssm_kernel,
        }
    m = _WEIGHTS_NAME_RE_NEW.match(base)
    if m:
        motif, n_rep, nh, nd, _dr_s, _smalllr = m.groups()
        return {
            "layer_pattern": motif,
            "n_pattern_repeats": int(n_rep),
            "n_head": int(nh),
            "d_model": int(nd),
            "between_block_mlp_layers": 1,
            "layer_norm": True,
            "nope": None,
            "ssm_kernel": "s4",
        }
    m = _WEIGHTS_NAME_RE_LEGACY.match(base)
    if m:
        variant, nl, nh, nd, _dr_s, _smalllr = m.groups()
        return {
            "layer_pattern": variant,
            "n_pattern_repeats": int(nl),
            "n_head": int(nh),
            "d_model": int(nd),
            "between_block_mlp_layers": 1,
            "layer_norm": True,
            "nope": None,
            "ssm_kernel": "s4",
        }
    raise ValueError(
        f"Cannot parse hybrid architecture from filename {base!r}. "
        "Expected a modular name (hyb<pattern><kernel>...mlp<pe><ln>..._weights*.pt), "
        "the old underscore pattern (hyb<pattern>_<repeats>l..._weights*.pt), "
        "or legacy hyb<as|sa><repeats>l..._weights*.pt. "
        "Otherwise pass explicit architecture flags."
    )


def build_task_dataset(
    task: str,
    max_test_length: int,
    monoid: str,
    monoid_n: int,
    key_size: int,
    query_fraction: float,
):
    match task:
        case "bin_majority":
            ds = BinaryMajorityDataset(TRAIN_LENGTH_RANGE, max_test_length, add_positional_offset=False)
        case "majority":
            ds = MajorityDataset(TRAIN_LENGTH_RANGE, max_test_length, add_positional_offset=False)
        case "bin_majority_interleave":
            ds = BinaryMajorityInterleaveDataset(TRAIN_LENGTH_RANGE, max_test_length, period=3, add_positional_offset=False)
        case "unique_copy":
            ds = UniqueCopyDataset(TRAIN_LENGTH_RANGE, max_test_length, add_positional_offset=False)
        case "repeat_copy":
            ds = RepeatCopyDataset(TRAIN_LENGTH_RANGE, max_test_length, add_positional_offset=False)
        case "sort":
            ds = SortDataset(TRAIN_LENGTH_RANGE, max_test_length, add_positional_offset=False)
        case "parity":
            ds = ParityDataset(TRAIN_LENGTH_RANGE, max_test_length, add_positional_offset=False)
        case "addition":
            ds = AdditionDataset(TRAIN_LENGTH_RANGE, max_test_length, add_positional_offset=False)
        case "mqar":
            ds = MQARWordProblemDataset(
                TRAIN_LENGTH_RANGE,
                max_test_length,
                add_positional_offset=False,
                key_size=key_size,
                query_fraction=query_fraction,
                monoid_type=monoid,
                monoid_n=monoid_n,
            )
        case _:
            raise ValueError(f"unknown task {task}")
    return ds


def infer_max_test_length_from_state(task: str, state: dict, fallback: int) -> int:
    """
    Infer dataset max_test_length from checkpoint positional embedding size.
    This keeps n_positions (and task vocab for sort) consistent with training.
    """
    wpe = state.get("wpe.weight")
    if wpe is None:
        return fallback
    n_positions = int(wpe.shape[0])
    if task in {"bin_majority", "majority", "bin_majority_interleave", "parity", "addition"}:
        return n_positions - 4
    if task in {"unique_copy", "repeat_copy", "sort"}:
        return (n_positions - 3) // 2
    # MQAR n_positions depends on derived (T, Q), not a simple inverse.
    return fallback


def tokens_string_to_ids(tokenizer, text: str) -> list[int]:
    parts = text.split()
    out = []
    for p in parts:
        if p not in tokenizer.vocab:
            raise ValueError(f"Unknown token {p!r} for this task's vocabulary.")
        out.append(tokenizer.vocab[p])
    return out


def infer_one_sample(
    model: HybridGPT2S4LMHeadModel,
    tokenizer,
    instance: list[int],
    pos_ids: list[int],
    label: list[int],
    device: torch.device,
):
    pad_id = tokenizer.pad_token_id
    answer_indices = [i for i in range(1, len(instance)) if label[i] != pad_id]
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

    seq_tokens = tokenizer.convert_ids_to_tokens(instance, rm_special=False)
    gold_str = " ".join(tokenizer.convert_ids_to_tokens(gold_answer, rm_special=True))
    pred_str = " ".join(tokenizer.convert_ids_to_tokens(pred_answer, rm_special=True))

    return {
        "sequence_tokens": seq_tokens,
        "gold_answer_tokens": tokenizer.convert_ids_to_tokens(gold_answer, rm_special=True),
        "pred_answer_tokens": tokenizer.convert_ids_to_tokens(pred_answer, rm_special=True),
        "gold_answer_str": gold_str,
        "pred_answer_str": pred_str,
        "correct": correct,
    }


def main():
    parser = argparse.ArgumentParser(description="Inference for hybrid SSM/attention LM.")
    parser.add_argument("--weights", type=str, required=True, help="Path to *_weights.pt from training.")
    parser.add_argument("--task", type=str, choices=TASK_CHOICES, required=True)
    parser.add_argument("--seed", type=int, default=0, help="RNG seed when drawing a random sample (default: 0).")
    parser.add_argument(
        "--tokens",
        type=str,
        default=None,
        help="Optional: whitespace-separated token strings (vocabulary keys, e.g. '<bos>' '0' '1' '<sep>' '0' '<eos>'). "
        "If omitted, random examples are sampled from the task generator.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1,
        help="Number of random task samples to generate when --tokens is omitted (default: 1).",
    )
    parser.add_argument(
        "--nope",
        action="store_true",
        help="Force no positional embeddings for explicit/legacy configs. Modular filenames infer this automatically.",
    )
    parser.add_argument("--monoid", type=str, default="parity", choices=["parity", "cyclic"])
    parser.add_argument("--monoid_n", type=int, default=2)
    parser.add_argument("--key_size", type=int, default=32)
    parser.add_argument("--query_fraction", type=float, default=0.2)
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
    args = parser.parse_args()

    explicit = (
        args.n_layer is not None,
        args.n_head is not None,
        args.d_model is not None,
    )
    if any(explicit) and not all(explicit):
        parser.error("If you pass any of --n-layer, --n-head, --d-model, pass all three.")
    if args.layer_norm and args.no_layer_norm:
        parser.error("Pass at most one of --layer-norm or --no-layer-norm.")
    if args.num_samples < 1:
        parser.error("--num-samples must be at least 1.")
    if args.tokens is not None and args.num_samples != 1:
        parser.error("--num-samples can only be used when --tokens is omitted.")

    if all(explicit):
        if args.layer_pattern is not None:
            motif = args.layer_pattern.strip().lower()
        else:
            motif = "as" if args.start_with_attention else "sa"
        layer_norm = not args.no_layer_norm
        arch = {
            "layer_pattern": motif,
            "n_pattern_repeats": args.n_layer,
            "n_head": args.n_head,
            "d_model": args.d_model,
            "between_block_mlp_layers": args.between_block_mlp_layers,
            "layer_norm": layer_norm,
            "nope": args.nope,
            "ssm_kernel": args.ssm_kernel,
        }
    else:
        try:
            arch = parse_architecture_from_weights_path(args.weights)
        except ValueError as e:
            raise SystemExit(str(e)) from e

    # Legacy filename formats do not encode NoPE; use explicit flag there.
    nope = arch["nope"] if arch["nope"] is not None else args.nope
    if arch["nope"] is not None and arch["nope"] != args.nope:
        print(f"[info] Ignoring --nope={args.nope}: filename implies nope={arch['nope']}.")

    state = torch.load(args.weights, map_location="cpu")
    max_test_length = infer_max_test_length_from_state(args.task, state, MAX_TEST_LENGTH)
    if max_test_length != MAX_TEST_LENGTH:
        print(
            f"[info] Inferred max_test_length={max_test_length} from checkpoint for task={args.task} "
            f"(default was {MAX_TEST_LENGTH})."
        )

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    train_ds = build_task_dataset(
        args.task,
        max_test_length,
        args.monoid,
        args.monoid_n,
        args.key_size,
        args.query_fraction,
    )
    tokenizer = train_ds.tokenizer

    if args.tokens is not None:
        instance = tokens_string_to_ids(tokenizer, args.tokens)
        start_of_inference = instance.index(tokenizer.sep_token_id)
        label = copy.deepcopy(instance)
        label[:start_of_inference + 1] = [tokenizer.pad_token_id] * (start_of_inference + 1)
        pos_ids = train_ds.get_pos_ids(len(instance), max(0, train_ds.n_positions - len(instance)))
    else:
        eval_ds = EvalDataset(train_ds, args.num_samples)

    cfg = HybridConfig(
        vocab_size=len(tokenizer),
        n_positions=train_ds.n_positions,
        n_embd=arch["d_model"],
        n_head=arch["n_head"],
        dropout=0.0,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        nope=nope,
        n_pattern_repeats=arch["n_pattern_repeats"],
        layer_pattern=arch["layer_pattern"],
        between_block_mlp_layers=arch["between_block_mlp_layers"],
        layer_norm=arch["layer_norm"],
        ssm_kernel=arch["ssm_kernel"],
    )
    model = HybridGPT2S4LMHeadModel(cfg)
    model.load_state_dict(state, strict=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # if args.tokens is not None:
    #     # One forward pass: greedy next-token prediction at every position (teacher-forced input).
    #     input_ids = torch.tensor([instance], dtype=torch.long, device=device)
    #     position_ids = torch.tensor([pos_ids], dtype=torch.long, device=device)
    #     model.eval()
    #     with torch.no_grad():
    #         logits = model(input_ids, position_ids=position_ids).logits
    #     shift_pred = logits[0, :-1, :].argmax(dim=-1)
    #     gold_next = instance[1:]
    #     pred_next = shift_pred.tolist()
    #     correct = pred_next == gold_next
    #     print("Task:", args.task)
    #     print("Sequence:", " ".join(tokenizer.convert_ids_to_tokens(instance, rm_special=False)))
    #     print("Greedy next-token exact match (entire sequence):", correct)
    #     if not correct:
    #         for t in range(len(pred_next)):
    #             if pred_next[t] != gold_next[t]:
    #                 print(
    #                     f"  pos {t}->{t+1}: gold {tokenizer.vocab_inv[gold_next[t]]!r} "
    #                     f"pred {tokenizer.vocab_inv[pred_next[t]]!r}"
    #                 )
    #     return

    if args.tokens is not None:
        out = infer_one_sample(model, tokenizer, instance, pos_ids, label, device)
        print("Task:", args.task)
        print("Full sequence:", " ".join(out["sequence_tokens"]))
        print("Gold answer:  ", " ".join(out["gold_answer_tokens"]), f"({out['gold_answer_str']!r})")
        print("Pred answer:  ", " ".join(out["pred_answer_tokens"]), f"({out['pred_answer_str']!r})")
        print("Answer correct:", out["correct"])
    else:
        results = [
            infer_one_sample(model, tokenizer, instance, pos_ids, label, device)
            for instance, pos_ids, label in eval_ds
        ]
        average_accuracy = sum(result["correct"] for result in results) / len(results)
        print("Task:", args.task)
        print("Num samples:", len(results))
        print("Average accuracy:", average_accuracy)


if __name__ == "__main__":
    main()
