"""
Load a trained HybridGPT2S4 checkpoint and run one forward pass on a task sample.

Architecture is read from the weights filename (same convention as language_modeling_train.py),
unless you pass explicit --n-layer, --n-head, --d-model, --dropout.

Examples:
  python language_modeling_infer_hybrid.py --weights logs/parity/hybas_2l1h16d0.0dr_weights.pt --task parity
  python language_modeling_infer_hybrid.py --weights path/to/hybsa_2l2h64d0dr_weights.pt --task addition --nope
  python language_modeling_infer_hybrid.py --weights ckpt.pt --task mqar --monoid cyclic --monoid_n 5 \\
      --n-layer 2 --n-head 2 --d-model 64 --dropout 0
"""

from __future__ import annotations

import argparse
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

# Same as language_modeling_train_hybrid.py (position embedding size / eval setup).
TRAIN_LENGTH_RANGE = (0, 50)
MAX_TEST_LENGTH = 150


_WEIGHTS_NAME_RE_NEW = re.compile(
    r"^hyb([as]+)_(\d+)l(\d+)h(\d+)d([0-9.]+)dr(smalllr)?_weights\.pt$"
)
_WEIGHTS_NAME_RE_LEGACY = re.compile(
    r"^hyb(as|sa)(\d+)l(\d+)h(\d+)d([0-9.]+)dr(smalllr)?_weights\.pt$"
)


def parse_architecture_from_weights_path(path: str) -> dict:
    base = os.path.basename(path)
    m = _WEIGHTS_NAME_RE_NEW.match(base)
    if m:
        motif, n_rep, nh, nd, dr_s, _smalllr = m.groups()
        return {
            "layer_pattern": motif,
            "n_pattern_repeats": int(n_rep),
            "n_head": int(nh),
            "d_model": int(nd),
            "dropout": float(dr_s),
        }
    m = _WEIGHTS_NAME_RE_LEGACY.match(base)
    if m:
        variant, nl, nh, nd, dr_s, _smalllr = m.groups()
        return {
            "layer_pattern": variant,
            "n_pattern_repeats": int(nl),
            "n_head": int(nh),
            "d_model": int(nd),
            "dropout": float(dr_s),
        }
    raise ValueError(
        f"Cannot parse hybrid architecture from filename {base!r}. "
        "Expected hyb<pattern>_<repeats>l<n_head>h<d_model>d<dropout>dr[smalllr]_weights.pt "
        "or legacy hyb<as|sa><repeats>l..., "
        "or pass --n-layer, --n-head, --d-model, --dropout explicitly."
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
    input_ids = torch.tensor([instance], dtype=torch.long, device=device)
    position_ids = torch.tensor([pos_ids], dtype=torch.long, device=device)

    model.eval()
    with torch.no_grad():
        logits = model(input_ids, position_ids=position_ids).logits

    # logits[t] predicts token at t+1 (same as Trainer compute_metrics shift).
    shift_pred = logits[0, :-1, :].argmax(dim=-1)

    answer_indices = [i for i in range(1, len(instance)) if label[i] != pad_id]
    gold_answer = [instance[i] for i in answer_indices]
    pred_answer = [shift_pred[i - 1].item() for i in answer_indices]
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
    parser = argparse.ArgumentParser(description="Single-sample inference for hybrid SSM/attention LM.")
    parser.add_argument("--weights", type=str, required=True, help="Path to *_weights.pt from training.")
    parser.add_argument("--task", type=str, choices=TASK_CHOICES, required=True)
    parser.add_argument("--nope", action="store_true", help="Must match the checkpoint (no positional embeddings).")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed when drawing a random sample (default: 0).")
    parser.add_argument(
        "--tokens",
        type=str,
        default=None,
        help="Optional: whitespace-separated token strings (vocabulary keys, e.g. '<bos>' '0' '1' '<sep>' '0' '<eos>'). "
        "If omitted, one example is sampled from the task generator.",
    )
    parser.add_argument("--monoid", type=str, default="parity", choices=["parity", "cyclic"])
    parser.add_argument("--monoid_n", type=int, default=2)
    parser.add_argument("--key_size", type=int, default=32)
    parser.add_argument("--query_fraction", type=float, default=0.2)
    parser.add_argument("--n-layer", type=int, default=None)
    parser.add_argument("--n-head", type=int, default=None)
    parser.add_argument("--d-model", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument(
        "--start-with-attention",
        action="store_true",
        help="Only when passing explicit --n-layer/--n-head/--d-model/--dropout: use alternating pattern starting with attention ('as').",
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
        args.dropout is not None,
    )
    if any(explicit) and not all(explicit):
        parser.error("If you pass any of --n-layer, --n-head, --d-model, --dropout, pass all four.")

    if all(explicit):
        if args.layer_pattern is not None:
            motif = args.layer_pattern.strip().lower()
        else:
            motif = "as" if args.start_with_attention else "sa"
        arch = {
            "layer_pattern": motif,
            "n_pattern_repeats": args.n_layer,
            "n_head": args.n_head,
            "d_model": args.d_model,
            "dropout": args.dropout,
        }
    else:
        try:
            arch = parse_architecture_from_weights_path(args.weights)
        except ValueError as e:
            raise SystemExit(str(e)) from e

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    train_ds = build_task_dataset(
        args.task,
        MAX_TEST_LENGTH,
        args.monoid,
        args.monoid_n,
        args.key_size,
        args.query_fraction,
    )
    tokenizer = train_ds.tokenizer

    if args.tokens is not None:
        instance = tokens_string_to_ids(tokenizer, args.tokens)
        pos_ids = train_ds.get_pos_ids(len(instance), max(0, train_ds.n_positions - len(instance)))
    else:
        eval_ds = EvalDataset(train_ds, 1)
        instance, pos_ids, label = eval_ds[0]

    cfg = HybridConfig(
        vocab_size=len(tokenizer),
        n_positions=train_ds.n_positions,
        n_embd=arch["d_model"],
        n_head=arch["n_head"],
        dropout=arch["dropout"],
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        nope=args.nope,
        n_pattern_repeats=arch["n_pattern_repeats"],
        layer_pattern=arch["layer_pattern"],
    )
    model = HybridGPT2S4LMHeadModel(cfg)
    state = torch.load(args.weights, map_location="cpu")
    model.load_state_dict(state, strict=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    if args.tokens is not None:
        # One forward pass: greedy next-token prediction at every position (teacher-forced input).
        input_ids = torch.tensor([instance], dtype=torch.long, device=device)
        position_ids = torch.tensor([pos_ids], dtype=torch.long, device=device)
        model.eval()
        with torch.no_grad():
            logits = model(input_ids, position_ids=position_ids).logits
        shift_pred = logits[0, :-1, :].argmax(dim=-1)
        gold_next = instance[1:]
        pred_next = shift_pred.tolist()
        correct = pred_next == gold_next
        print("Task:", args.task)
        print("Sequence:", " ".join(tokenizer.convert_ids_to_tokens(instance, rm_special=False)))
        print("Greedy next-token exact match (entire sequence):", correct)
        if not correct:
            for t in range(len(pred_next)):
                if pred_next[t] != gold_next[t]:
                    print(
                        f"  pos {t}->{t+1}: gold {tokenizer.vocab_inv[gold_next[t]]!r} "
                        f"pred {tokenizer.vocab_inv[pred_next[t]]!r}"
                    )
        return

    out = infer_one_sample(model, tokenizer, instance, pos_ids, label, device)
    print("Task:", args.task)
    print("Full sequence:", " ".join(out["sequence_tokens"]))
    print("Gold answer:  ", " ".join(out["gold_answer_tokens"]), f"({out['gold_answer_str']!r})")
    print("Pred answer:  ", " ".join(out["pred_answer_tokens"]), f"({out['pred_answer_str']!r})")
    print("Answer correct:", out["correct"])


if __name__ == "__main__":
    main()
