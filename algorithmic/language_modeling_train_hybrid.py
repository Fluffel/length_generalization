from transformers import TrainingArguments, Trainer, TrainerCallback
import torch
import numpy as np
import random
import argparse
import os

from dataset_generators import BinaryMajorityDataset, MajorityDataset, BinaryMajorityInterleaveDataset, UniqueCopyDataset, RepeatCopyDataset, SortDataset, ParityDataset, AdditionDataset, MQARWordProblemDataset
from dataset_generators import EvalDataset
from models import HybridConfig, HybridGPT2S4LMHeadModel


def compute_metrics(eval_preds):
    logits, labels = eval_preds
    shift_logits = logits[:, :-1]
    shift_labels = labels[:, 1:]
    predictions = np.argmax(shift_logits, axis=-1)
    correct = np.all((predictions == shift_labels) | (shift_labels == -100), axis=1)
    return {"acc": correct.sum() / len(correct)}


class HybridCallback(TrainerCallback):
    def on_evaluate(self, state, args, control, metrics=None, logs=None, eval_dataloader=None, **kwargs):
        assert metrics["epoch"] >= getattr(self, "current_epoch", 0)
        if metrics["epoch"] > getattr(self, "current_epoch", 0):
            self.latest_acc = {}
            self.current_epoch = metrics["epoch"]
        for key in metrics.keys():
            if key.endswith("acc"):
                self.latest_acc[key] = metrics[key]
        if len(self.latest_acc) == len(test_length_ranges):
            if (self.latest_acc[f"eval_len{train_length_range[0]}-{train_length_range[1]}_acc"] == 1.0) or (self.current_epoch == 1.0):
                if self.latest_acc[f"eval_len{train_length_range[0]}-{train_length_range[1]}_acc"] == 1.0:
                    control.should_training_stop = True
                    global fit_train_data
                    fit_train_data = True
                    msg = f"early stop {self.current_epoch}\t\t"
                else:
                    msg = "reach max step\t\t"
                if self.latest_acc[f"eval_len{train_length_range[0]}-{train_length_range[1]}_acc"] >= 0.99:
                    msg = ">> " + msg
                print(
                    f"hyb{n_layer}l{n_head}h{d_model}d{dropout}dr\t\t",
                    msg,
                    "\t\t".join([f"{k}: {v}" for k, v in self.latest_acc.items()]),
                    f"\t\tlr: {lr}",
                    file=summary_f,
                )
                summary_f.flush()

                if (self.latest_acc[f"eval_len{train_length_range[0]}-{train_length_range[1]}_acc"] == 1.0) and (self.latest_acc[f"eval_len{test_length_ranges[1][0]}-{test_length_ranges[1][1]}_acc"] == 1.0):
                    global should_stop
                    should_stop = True


class customCollator():
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, examples):
        input_ids, pos_ids, labels = tuple(zip(*examples))
        max_len = max(len(item) for item in input_ids)

        [item.extend([self.pad_id,] * (max_len - len(item))) for item in input_ids]
        input_ids = torch.LongTensor(input_ids)
        [item.extend([self.pad_id,] * (max_len - len(item))) for item in labels]
        labels = torch.LongTensor(labels)
        labels[labels == self.pad_id] = -100
        [item.extend([item[-1],] * (max_len - len(item))) for item in pos_ids]
        pos_ids = torch.LongTensor(pos_ids)

        batch = {"input_ids": input_ids, "position_ids": pos_ids, "labels": labels}
        return batch


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, choices=["bin_majority", "majority", "bin_majority_interleave", "unique_copy", "repeat_copy", "sort", "parity", "addition", "mqar"])
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--job-id", type=int, default=0)
    parser.add_argument("--nope", action="store_true", help="Disable positional embeddings in the hybrid stack")
    parser.add_argument("--start_with_ssm", action="store_true", help="If set, layer order is [s4, attn, ...]. Default is [attn, s4, ...]")
    parser.add_argument("--monoid", type=str, default="parity", choices=["parity", "cyclic"])
    parser.add_argument("--monoid_n", type=int, default=2)
    parser.add_argument("--key_size", type=int, default=32)
    parser.add_argument("--query_fraction", type=float, default=0.2)
    args = parser.parse_args()

    configs = [(l, h, d, dr, lr) for l in [1, 2, 4] for h in [1, 2, 4] for d in [16, 64, 256] for dr in [0, 0.1] for lr in [1e-3, 1e-4]]
    # configs = [(l, h, d, dr, lr) for l in [2] for h in [2] for d in [64] for dr in [0] for lr in [1e-3]]

    train_length_range = (0, 50)
    test_length_ranges = [train_length_range] + [(51, 100), (101, 150)]
    max_test_length = test_length_ranges[-1][1]
    batch_size = 64
    per_device_bz = batch_size // torch.cuda.device_count() if torch.cuda.is_available() else batch_size
    test_num = 2_000

    for seed in range(args.seeds):
        torch.manual_seed(seed)
        random.seed(seed)

        match args.task:
            case "bin_majority":
                train_dataset = BinaryMajorityDataset(train_length_range, max_test_length)
                test_dataset = {f"len{test_range[0]}-{test_range[1]}": EvalDataset(BinaryMajorityDataset(test_range, max_test_length, add_positional_offset=False), test_num) for test_range in test_length_ranges}
            case "majority":
                train_dataset = MajorityDataset(train_length_range, max_test_length)
                test_dataset = {f"len{test_range[0]}-{test_range[1]}": EvalDataset(MajorityDataset(test_range, max_test_length, add_positional_offset=False), test_num) for test_range in test_length_ranges}
            case "bin_majority_interleave":
                train_dataset = BinaryMajorityInterleaveDataset(train_length_range, max_test_length, period=3)
                test_dataset = {f"len{test_range[0]}-{test_range[1]}": EvalDataset(BinaryMajorityInterleaveDataset(test_range, max_test_length, period=3, add_positional_offset=False), test_num) for test_range in test_length_ranges}
            case "unique_copy":
                train_dataset = UniqueCopyDataset(train_length_range, max_test_length)
                test_dataset = {f"len{test_range[0]}-{test_range[1]}": EvalDataset(UniqueCopyDataset(test_range, max_test_length, add_positional_offset=False), test_num) for test_range in test_length_ranges}
            case "repeat_copy":
                train_dataset = RepeatCopyDataset(train_length_range, max_test_length)
                test_dataset = {f"len{test_range[0]}-{test_range[1]}": EvalDataset(RepeatCopyDataset(test_range, max_test_length, add_positional_offset=False), test_num) for test_range in test_length_ranges}
            case "sort":
                train_dataset = SortDataset(train_length_range, max_test_length)
                test_dataset = {f"len{test_range[0]}-{test_range[1]}": EvalDataset(SortDataset(test_range, max_test_length, add_positional_offset=False), test_num) for test_range in test_length_ranges}
            case "parity":
                train_dataset = ParityDataset(train_length_range, max_test_length)
                test_dataset = {f"len{test_range[0]}-{test_range[1]}": EvalDataset(ParityDataset(test_range, max_test_length, add_positional_offset=False), test_num) for test_range in test_length_ranges}
            case "addition":
                train_dataset = AdditionDataset(train_length_range, max_test_length)
                test_dataset = {f"len{test_range[0]}-{test_range[1]}": EvalDataset(AdditionDataset(test_range, max_test_length, add_positional_offset=False), test_num) for test_range in test_length_ranges}
            case "mqar":
                train_dataset = MQARWordProblemDataset(train_length_range, max_test_length, key_size=args.key_size, query_fraction=args.query_fraction, monoid_type=args.monoid, monoid_n=args.monoid_n)
                test_dataset = {
                    f"len{test_range[0]}-{test_range[1]}": EvalDataset(
                        MQARWordProblemDataset(test_range, max_test_length, add_positional_offset=False, key_size=args.key_size, query_fraction=args.query_fraction, monoid_type=args.monoid, monoid_n=args.monoid_n),
                        test_num,
                    )
                    for test_range in test_length_ranges
                }

        n_positions = train_dataset.n_positions
        tokenizer = train_dataset.tokenizer

        task_path = f"./logs/{args.task}"
        if not os.path.exists(task_path):
            os.mkdir(task_path)
        suffix = args.job_id if args.job_id != 0 else ""
        summary_f = open(os.path.join(task_path, f"summaryhybrid{suffix}.txt"), "a")

        should_stop = False
        fit_train_data = False
        for n_layer, n_head, d_model, dropout, lr in configs:
            if n_layer > 2:
                max_steps = 60_000
                warmup_steps = 3000
                if fit_train_data:
                    break
            else:
                max_steps = 30_000
                warmup_steps = 0

            output_dir = f"hyb{n_layer}l{n_head}h{d_model}d{dropout}dr{'smalllr' if lr == 1e-4 else ''}"
            output_dir = os.path.join(task_path, output_dir)

            cfg = HybridConfig(
                vocab_size=len(tokenizer),
                n_positions=n_positions,
                n_embd=d_model,
                n_layer=n_layer,
                n_head=n_head,
                dropout=dropout,
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                nope=args.nope,
                start_with_attention=(not args.start_with_ssm),
            )
            model = HybridGPT2S4LMHeadModel(cfg)

            training_args = TrainingArguments(
                output_dir=output_dir,
                per_device_train_batch_size=per_device_bz,
                per_device_eval_batch_size=per_device_bz,
                max_steps=max_steps,
                eval_strategy="steps",
                eval_steps=3_000,
                save_strategy="no",
                logging_strategy="steps",
                logging_steps=3_000,
                learning_rate=lr,
                weight_decay=0.01,
                optim='adamw_torch',
                lr_scheduler_type='linear',
                warmup_steps=warmup_steps,
                report_to="none",
            )

            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=test_dataset,
                data_collator=customCollator(tokenizer.pad_token_id),
                compute_metrics=compute_metrics,
                callbacks=[HybridCallback],
            )
            trainer.train()
            if should_stop:
                break

        summary_f.close()
