from transformers import GPT2Config, TrainingArguments, Trainer, TrainerCallback
import torch
import numpy as np
import random
import argparse
import os

from dataset_generators import BinaryMajorityDataset, MajorityDataset, BinaryMajorityInterleaveDataset, UniqueCopyDataset, RepeatCopyDataset, SortDataset, ParityDataset, AdditionDataset, MQARWordProblemDataset
from dataset_generators import EvalDataset
from models import GPT2LMHeadModel, RegGPT2LMHeadModel, NoPEGPT2LMHeadModel, S4ForSequenceModeling
from models import S4Config
    
def compute_metrics(eval_preds):
    logits, labels = eval_preds
    shift_logits = logits[:, :-1]
    shift_labels = labels[:, 1:]
    predictions = np.argmax(shift_logits, axis=-1)
    correct = np.all((predictions == shift_labels) | (shift_labels == -100), axis=1)
    return {"acc": correct.sum() / len(correct)}

# class TransformerCallback(TrainerCallback):
#     def on_evaluate(self, state, args, control, metrics=None, logs=None, eval_dataloader=None, **kwargs):
#         assert metrics["epoch"] >= getattr(self, "current_epoch", 0)
#         if metrics["epoch"] > getattr(self, "current_epoch", 0):
#             self.latest_acc = {}
#             self.current_epoch = metrics["epoch"]
#         for key in metrics.keys():
#             if key.endswith("acc"):
#                 self.latest_acc[key] = metrics[key]
#         if len(self.latest_acc) == len(test_length_ranges):
#             if (self.latest_acc[f"eval_len{train_length_range[0]}-{train_length_range[1]}_acc"] == 1.0) or (self.current_epoch == 1.0):  
#                 if self.latest_acc[f"eval_len{train_length_range[0]}-{train_length_range[1]}_acc"] == 1.0: 
#                     control.should_training_stop = True
#                     global fit_train_data
#                     fit_train_data = True
#                     msg = f"early stop {self.current_epoch}\t\t"
#                 else:
#                     msg = "reach max step\t\t"
#                 if self.latest_acc[f"eval_len{train_length_range[0]}-{train_length_range[1]}_acc"] >= 0.99:
#                     msg = ">> " + msg
#                 print(f"{n_layer}l{n_head}h{d_model}d\t\t", msg, "\t\t".join([f"{k}: {v}" for k, v in self.latest_acc.items()]), f"\t\tlr: {lr}", file=summary_f)
#                 summary_f.flush()

#                 if (self.latest_acc[f"eval_len{train_length_range[0]}-{train_length_range[1]}_acc"] == 1.0) and (self.latest_acc[f"eval_len{test_length_ranges[1][0]}-{test_length_ranges[1][1]}_acc"] == 1.0):
#                     global should_stop
#                     should_stop = True

class SSMCallback(TrainerCallback):
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
                print(f"{n_layer}l{d_model}d{dropout}dr\t\t", msg, "\t\t".join([f"{k}: {v}" for k, v in self.latest_acc.items()]), f"\t\tlr: {lr}", file=summary_f)
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
    # parser.add_argument("--model", type=str, choices=["gpt", "gpt_nope", "gpt_reg", "s4"])
    # parser.add_argument("--nope", action="store_true")
    # parser.add_argument("--regularize", type=float, default=0.0)
    # MQAR-specific arguments
    parser.add_argument("--monoid", type=str, default="parity", choices=["parity", "cyclic"],
                        help="Monoid preset for mqar_word_problem task")
    parser.add_argument("--monoid_n", type=int, default=2,
                        help="Order n for cyclic monoid Z_n (only used when --monoid=cyclic)")
    parser.add_argument("--key_size", type=int, default=32,
                        help="Number of distinct keys |K| for mqar_word_problem")
    parser.add_argument("--query_fraction", type=float, default=0.2,
                        help="Fraction of content length devoted to queries for mqar_word_problem")
    args = parser.parse_args()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    torch.manual_seed(0)
    random.seed(0)

    train_length_range = (0, 50)
    test_length_ranges = [train_length_range] + [(51, 100), (101, 150)]
    max_test_length = test_length_ranges[-1][1]
    batch_size = 64
    per_device_bz = batch_size // torch.cuda.device_count() if torch.cuda.is_available() else batch_size 
    test_num = 2_000

    # configs = [(l, d, dr, lr) for l in [1, 2, 4] for d in [16, 64, 256] for dr in [0] for lr in [1e-3, 1e-4]]
    configs = [(l, d, dr, lr) for l in [4] for d in [256] for dr in [0] for lr in [1e-3, 1e-4]]
    # configs.append((12, 12, 768, 1e-4))
    # configs = [(12, 12, 768, 1e-4)]

    match args.task:
        case "bin_majority":
            train_dataset = BinaryMajorityDataset(train_length_range, max_test_length)

            test_dataset = {
                f"len{test_range[0]}-{test_range[1]}": EvalDataset(BinaryMajorityDataset(test_range, max_test_length, add_positional_offset=False), test_num)
                    for test_range in test_length_ranges
            }

        case "majority":
            train_dataset = MajorityDataset(train_length_range, max_test_length)

            test_dataset = {
                f"len{test_range[0]}-{test_range[1]}": EvalDataset(MajorityDataset(test_range, max_test_length, add_positional_offset=False), test_num)
                    for test_range in test_length_ranges
            }

        case "bin_majority_interleave":
            train_dataset = BinaryMajorityInterleaveDataset(train_length_range, max_test_length, period=3)

            test_dataset = {
                f"len{test_range[0]}-{test_range[1]}": EvalDataset(BinaryMajorityInterleaveDataset(test_range, max_test_length, period=3, add_positional_offset=False), test_num)
                    for test_range in test_length_ranges
            }

        case "unique_copy":
            train_dataset = UniqueCopyDataset(train_length_range, max_test_length)

            test_dataset = {
                f"len{test_range[0]}-{test_range[1]}": EvalDataset(UniqueCopyDataset(test_range, max_test_length, add_positional_offset=False), test_num)
                    for test_range in test_length_ranges
            }
        
        case "repeat_copy":
            train_dataset = RepeatCopyDataset(train_length_range, max_test_length)

            test_dataset = {
                f"len{test_range[0]}-{test_range[1]}": EvalDataset(RepeatCopyDataset(test_range, max_test_length, add_positional_offset=False), test_num)
                    for test_range in test_length_ranges
            }

        case "sort":
            train_dataset = SortDataset(train_length_range, max_test_length)

            test_dataset = {
                f"len{test_range[0]}-{test_range[1]}": EvalDataset(SortDataset(test_range, max_test_length, add_positional_offset=False), test_num)
                    for test_range in test_length_ranges
            }

        case "parity":
            train_dataset = ParityDataset(train_length_range, max_test_length)

            test_dataset = {
                f"len{test_range[0]}-{test_range[1]}": EvalDataset(ParityDataset(test_range, max_test_length, add_positional_offset=False), test_num)
                    for test_range in test_length_ranges
            }

        case "addition":
            train_dataset = AdditionDataset(train_length_range, max_test_length)

            test_dataset = {
                f"len{test_range[0]}-{test_range[1]}": EvalDataset(AdditionDataset(test_range, max_test_length, add_positional_offset=False), test_num)
                    for test_range in test_length_ranges
            }

        case "mqar":
            train_dataset = MQARWordProblemDataset(
                
                train_length_range, max_test_length,
                key_size=args.key_size, query_fraction=args.query_fraction, monoid_type=args.monoid, monoid_n=args.monoid_n
            )

            test_dataset = {
                f"len{test_range[0]}-{test_range[1]}": EvalDataset(
                    MQARWordProblemDataset(
                        test_range, max_test_length, add_positional_offset=False,
                        key_size=args.key_size, query_fraction=args.query_fraction, monoid_type=args.monoid, monoid_n=args.monoid_n
                    ), test_num)
                for test_range in test_length_ranges
            }

    n_positions = train_dataset.n_positions
    tokenizer = train_dataset.tokenizer

    # task_path = f"./lm-out-new-{args.model}-{args.task}"
    task_path = f"./ssm-out-new-{args.task}"
    if not os.path.exists(task_path):
        os.mkdir(task_path)
    # if args.nope:
    #     suffix = "-nope"
    # elif args.regularize != 0:
    #     suffix = f"-reg{args.regularize}"
    # else:
    #     suffix = ""
    summary_f = open(os.path.join(task_path, f"summary.txt"), "a")

    for i in range(3):
        print("\ninput example:")
        print(" ".join(tokenizer.convert_ids_to_tokens(test_dataset[f"len{test_length_ranges[0][0]}-{test_length_ranges[0][1]}"][i][0])))
        print("label example:")
        print(" ".join(tokenizer.convert_ids_to_tokens(test_dataset[f"len{test_length_ranges[0][0]}-{test_length_ranges[0][1]}"][i][2])))

    should_stop = False
    fit_train_data = False
    for n_layer, d_model, dropout, lr in configs: 

        if n_layer > 4:
            max_steps = 60_000
            warmup_steps = 3000
            if fit_train_data:
                break
        else:
            max_steps = 30_000
            warmup_steps = 0

        output_dir = f"{n_layer}l{d_model}d{dropout}dr{'smalllr' if lr == 1e-4 else ''}"
        output_dir = os.path.join(task_path, output_dir)

        # if args.model in ["gpt", "gpt_nope", "gpt_reg"]:
        #     cfg = GPT2Config(vocab_size=len(tokenizer), 
        #             n_positions=n_positions,
        #             n_embd=d_model,
        #             n_layer=n_layer,
        #             n_head=n_head,
        #             bos_token_id=tokenizer.bos_token_id, 
        #             eos_token_id=tokenizer.eos_token_id,
        #             pad_token_id=tokenizer.pad_token_id,
        #             attn_pdrop=0,
        #             resid_pdrop=0,
        #             embd_pdrop=0,
        #             )
        # else: # s4 only s4 model currently
        cfg = S4Config(vocab_size=len(tokenizer),
                        n_embd=d_model,
                        n_layers=n_layer,
                        dropout=dropout
        )

                
        # match args.model:
        #     case "gpt":
        #         model = GPT2LMHeadModel(cfg)
        #     case "gpt_nope":
        #         model = NoPEGPT2LMHeadModel(cfg)
        #     case "gpt_reg":
        #         model = RegGPT2LMHeadModel(cfg, args.regularize)
        #     case "s4":
        model = S4ForSequenceModeling(cfg)


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

        data_collator = customCollator(tokenizer.pad_token_id)

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            callbacks=[SSMCallback],
        )

        trainer.train()

        if should_stop:
            break

    
    summary_f.close()
    