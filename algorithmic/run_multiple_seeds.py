import torch

import random
import argparse
import os
import re
from algorithmic.language_modeling_train_transformer import *


class myCallback2(TrainerCallback):
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
                    msg = f"early stop {self.current_epoch}\t\t"
                else:
                    msg = "reach max step\t\t"
                if self.latest_acc[f"eval_len{train_length_range[0]}-{train_length_range[1]}_acc"] >= threshold:
                    msg = "** " + msg
                    for key, value in self.latest_acc.items():
                        results[key].append(value)
                print(f"{n_layer}l{n_head}h{d_model}d\t\t", msg, "\t\t".join([f"{k}: {v}" for k, v in self.latest_acc.items()]), f"\t\tlr: {lr}", file=summary_f)
                summary_f.flush()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_run", type=int, default=5)
    parser.add_argument("--nope", action="store_true")
    parser.add_argument("--regularize", type=float, default=0.0)
    parser.add_argument("--tasks", nargs='+', required=True)
    args = parser.parse_args()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    if not args.nope:
        task_arch = {"bin_majority": "1l1h16d",
                    "majority": "1l2h256d",
                    "bin_majority_interleave": "2l4h256dsmalllr",
                    "unique_copy": "2l1h64d", 
                    "repeat_copy": "4l4h256d", 
                    "sort": "1l2h256dsmalllr", 
                    "parity": "4l2h256dsmalllr", 
                    "addition": "big", 
        }
    else:
        task_arch = {"bin_majority": "1l1h16d",
                    "majority": "1l1h64d",
                    "bin_majority_interleave": "big",
                    "unique_copy": "4l4h256d", 
                    "repeat_copy": "4l4h256d", 
                    "sort": "1l1h256d", 
                    "parity": "bigsmalllr", 
                    "addition": "big", 
        }

    train_length_range = (0, 50)
    test_length_ranges = [train_length_range] + [(51, 100), (101, 150)]
    max_test_length = test_length_ranges[-1][1]
    batch_size = 64
    per_device_bz = batch_size // torch.cuda.device_count() if torch.cuda.is_available() else batch_size 
    test_num = 2_000

    save_path = f"./lm-out-new-multi-run"
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    if args.nope:
        suffix = "-nope"
    elif args.regularize != 0:
        suffix = f"-reg{args.regularize}"
    else:
        suffix = ""
    

    for task in args.tasks:
        arch = task_arch[task]
        summary_f = open(os.path.join(save_path, f"{task}-average{suffix}.txt"), "w")
        print("\n\ntask: ", task, "\t", arch, "\n", file=summary_f)

        if not arch.startswith("big"):
            lr = 1e-3 if "smalllr" not in arch else 1e-4
            n_layer = int(re.search(r"(\d+)l", arch).group(1))
            n_head = int(re.search(r"l(\d+)h", arch).group(1))
            d_model = int(re.search(r"h(\d+)d", arch).group(1))
            max_steps = 30_000
            warmup_steps = 0
            threshold = 1.0
        else:
            lr = 1e-4 if "smalllr" not in arch else 3e-5
            n_layer = 12
            n_head = 12
            d_model = 768
            max_steps = 60_000
            warmup_steps = 3000
            threshold = 0.0



        print("hyper-parameters", n_layer, n_head, d_model, lr)

        results = {f"eval_len{test_range[0]}-{test_range[1]}_acc": [] for test_range in test_length_ranges}
        
        for seed in range(1000):
            torch.manual_seed(seed)
            random.seed(seed)

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

                case "mqar_word_problem":
                    train_dataset = MQARWordProblemDataset(
                        train_length_range, max_test_length,
                        args.key_size, args.query_fraction, args.monoid, args.monoid_n
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

    
            for j in range(3):
                print("\ninput example:")
                print(" ".join(tokenizer.convert_ids_to_tokens(test_dataset[f"len{test_length_ranges[0][0]}-{test_length_ranges[0][1]}"][j][0])))
                print("label example:")
                print(" ".join(tokenizer.convert_ids_to_tokens(test_dataset[f"len{test_length_ranges[0][0]}-{test_length_ranges[0][1]}"][j][2])))

        

            cfg = GPT2Config(vocab_size=len(tokenizer), 
                        n_positions=n_positions,
                        n_embd=d_model,
                        n_layer=n_layer,
                        n_head=n_head,
                        bos_token_id=tokenizer.bos_token_id, 
                        eos_token_id=tokenizer.eos_token_id,
                        pad_token_id=tokenizer.pad_token_id,
                        attn_pdrop=0,
                        resid_pdrop=0,
                        embd_pdrop=0,
                        )

            if args.nope:
                model = NoPEGPT2LMHeadModel(cfg)
            elif args.regularize != 0:
                model = RegGPT2LMHeadModel(cfg, args.regularize)
            else:
                model = GPT2LMHeadModel(cfg)

            training_args = TrainingArguments(
                output_dir=os.path.join(save_path, "temp"),    
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
                callbacks=[myCallback2],
            )

            trainer.train()
        
            if len(results[f"eval_len{train_length_range[0]}-{train_length_range[1]}_acc"]) == args.num_run:
                break
        
        # trainer.save_model(os.path.join(save_path, f"{task}-average{suffix}"))
        print("mean results\t\t",  "\t\t".join([f"{key}: {(sum(value) / args.num_run):.4f}" for key, value in results.items()]), file=summary_f)
        summary_f.flush()
        
        summary_f.close()
