### Guidelines to reproduce the experiments for algorithmic tasks

1. Run `python language_modeling_train.py --task [TASK]` to sweep hyperparamters (check the code for possible tasks), and check the results in corresponding output folder (`lm-out-new-[TASK]`), select the optimal set of hyperparamters according to section E.3, and put it into the dictionary `task_arch` in `run_multiple_seeds.py`. Run this script for one task at a time. It runs experiments for APE by default, add `--nope` to run for NoPE models. 
2. After getting the best configuration, use `python run_multiple_seeds.py --tasks [TASK1] [TASK2] [TASK3]` to run each task repeatedly with multiple random seeds, it will generate files in foler `lm-out-new-multi-run`, you can check the average accuracies there. You can run multiple tasks together. Again, add `--nope` if you want to run for NoPE models. You can also run experiments with the regularizer activated, by setting `--regularize [VALUE]`.

#### Inspecting Datasets

Datasets can be inspected with a script in `convenience_scripts/print_dataset_words.py`. Example usage:
```
    $ python algorithmic/print_dataset_words.py \
  --dataset MajorityDataset \
  --dataset-kwargs '{"length_range":[20,30],"max_test_length":100}'
```

#### Aggregating Summary Results and Plotting

You can aggregate all `logs/**/summary.txt` files into one CSV and generate per-task plots with:
`convenience_scripts/generate_summary_plots.py`.

Build or update the unified CSV (default: `logs/all_summary_results.csv`):
```
python algorithmic/convenience_scripts/generate_summary_plots.py
```

Generate a plot for one task (all models and learning rates in that task):
```
python algorithmic/convenience_scripts/generate_summary_plots.py \
  --plot \
  --task lm-out-new-sort
```

Generate a filtered plot by model and learning rate:
```
python algorithmic/convenience_scripts/generate_summary_plots.py \
  --plot \
  --task lm-out-new-sort \
  --include 1l1h64d:0.001,0.0001 \
  --include 1l1h256d:0.0001
```

Notes:
- `--task` is the directory name containing the corresponding `summary.txt`.
- `--include MODEL` keeps all learning rates for that model.
- `--include MODEL:lr1,lr2` keeps only selected learning rates.
- If multiple datapoints exist for the same `(task, model, learning_rate, bucket)`, the plot shows the mean with `+- std` error bars.