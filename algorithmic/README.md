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

This workflow is now split into two scripts:

- `convenience_scripts/generate_summary_csv.py`: build/update the unified CSV and inspect model results.
- `convenience_scripts/generate_summary_plots.py`: generate plots from the unified CSV only.

##### 1) Build/update CSV and list results

Build or update the unified CSV (default: `exports/all_summary_results.csv`):
```
python algorithmic/convenience_scripts/generate_summary_csv.py --create-csv
```

List models/results for one task given an existing .csv file:
```
python algorithmic/convenience_scripts/generate_summary_csv.py \
  --list-models \
  --task bin_majority
  
```

List models/results for multiple tasks and patterns:
```
python algorithmic/convenience_scripts/generate_summary_csv.py \
  --list-models \
  --task bin_majority --task majority,mqar \
  --include-pattern ssm \
  --include-pattern h1l1
```

The listing prints one line per `(task, model, learning_rate)` with per-bin mean accuracy and sample count.

##### 2) Generate plots from CSV

Usage notes (plot script):
- `--task` is required and must match the task directory name in the CSV (e.g. `sort`, `bin_majority`).
- `--include` filters by exact model and optional learning rates: `model` or `model:lr1,lr2` (repeatable; OR across flags).
- `--include-pattern` filters by model-spec patterns (repeatable; OR across flags; order-insensitive token matching).
- `--group-pattern` groups matching models into one line per pattern.
- If `--group-pattern` is used without `\*`, unmatched models are still plotted individually.
- Use `--group-pattern \*` to group all remaining unmatched models into one final group (`\*` must be escaped in shell).
- `--include-max` adds dotted `max:...` lines per plotted line (original lines remain).
- `--include-max-only` plots only dotted `max:...` lines (mutually exclusive with `--include-max`).
- If multiple datapoints map to a plotted series and bucket, mean and sample std are shown as error bars.

Note: The --include-max flag adds a plot line over all datapoints that have a maximum value in any of the three bins. The max-plot line shows the mean and standard deviation over those (at most 3) datapoints.

Example: Plot `sort` while keeping only `4l`/`8l` and `hyb` models. Generate a plot-line for subgroups `16d` and `256d` for the ssm model, and group the remaining models into with `\*`. Additionally, add dotted max lines on top of all stem lines.
```
python algorithmic/convenience_scripts/generate_summary_plots.py \
  --task sort \
  --include-pattern 4l \
  --include-pattern 8l \
  --include-pattern hyb \
  --group-pattern ssm16d \
  --group-pattern ssm256d \
  --group-pattern hyb \
  --group-pattern \* \
  --include-max
```