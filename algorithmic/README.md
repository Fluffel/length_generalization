### Guidelines to reproduce the experiments for algorithmic tasks

1. Run `python language_modeling_train.py --task [TASK]` to sweep hyperparamters (check the code for possible tasks), and check the results in corresponding output folder (`lm-out-new-[TASK]`), select the optimal set of hyperparamters according to section E.3, and put it into the dictionary `task_arch` in `run_multiple_seeds.py`. Run this script for one task at a time. It runs experiments for APE by default, add `--nope` to run for NoPE models. 
2. After getting the best configuration, use `python run_multiple_seeds.py --tasks [TASK1] [TASK2] [TASK3]` to run each task repeatedly with multiple random seeds, it will generate files in foler `lm-out-new-multi-run`, you can check the average accuracies there. You can run multiple tasks together. Again, add `--nope` if you want to run for NoPE models. You can also run experiments with the regularizer activated, by setting `--regularize [VALUE]`.

#### Inspecting Datasets

Datasets can be inspected with `convenience_scripts/print_dataset_words.py`. Example:

```bash
python algorithmic/convenience_scripts/print_dataset_words.py \
  --dataset MajorityDataset \
  --dataset-kwargs '{"length_range":[20,30],"max_test_length":100}'
```

#### Aggregating summary results and plotting

Two scripts (repo root is three levels up from `algorithmic/convenience_scripts/`):

- **`convenience_scripts/generate_summary_csv.py`** — scan `logs/**/summary*.txt`, merge into one CSV, optional CLI listing.
- **`convenience_scripts/generate_summary_plots.py`** — read that CSV and plot one task.

##### Unified CSV (`generate_summary_csv.py`)

- **Input:** `logs/<task>/summary*.txt` (default `--logs-root` is repo `logs/`).
- **Output:** `exports/all_summary_results.csv` by default (`--csv` to override).
- Each summary line can list **any** `eval_len<N>-<M>_acc:` buckets; they are stored as plain ranges in the `bucket` column (e.g. `0-50`, `101-150`, `0-24`).
- Extra columns parse the model string: `arch`, `layers`, `heads`, `d_model`, `dropout`, `mlp_size`, `kernel`, `pe`, `ln`, `train_steps_k`, `layer_order`.
- **`--bucket-end-digit D`** (with **`--create` only**, repeat for several digits): drop rows unless the bucket **upper** bound’s ones digit is `D` (e.g. `0` keeps `0-50` and `51-100`; `9` keeps `25-49`).

```bash
python algorithmic/convenience_scripts/generate_summary_csv.py --create
python algorithmic/convenience_scripts/generate_summary_csv.py --csv exports/all_results.csv --create
python algorithmic/convenience_scripts/generate_summary_csv.py --create --bucket-end-digit 0 --bucket-end-digit 5
```

##### Pattern notation (CSV listing and plots)

Short tokens are matched against **CSV columns** (comma **inside** one pattern = AND; repeat `--include-pattern` = OR). Examples: `1l` (layers), `lm` / `hyb` / `ssm` (arch), `nope` / `pe`, `s4` (kernel), `0.001lr` (learning rate), `0-50` (bucket). A pattern **without** commas falls back to legacy model-string token matching after a single-token structured check.

- **`--remove-pattern`** (repeatable): drop rows that match **any** of these patterns (applied before includes).
- **`--include-pattern`** (repeatable): if any are given, keep rows that match **any** pattern **or** satisfy `--include` (exact `model` / `model:lr1,lr2`).

Listing:

```bash
python algorithmic/convenience_scripts/generate_summary_csv.py --list-models --task bin_majority

python algorithmic/convenience_scripts/generate_summary_csv.py \
  --list-models \
  --task bin_majority --task majority,mqar \
  --include-pattern ssm \
  --include-pattern 1l,1h1

python algorithmic/convenience_scripts/generate_summary_csv.py \
  --list-models \
  --task mqar \
  --remove-pattern lm \
  --include-pattern ssm \
  --include-max-only
```

`--include-max-only` with `--list-models` keeps only “max contributor” `(model, lr)` pairs per task (per-bin maxima, then non-dominated across bins). Listing prints one line per `(task, model, learning_rate)` with **all** bucket columns present in the filtered rows (`len<range>=…`).

##### Plots (`generate_summary_plots.py`)

- **`--task`** (required): must match the CSV **`task` column exactly** (usually the `logs/<name>/` folder name, e.g. `parity`, `sort`, `unique_copy`). Same comma/repeat syntax as `generate_summary_csv.py`; **one task per plot invocation** (not a comma-separated list of several tasks).
- **`--input-csv` / `--csv`:** CSV path (default `exports/all_summary_results.csv`). Use the **same file** you passed to `generate_summary_csv.py` when debugging “listing finds rows but plot does not.”
- **`--output` / `--plot-path`:** image path (default `exports/plots/<task>.png`).
- **`--include-pattern` / `--remove-pattern`:** same semantics as the CSV script (plot script has **no** `--include`; use patterns or rely on task-only filter).
- **`--group-pattern`:** **after** `--task` / include / remove filtering, rows matching each pattern merge into plotted series (this flag **never** drops CSV rows). Bin sets must be **nested** to stay one line; **incomparable** grids split with a warning. **Different** patterns (hyb vs ssm) are separate series. **`--group-pattern '*'`** pools leftovers under the same nesting rule.
- **Legend:** compact training-style string rebuilt with `parse_model_spec` (arch / hybrid order / kernel / `Nl` `Nh` `Nd` / dropout / mlp / pe–ln flags / `stp`–`k` / learning rate)—not the raw CSV `model` cell. Groups with several configs join with ` | `.
- **X-axis:** data positions are **bin upper bounds** (e.g. bucket `0-49` → x = 49); axis starts at 0. **`--x-ticks ends`** (default): tick at each bin end with `<N>` labels. **`--x-ticks regular --x-tick-step 10`:** evenly spaced numeric ticks.
- **`--include-max`:** for **each** plotted series, add a dotted `max:…` line (same color as that series when solids are drawn). Winners use a **shared** bin-end grid across the figure, **finest** (narrowest) bin at each end, then Pareto pruning **within** that series.
- **`--include-max-only`:** only those dotted max lines (one per series), no solid lines.
- Duplicate `(series, bucket)` values in the CSV are aggregated with **mean** and sample **std** as error bars.

Example — three architectures with MLP, three grouped lines, max-only:

```bash
python algorithmic/convenience_scripts/generate_summary_plots.py \
  --input-csv exports/all_results.csv \
  --task unique_copy \
  --include-pattern hyb,mlp \
  --include-pattern ssm,mlp \
  --include-pattern lm,mlp \
  --group-pattern hyb,mlp \
  --group-pattern ssm,mlp \
  --group-pattern lm,mlp \
  --include-max-only
```

Example — solids + per-series max, wildcard rest group:

```bash
python algorithmic/convenience_scripts/generate_summary_plots.py \
  --task sort \
  --output exports/plots/sort_custom.png \
  --title "Sort (selected)" \
  --include-pattern 4l \
  --include-pattern 8l \
  --include-pattern hyb \
  --group-pattern ssm,16d \
  --group-pattern ssm,256d \
  --group-pattern hyb \
  --group-pattern '*' \
  --include-max
```