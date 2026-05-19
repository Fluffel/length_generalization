### Running code + convenience scripts

## Where code lives and what to edit

- `algorithmic/language_modeling_train.py`  
  Main training/eval implementation (dataset construction, model build, trainer/eval loop).

- `algorithmic/utils.py`  
  Shared config dataclasses/default config factories (`RunConfig`, default sweeps).

- `algorithmic/run_scripts/language_modeling_train_shared.py`  
  Shared CLI parser and argument-to-config mapping.

- `algorithmic/run_scripts/language_modeling_train_hybrid.py`  
  Hybrid run entrypoint with architecture sweep definition.

### Parameter control: where to change what

- **Architecture control (edit code):**  
  Adjust architecture sweep in `run_scripts/language_modeling_train_hybrid.py` inside `build_architectures(...)` (layers/heads/d_model/dropout/lr/etc).

- **Most experiment controls (CLI args):**  
  Pass via script arguments parsed in `run_scripts/language_modeling_train_shared.py`, e.g.:
  - task/seeds: `--task`, `--seeds`
  - train schedule: `--train-steps`, `--warmup-steps`, `--eval-steps`, `--logging-steps`
  - length setup: `--train-length-range`
  - model toggles: `--nope`, `--noln`, `--use-olmo`, `--ssm-kernel`, `--hybrid-layer-pattern`, `--regularize`
  - task params: `--key-len`, `--mkar-vocab-size`, `--marker-vocab-size`, `--key_size`, `--monoid`, `--monoid_n`, `--query-fraction-lower`, `--query-fraction-upper`

### Training example

```bash
python algorithmic/run_scripts/language_modeling_train_hybrid.py \
  --task mkar \
  --seeds 5 \
  --train-length-range 0,50 \
  --nope \
  --key-len 4 \
  --mkar-vocab-size 128
```

## Dataset inspection helper

Script: `convenience_scripts/print_dataset_words.py`

What it does:
- Prints generated samples token-by-token for quick dataset debugging
- Supports two modes:
  - `--mode class`: instantiate a dataset class directly
  - `--mode build`: call `build_datasets(run_config)` and inspect train/test splits

Direct class example:

```bash
python algorithmic/convenience_scripts/print_dataset_words.py \
  --mode class \
  --module algorithmic.dataset_generators \
  --dataset MajorityDataset \
  --dataset-kwargs '{"length_range":[20,30],"max_test_length":100}'
```

Build-datasets example:

```bash
python algorithmic/convenience_scripts/print_dataset_words.py \
  --mode build \
  --module algorithmic.dataset_generators \
  --task parity \
  --split train \
  --num 5
```

## Convenience scripts (new workflow)

### 1) Build a unified summary CSV

Script: `convenience_scripts/generate_summary_csv.py`

What it does:
- Scans `logs/**/summary*.txt`
- Parses model/task metadata and evaluation buckets
- Writes one flat CSV

Default paths:
- Logs root: `logs/`
- Output CSV: `exports/summary.csv`

Example:

```bash
python algorithmic/convenience_scripts/generate_summary_csv.py
```

Custom output:

```bash
python algorithmic/convenience_scripts/generate_summary_csv.py \
  --logs-root logs \
  --csv exports/all_spec_task.csv
```

### 2) Query the CSV with DataFrame filters

Script: `convenience_scripts/query_summary_df.py`

Supported filtering:
- `--keep column=v1,v2`
- `--remove column=v1,v2`
- `--query "pandas_expr"`
- `--exclude-query "pandas_expr"`

Output behavior:
- Prints filtered rows to stdout
- By default shows datapoint columns: `task`, `model`, `bucket`, `accuracy`
- Add extra columns with repeatable `--show-cols`

Example:

```bash
python algorithmic/convenience_scripts/query_summary_df.py \
  --input-csv exports/all_spec_task.csv \
  --keep task=mqar,mkar \
  --keep arch=hyb \
  --query "bucket == '101-150' and accuracy >= 0.9" \
  --show-cols learning_rate \
  --show-cols mkar_vocab_size
```

### 3) Plot directly from DataFrame-filtered rows

Script: `convenience_scripts/generate_plot_df.py`

Supported filtering (same semantics as query script):
- `--keep column=v1,v2`
- `--remove column=v1,v2`
- `--query "pandas_expr"`
- `--exclude-query "pandas_expr"`

Key plotting controls:
- `--task <task>` (required)
- `--group-by <column>` (repeatable)
- `--group-label-mode {model,group,custom}`
- `--group-custom-labels "Label A,Label B,..."`
- `--max-aggregation {pareto_mean,bin_max}` (also accepts aliases `mean`/`max`)
- `--num-bins N`
- `--x-ticks {bins,ends,regular}`
- `--x-tick-step N`
- `--x-axis-break <number|auto>`

Example:

```bash
python algorithmic/convenience_scripts/generate_plot_df.py \
  --input-csv exports/all_spec_task.csv \
  --task mkar \
  --keep arch=hyb \
  --group-by mkar_vocab_size \
  --group-label-mode group \
  --x-ticks bins \
  --output exports/plots/mkar_vocab_size.svg
```

Custom group labels:

```bash
python algorithmic/convenience_scripts/generate_plot_df.py \
  --input-csv exports/all_spec_task.csv \
  --task selective_copy \
  --keep arch=hyb \
  --group-by pe \
  --group-label-mode custom \
  --group-custom-labels "With PE,No PE"
```

## Shared helper modules

- `convenience_scripts/dataframe_query_utils.py`  
  Shared filter parsing/application utilities used by `query_summary_df.py` and
  `generate_plot_df.py`.

- `convenience_scripts/plot_utils.py`  
  Shared plotting/statistics/legend helper functions used by `generate_plot_df.py`.
