# Length Generalization

Code adapted from [*A Formal Framework for Understanding Length Generalization in Transformers*](https://openreview.net/forum?id=U49N5V51rU).

Train on short sequences (default lengths 0–50), evaluate on longer ones (51–100, 101–150). All architectures share one entrypoint, `algorithmic/language_modeling_train.py`, which is told which model to train with `--model <spec>`.

## Setup

### Cluster (HTCondor + Docker)

Dependencies live in the Docker image; `example.sub` is the job blueprint. For a minimal run, edit:

- `execute.sh` — set the local path to this project directory (line 2).
- `example.sub` — set `WANDB_*` environment variables if needed, and the log paths.

```bash
condor_submit example.sub
```

## Project structure

Active code lives in two places. Other experiment directories in this repo are leftover and will be removed.

```
algorithmic/                          training code
  language_modeling_train.py          single training entrypoint (CLI, loop, eval)
  language_modeling_infer_hybrid.py   inference from a saved checkpoint
  models.py / model_extensions.py     GPT-2, SSM, hybrid, and OLMo builders
  model_spec.py                       YAML spec loader
  model_specs/<family>/<variant>.yaml model specs (selected with --model)
  task_datasets.py                    dataset classes (majority, copy, MQAR, …)
  dataset_generators.py               task registry and train/eval construction
  utils.py                            RunConfig, architecture slots, curriculum
  run_record.py                       JSON run records

length-generalization-log-analysis/   git submodule: collected logs + plots
  logs/<task>/summary*.txt            text summaries copied from training
  scripts/                            CSV aggregation and plotting
```

**Train code.** `algorithmic/language_modeling_train.py` is the only entrypoint. It loads a model spec, builds the task datasets, runs HuggingFace `Trainer`, and writes summaries.

**Tasks.** Dataset classes live in `algorithmic/task_datasets.py`. `algorithmic/dataset_generators.py` maps `--task` names onto those classes and builds train/eval length bins. Algorithmic tasks stream examples on the fly; formal-language tasks (`tomita_*`, `d_*`, `aa_star`, …) materialize a fixed corpus once at the start of the run (seeded by `--dataset-seed`).

**Model specs.** Everything model-specific lives in `algorithmic/model_specs/<family>/<variant>.yaml`, loaded by `algorithmic/model_spec.py`. `--model hybrid/olmo_sa` selects that file. Families are `transformer`, `ssm`, and `hybrid`. List them with:

```bash
python algorithmic/language_modeling_train.py --list-models
```

A spec owns `model_family`, positional encoding (`use_nope`), the PE regularizer, SSM kernel, hybrid layer pattern, OLMo mixer settings, and the `architectures` sweep. Fields of an `architectures` entry may be lists; training iterates over the cross product. A spec can `extends:` another one (child keys win; `architectures` is replaced, not merged). `--model` also accepts a filesystem path, so one-off specs can live outside `model_specs/`. Every other hyperparameter is a CLI flag.

```yaml
# algorithmic/model_specs/transformer/gpt2.yaml (excerpt)
model_family: transformer
architectures:
  - n_layer: [1, 2, 4]
    n_head: [1, 2, 4]
    d_model: [16, 64, 256]
    dropout: [0, 0.1]
    lr: [1.0e-3, 1.0e-4]
    between_block_mlp_layers: 1
    layer_norm: true
```

## Training

```bash
python algorithmic/language_modeling_train.py --model transformer/gpt2 --task bin_majority --report-to none
python algorithmic/language_modeling_train.py --model transformer/gpt2_nope --task parity --seeds 5
python algorithmic/language_modeling_train.py --model hybrid/olmo_sa --task mkar --key-len 4 --mkar-vocab-size 128
python algorithmic/language_modeling_train.py --help
```

`--task` choices:

- Algorithmic: `bin_majority`, `majority`, `bin_majority_interleave`, `unique_copy`, `repeat_copy`, `sort`, `parity`, `parity_majority`, `addition`, `multiplication`, `mqar`, `s5`, `s5_limited`, `selective_state_tracking`, `flipflop`, `selective_copy`, `mkar`, `dyck_2`
- Formal languages: `tomita_1`–`tomita_7`, `d_2`, `d_3`, `d_4`, `d_12`, `aa_star`, `abab_star`, `aa_star_bb_star`, `ab_star_d_bc_star`, `012_star_0_2_star`, `an_star_a2`

Sequence format for algorithmic tasks is `<bos> [input] <sep> [answer] <eos>`; loss is only on answer tokens. During training, position IDs are randomly offset so the model cannot rely on absolute position.

Default eval bins follow the train window: with `--train-length-range 0,50` they are 0–50, 51–100, 101–150.

### Flags

`--model` and `--task` are required. Task-specific flags are ignored unless that task uses them. Live help: `python algorithmic/language_modeling_train.py --help`.

#### Model and run

| Flag | Default | Description |
|---|---|---|
| `--model` / `--model-specs` | (required) | Spec name under `model_specs/` (e.g. `hybrid/olmo_sa`) or a YAML path. |
| `--list-models` | | Print available specs and exit. |
| `--task` | (required) | Task name (see list above). |
| `--seed` | random | Single training seed (model init + on-the-fly stream). Cannot be combined with `--seeds`. |
| `--seeds` | `1` | Number of iterations, each with a freshly drawn seed. |
| `--dataset-seed` | random | Seed for materializing eval bins (and formal-language train corpora). Independent of `--seed`. |
| `--job-id` | `""` | Embedded in summary filenames (`summarylm{id}`, `summaryssm{id}`, `summaryhybrid{id}`). |

#### Schedule and lengths

If `--train-steps` / `--warmup-steps` are omitted, larger architectures get a bigger budget (30k vs 60k steps; warmup 0 vs 3000). `--eval-steps` and `--logging-steps` default to 3000.

| Flag | Default | Description |
|---|---|---|
| `--train-steps` | size-dependent | Trainer steps (overrides both the small and large budgets). |
| `--warmup-steps` | size-dependent | LR warmup steps. |
| `--eval-steps` | `3000` | Eval cadence. |
| `--logging-steps` | `3000` | Logging cadence. |
| `--train-length-range` | `0,50` | Inclusive `min,max` training length. Ignored if curriculum flags are set. |
| `--early-stop` | off | Stop a slot when the train-length bin is ~perfect. |
| `--solved-acc-threshold` | `0.98` | Stop when **every** eval bin reaches this accuracy. Set `> 1` to disable. |

Curriculum learning (all three required together) replaces `--train-length-range` with a sliding, non-overlapping window. Stage `i` trains on lengths `(i*S, (i+1)*S - 1)`; eval is still 1×/2×/3× of the *cumulative* length. A stage ending early advances to the next; only the last stage stops training.

| Flag | Description |
|---|---|
| `--curriculum-num-steps` | Number of stages `N`. |
| `--curriculum-step-size` | Window width `S`. |
| `--curriculum-steps-per-stage` | Trainer steps per stage. |

#### Logging and checkpoints

| Flag | Default | Description |
|---|---|---|
| `--report-to` | `wandb` | `wandb` or `none`. |
| `--wandb-project` | | W&B project. |
| `--wandb-entity` | | W&B entity. |
| `--wandb-group` | | W&B group. |
| `--json-log-dir` | `./json_logs` | Root for JSON run records (`{task}/` layout, same as the text logs). |
| `--save-final-weights` | off | Write a `*_weights.pt` checkpoint at the end of each slot. |

Text summaries always go to `./logs/<task>/` (not a flag).

#### Hybrid freezing

| Flag | Default | Description |
|---|---|---|
| `--freeze` | | Freeze `attention` or `ssm` weights in a hybrid model for the first `--freeze-fraction` of steps (re-applied at the start of every curriculum stage). Hybrid specs only. |
| `--freeze-fraction` | `0.5` | Fraction `(0, 1]` of steps (or of each curriculum stage) to keep those weights frozen. Requires `--freeze`. |

#### Task-specific

| Flag | Default | Used by | Description |
|---|---|---|---|
| `--monoid` | `parity` | `mqar`, `selective_state_tracking` | `parity` (Z₂ XOR), `cyclic` (Zₙ addition, MQAR only), `s5`, `s5_limited`. |
| `--monoid_n` | `2` | `mqar` + `--monoid cyclic` | Order of Zₙ. |
| `--query-fraction-lower` / `--query-fraction-upper` | `0.2` / `0.2` | `mqar` | Per-example query-length fraction is uniform in `[lower, upper]`. |
| `--key-len` | `4` | `mkar` | Query is the last `k` content tokens. |
| `--mkar-vocab-size` | `128` | `mkar` | Number of distinct content tokens. |
| `--marker-vocab-size` | `16` | `selective_copy` | Numbered markers `#1..#N`. |
| `--misc-vocab-size` | `16` | `selective_copy` | Filler tokens in the non-marker half of the vocab. |
| `--marker-frequency` | `0.2` | `selective_copy` | Lower bound; per-example frequency is uniform in `[this, 1]`. |
| `--sort-vocab-size` | train-length max | `sort` | Distinct content tokens. Raised to the max sequence length if smaller. |
| `--formal-packed-targets` | off | formal-language tasks | Serialize as `<bos> src <sep> tgt <eos>` instead of one target token per source token. |

## Logging

Training writes three things:

1. **Text summaries** under `./logs/<task>/summary{lm,ssm,hybrid}{job-id}.txt`. Each line is one architecture slot (and curriculum stage, if any) with per-bin accuracies. Transformer filenames also encode NoPE / regularizer (`summarylm-nope…`, `summarylm-reg0.0001…`). The file is append-only: a grid of architectures, or several seeds, all land in the same summary. Eval rows kept in the file are the Pareto front of per-bin maxima (at most one snapshot per length bin).
2. **JSON run records** under `./json_logs/<task>/` (override with `--json-log-dir`). Same stem as the text summary, `.json` suffix. The record stores the full `RunConfig`, dataset seed, and one entry per (seed, architecture) with those logged snapshots.
3. **W&B** when `--report-to wandb` (the CLI default). One W&B run per training seed.

Copy the text summaries into the **`length-generalization-log-analysis`** git submodule (`git@github.com:Fluffel/length-generalization-log-analysis.git`) under `logs/<task>/`. That repo is the collected log archive and the plotting pipeline: it parses `summary*.txt` into a CSV, draws per-task plots, and publishes them to [GitHub Pages](https://fluffel.github.io/length-generalization-log-analysis/). From the submodule:

```bash
python scripts/generate_summary_csv.py --logs-root logs --csv summary.csv
python scripts/update_plots.py --arch ssm    # also: hyb, lm
```

`scripts/generate_plot_df.py` plots a filtered slice of that CSV (keep/remove columns, bin layout, group-by, …). Pushing new files under `logs/` on the submodule’s `main` branch regenerates the site via GitHub Actions.
