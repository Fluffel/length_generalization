# Quickstart

Code for [*A Formal Framework for Understanding Length Generalization in Transformers*](https://openreview.net/forum?id=U49N5V51rU), extended with a new **MQAR Word Problem** task.

## What's new in this fork

- **MQAR Word Problem task** (`MQARWordProblemDataset`) — combines associative recall with monoid state tracking. See [Task description](#mqar-word-problem) below.
- **Monoid presets** — `parity_monoid()` (Z_2 XOR), `cyclic_monoid(n)` (Z_n), `monoid_from_cayley_table()` (arbitrary)
- **`pyproject.toml`** for dependency management
- **API updates** — migrated `evaluation_strategy` to `eval_strategy` for transformers >= 4.45 compatibility

## Setup

Requires Python >= 3.10. Using [uv](https://github.com/astral-sh/uv) (recommended):

```bash
git clone git@github.com:Fluffel/length_generalization.git
cd length_generalization

uv venv .venv
source .venv/bin/activate

# Core deps only (algorithmic/ and appendix-G* experiments)
uv pip install -e "."

# Everything (includes hydra, wandb, matplotlib, etc.)
uv pip install -e ".[all]"
```

Or with pip:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e "."       # core
pip install -e ".[all]"  # everything
```

### Optional dependency groups

| Group | What it includes | Needed for |
|---|---|---|
| (core) | torch, numpy, transformers, accelerate | `algorithmic/`, `appendix-G2/`, `appendix-G7/` |
| `formal` | hydra, wandb, pydantic-settings | `formal_lang_suite/` |
| `appendix` | easydict | `appendix-G7/` |
| `viz` | matplotlib, seaborn | Plotting scripts |
| `all` | Everything above | Full repo |

## Repository layout

```
algorithmic/                 # Main experiments (Table 1 in paper)
  language_modeling_train.py # Models, datasets, training (monolithic)
  run_multiple_seeds.py      # Multi-seed averaged evaluation
formal_lang_suite/           # Formal language experiments (Hydra + W&B)
appendix-G2/                 # Attention pattern expressiveness (Appendix G.2)
appendix-G7/                 # Generalized copy task (Appendix G.7)
```

## Running experiments

All training commands assume you are in the `algorithmic/` directory:

```bash
cd algorithmic
```

### Original tasks (from the paper)

Train on lengths 0-50, evaluate on 0-50 / 51-100 / 101-150. Runs a grid search over architectures (layers, heads, d_model, lr).

```bash
# Standard positional embeddings (APE)
python language_modeling_train.py --task bin_majority
python language_modeling_train.py --task parity
python language_modeling_train.py --task sort

# No positional embeddings (NoPE)
python language_modeling_train.py --task bin_majority --nope

# With PE regularizer
python language_modeling_train.py --task bin_majority --regularize 0.0001
```

Available tasks: `bin_majority`, `majority`, `bin_majority_interleave`, `unique_copy`, `repeat_copy`, `sort`, `parity`, `addition`.

Results are written to `lm-out-new-{task}/summary.txt`.

### Multi-seed evaluation

After the grid search finds a good architecture, run it across multiple seeds:

```bash
python run_multiple_seeds.py --tasks bin_majority sort --num_run 5
python run_multiple_seeds.py --tasks parity --num_run 5 --nope
```

Results are written to `lm-out-new-multi-run/`.

### MQAR Word Problem

```bash
# Parity monoid (Z_2 under XOR), 32 keys, 20% of sequence as queries
python language_modeling_train.py --task mqar_word_problem --monoid parity --key_size 32 --query_fraction 0.2

# Cyclic monoid Z_5, 64 keys
python language_modeling_train.py --task mqar_word_problem --monoid cyclic --monoid_n 5 --key_size 64

# With NoPE
python language_modeling_train.py --task mqar_word_problem --monoid parity --key_size 32 --nope
```

See the dedicated section below for details on the task.

### Formal language suite

Requires the `formal` dependency group and W&B credentials:

```bash
export WANDB_API_KEY=your_key
export WANDB_TEAM=your_team

cd formal_lang_suite
python train_with_ce.py dataset=tomita-1 model.use_nope=True model.num_layers=2
```

Dataset configs are in `formal_lang_suite/configs/dataset/`. Pre-generated datasets are in `formal_lang_suite/generated_ds/`.

## MQAR Word Problem

### Task definition

Given a finite monoid (M, ., e) and a key alphabet K:

1. **Update phase**: T unique key-value pairs `(k_i, m_i)` where `k_i` is from K and `m_i` is from M
2. **Query phase**: Q keys drawn without replacement from the T update keys
3. **Output**: a single monoid element — the left-fold of the retrieved values in query order

```
Example with parity (Z_2 under XOR):

  Update:  (a, 1) (b, 0) (c, 0) (d, 1)
  Query:   b c d
  Retrieve: 0 0 1
  Fold:    0 XOR 0 XOR 1 = 1
  Answer:  1

Sequence: <bos> a 1 b 0 c 0 d 1 <sep> b c d <sep> 1 <eos>
```

The task tests two capabilities simultaneously:
- **Associative recall** — finding the right value in a haystack of T pairs
- **State tracking** — computing the ordered monoid product of Q retrieved elements

### Parameters

| Flag | Default | Description |
|---|---|---|
| `--monoid` | `parity` | Monoid preset: `parity` (Z_2 XOR) or `cyclic` (Z_n addition) |
| `--monoid_n` | `2` | Order n for the cyclic monoid (only used with `--monoid cyclic`) |
| `--key_size` | `32` | Number of distinct keys \|K\| |
| `--query_fraction` | `0.2` | Fraction of content length devoted to queries |

### Length generalization behavior

- `length_range` controls the total content length (2T + Q tokens)
- Increasing length scales **both** T (more key-value pairs to search) and Q (longer monoid product)
- `query_fraction` determines the T/Q split: with the default 0.2, ~80% of content tokens are update pairs and ~20% are queries

### Adding new monoids

The dataset class takes an arbitrary binary operation. To add a new monoid:

```python
# In language_modeling_train.py or your own script:

# Option 1: Simple callable
op = lambda a, b: (a * b) % 6  # Z_6 under multiplication
identity = 1
monoid_size = 6

# Option 2: Cayley table (for non-abelian or irregular monoids)
from language_modeling_train import monoid_from_cayley_table
s3_table = [...]  # 6x6 table for S_3
op, identity, monoid_size = monoid_from_cayley_table(s3_table, identity=0)
```

Then pass `op`, `identity`, and `monoid_size` to `MQARWordProblemDataset`.

## Key concepts

- **APE vs NoPE**: Standard positional embeddings (APE) vs. no positional encoding (NoPE). The paper shows NoPE enables length generalization when the task has a CRASP[] program, while APE generalizes when the task has a CRASP[Periodic, Local] program.
- **Position ID randomization**: During training, position IDs are randomly offset so the model can't rely on absolute position. Critical for the length generalization setup.
- **PE regularizer**: Penalizes the norm of the positional attention pattern to encourage position-independent attention.
