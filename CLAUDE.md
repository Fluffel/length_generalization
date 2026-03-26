# CLAUDE.md - Length Generalization in Transformers

## Paper

Code for: *A Formal Framework for Understanding Length Generalization in Transformers* ([OpenReview](https://openreview.net/forum?id=U49N5V51rU)).
Fork origin: `Fluffel/length_generalization` (upstream: `lacoco-lab/length_generalization`).

## Repository Structure

Three independent experiment directories, no shared code between them:

### `algorithmic/` — Algorithmic Tasks (Section 5 / Table 1)

**Core file:** `language_modeling_train.py` (~700 lines, monolithic)

Contains everything: model variants, tokenizer, datasets, training loop, evaluation.

- **Models:** All GPT-2 based (HuggingFace `transformers`), trained from scratch.
  - `GPT2LMHeadModel` — standard (APE = Absolute Positional Embeddings)
  - `NoPEGPT2LMHeadModel` — replaces `wpe` with a no-op (NoPE = No Positional Encoding)
  - `RegGPT2LMHeadModel` — APE + regularizer on `PE @ W_Q @ (PE @ W_K)^T` (encourages position-independent attention)
- **Custom tokenizer:** `customTokenizer` with special tokens `<bos>`, `<sep>`, `<eos>`, `<pad>`.
- **Tasks** (each an `IterableDataset`):
  - `BinaryMajorityDataset` — binary majority vote
  - `MajorityDataset` — 26-letter majority vote
  - `BinaryMajorityInterleaveDataset` — interleaved binary majority (period=3)
  - `UniqueCopyDataset` — copy unique tokens
  - `RepeatCopyDataset` — copy with repetitions
  - `SortDataset` — sort tokens
  - `ParityDataset` — parity of binary string
  - `AdditionDataset` — binary addition
  - `MQARWordProblemDataset` — Multi-Query Associative Recall + Monoid Word Problem (see below)
- **Sequence format:** `<bos> [input tokens] <sep> [answer tokens] <eos>`, left-padded labels so loss is only on answer tokens.
- **Position ID randomization:** During training, position IDs are offset by a random amount within `[0, max_test_length - length]` to encourage position-invariance.
- **Grid search:** Iterates over `(n_layer, n_head, d_model, lr)` configs. Early stops when train accuracy hits 1.0.
- **Length ranges:** Train on lengths 0-50, test on 0-50, 51-100, 101-150.

**Multi-seed runner:** `run_multiple_seeds.py` — runs the best architecture (hardcoded per task) across multiple seeds, averages results.

**Output:** `lm-out-new-{task}/summary*.txt` files with per-config accuracy results, and `lm-out-new-multi-run/` for averaged results.

**CLI:**
```bash
python language_modeling_train.py --task bin_majority [--nope] [--regularize 0.0001]
python language_modeling_train.py --task mqar_word_problem --monoid parity --key_size 32 --query_fraction 0.2
python language_modeling_train.py --task mqar_word_problem --monoid cyclic --monoid_n 5 --key_size 64
python run_multiple_seeds.py --tasks bin_majority sort --num_run 5 [--nope] [--regularize 0.0001]
```

#### MQAR Word Problem Task

Combines **associative recall** (finding key-value pairs in a haystack) with **monoid state tracking** (folding retrieved values via a binary operation).

**Sequence format:** `<bos> k1 m1 k2 m2 ... kT mT <sep> q1 ... qQ <sep> answer <eos>`
- Update phase: T unique (key, monoid_element) pairs
- Query phase: Q keys (sampled without replacement from the T update keys)
- Answer: single monoid element = left-fold of retrieved values in query order

**Parameters:**
- `--monoid {parity, cyclic}` — monoid preset (`parity` = Z_2 XOR, `cyclic` = Z_n addition)
- `--monoid_n N` — order for cyclic monoid
- `--key_size K` — number of distinct keys
- `--query_fraction F` — fraction of content length for queries (default 0.2)

**Vocabulary:** `[k0..k_{K-1}, m0..m_{M-1}]` — keys and monoid elements in a single flat vocab.

**Extensibility:** The class takes an arbitrary `op(a, b) -> c` callable + `identity`, so new monoids (S_3, arbitrary Cayley tables via `monoid_from_cayley_table()`, etc.) can be added without changing the dataset class.

**Planned extension (TODO):** CoT/steps mode where intermediate fold results are output per query position (e.g., for Q queries producing values v1, v2, v3: output `v1, op(v1,v2), op(op(v1,v2),v3)` instead of just the final result).

### `formal_lang_suite/` — Formal Language Suite (Section 5 / Formal languages)

Uses **Hydra** for config management and **W&B** for experiment tracking. Custom training loop (not HuggingFace Trainer).

**Training entry point:** `train_with_ce.py`
- Hydra config at `configs/defaults.yaml`, overrides via `configs/{dataset,model,optimizer,scheduler,train}/*.yaml`
- `get_model()` builds GPT-2 with the same three variants (standard / NoPE / Reg)
- `train_with_ce()` — manual training loop with CE loss, per-epoch eval on validation bins
- `offset_and_forward()` — randomizes position IDs during training (same idea as algorithmic/)
- Requires `WANDB_API_KEY` and `WANDB_TEAM` env vars (via `config.py` / python-decouple)

**Data pipeline:**
- `dataloader.py` — corpus classes: `TomitaCorpus`, `StarFreeCorpus`, `NonStarFreeCorpus`, `StarFreePostLanguageCorpus`
- `dataloader_utils.py` — corpus creation with train/val split and multiple validation bins at increasing lengths
- `dataset_utils.py` — preprocessing, tokenization (character-level), `DatasetClass(Dataset)`, dataloader creation
- `generators/` — language generators:
  - `tomita_generator.py` — Tomita grammars 1-7 (DFA-based)
  - `starfree_generator.py` — Star-free languages: `D_n` (Dyck), `AAStarBBStar`, `AB_D_BC`, `ZOT_Z_T`
  - `nonstarfree_generator.py` — Non-star-free: `ABABStar`, `AAStar`, `AnStarA2`
  - `crl_generator.py` — Cyclic regular languages (DFA-based, CRL1-5)
- `generated_ds/` — pre-generated datasets as text files (`train_src.txt`, `train_tgt.txt`, `val_src_bin{0,1,2}.txt`, `val_tgt_bin{0,1,2}.txt`)

**Dataset configs** (`configs/dataset/*.yaml`): Each specifies `lang_fam` (Tomita/StarFree/NonStarFree), window sizes, training/test sizes, number of validation bins, and `len_incr` for OOD length bins.

**Experiment sweep configs** (`experiments/*.yaml`): W&B sweep definitions (grid search over model hyperparams).

**Visualization:** `visualise/vis_algo_formal_size_by_side_bigger.py` — generates paper Figure 1 comparing APE vs NoPE across tasks.

**CLI:**
```bash
cd formal_lang_suite
python train_with_ce.py dataset=tomita-1 model.use_nope=True model.num_layers=2
```

### `appendix-G2/` — Attention Pattern Expressiveness (Appendix G.2)

Tests whether `p^T A p` (positional embedding inner product through attention weights) can represent various attention patterns.

- `test_multi_func_L2.py` — trains a `pTAp` module to fit target attention patterns:
  - `j = i - c` (fixed offset)
  - `j > i - c` (threshold)
  - `(i-j) mod c1 == c2` (periodic)
  - `j < i - c`, `i-j is prime`, combined patterns
- Tests with d=32 and d=256 to study effect of embedding dimension on expressiveness.
- `test_multi_func_L2_vis.py` — visualization of results.

### `appendix-G7/` — Generalized Copy Task (Appendix G.7)

Extends the unique copy task with variable `diff_ij` (copy every k-th token instead of every token).

- `utils.py` — modified `customTokenizer` (adds `#` special token), `UniqueCopyDataset` with `diff_ij` parameter, `make_configs()` for hyperparameter search
- `search_hyper.py` — grid search over architectures for each `diff_ij` setting
- `run_multiple.py` — multi-seed evaluation of best architecture from search

**CLI:**
```bash
cd appendix-G7
python search_hyper.py --diff_ij 2 [--nope] [--length_range small]
python run_multiple.py --diff_ij 2 --num_run 5 [--nope]
```

## Key Concepts

- **Length generalization:** Train on short sequences (e.g., length 0-50), test on longer ones (51-100, 101-150).
- **APE vs NoPE:** The paper's main finding is that NoPE (removing positional encodings) enables length generalization when the task has a CRASP[] program, while APE generalizes when the task has a CRASP[Periodic, Local] program.
- **Position ID randomization:** Both `algorithmic/` and `formal_lang_suite/` randomly offset position IDs during training, which is critical for the length generalization setup.
- **Regularizer:** Penalizes `||tril(PE W_Q W_K^T PE^T)||^2` to encourage position-independent attention patterns.

## Dependencies

- `transformers` (HuggingFace) — GPT-2 models and Trainer
- `torch` — PyTorch
- `hydra-core`, `omegaconf` — config management (formal_lang_suite only)
- `wandb` — experiment tracking (formal_lang_suite only)
- `pydantic-settings`, `python-decouple` — settings/env management (formal_lang_suite only)
- `easydict` — config dicts (appendix-G7 only)
- `matplotlib`, `seaborn` — visualization

## Important Notes

- Dependencies are managed via `pyproject.toml`. Install with `uv pip install -e "."` (core) or `uv pip install -e ".[all]"` (everything). See `QUICKSTART.md`.
- Code is duplicated across directories (e.g., `NoPE`, `customTokenizer`, `EvalDataset` appear in `algorithmic/`, `formal_lang_suite/`, and `appendix-G7/` with slight variations).
- The `algorithmic/` directory is the most self-contained and easiest to work with.
- All models use dropout=0 by default (no regularization via dropout).
- Training uses HuggingFace `Trainer` in `algorithmic/` and `appendix-G7/`, but a manual loop in `formal_lang_suite/`.
