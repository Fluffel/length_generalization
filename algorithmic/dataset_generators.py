from dataclasses import dataclass

try:
    from utils import RunConfig
except ImportError:
    # When imported as `algorithmic.dataset_generators` with repo root on sys.path
    from algorithmic.utils import RunConfig

try:
    # NB: this module must not be named `datasets.py` -- that would shadow the
    # HuggingFace `datasets` package (imported internally by `transformers.Trainer`)
    # whenever this directory is on `sys.path`, causing very confusing failures
    # (e.g. `Trainer` treating our `IterableDataset`s as HF `datasets.Dataset`s).
    from task_datasets import (
        AdditionDataset,
        BinaryMajorityDataset,
        BinaryMajorityInterleaveDataset,
        CustomDataset,
        Dyck2Dataset,
        EvalDataset,
        FlipFlopDataset,
        FormalLanguageDataset,
        MajorityDataset,
        MKARDataset,
        MQARWordProblemDataset,
        NonStarFreeCorpus,
        ParityDataset,
        RepeatCopyDataset,
        SelectiveCopyDataset,
        SortDataset,
        StarFreeCorpus,
        StarFreePostLanguageCorpus,
        TomitaCorpus,
        UniqueCopyDataset,
        customTokenizer,
    )
except ImportError:
    from algorithmic.task_datasets import (
        AdditionDataset,
        BinaryMajorityDataset,
        BinaryMajorityInterleaveDataset,
        CustomDataset,
        Dyck2Dataset,
        EvalDataset,
        FlipFlopDataset,
        FormalLanguageDataset,
        MajorityDataset,
        MKARDataset,
        MQARWordProblemDataset,
        NonStarFreeCorpus,
        ParityDataset,
        RepeatCopyDataset,
        SelectiveCopyDataset,
        SortDataset,
        StarFreeCorpus,
        StarFreePostLanguageCorpus,
        TomitaCorpus,
        UniqueCopyDataset,
        customTokenizer,
    )


# ── Formal-language task routing ────────────────────────────────────────────
# Task names handled by the formal-language corpus/tokenizer pipeline
# (fixed-size pre-generated corpora) rather than the algorithmic on-the-fly
# IterableDataset pipeline below.
_NONSTARFREE_LANG_MAP = {
    "abab_star": ("ABABStar", 2),
    "aa_star": ("AAStar", 2),
    "an_star_a2": ("AnStarA2", 2),
}
_STARFREE_POST_LANG_TASKS = {"ab_star_d_bc_star", "012_star_0_2_star"}

# Every task `_make_task_dataset` can construct, in the order it matches them.
# Kept next to that function so the two cannot drift apart (enforced by
# algorithmic/tests/test_task_registry.py) and exported so CLIs can offer it as
# `--task` choices instead of re-listing the tasks themselves.
ALGORITHMIC_TASKS: tuple[str, ...] = (
    "bin_majority",
    "majority",
    "bin_majority_interleave",
    "unique_copy",
    "repeat_copy",
    "sort",
    "parity",
    "addition",
    "mqar",
    "flipflop",
    "selective_copy",
    "mkar",
    "dyck_2",
)


def is_formal_task(task: str) -> bool:
    return (
        task.startswith("tomita_")
        or task.startswith("d_")
        or task in _NONSTARFREE_LANG_MAP
        or task == "aa_star_bb_star"
        or task in _STARFREE_POST_LANG_TASKS
    )


def _make_task_dataset(
    run_config: RunConfig,
    length_range: tuple[int, int],
    max_test_length: int,
    add_positional_offset: bool,
) -> CustomDataset:
    """Instantiate the dataset class for ``run_config.task`` over an arbitrary length range.

    Factored out of ``build_datasets`` so curriculum learning (which needs fresh
    datasets for many different, dynamically-computed length ranges) can reuse the
    exact same per-task construction logic instead of duplicating it.
    """
    task = run_config.task

    match task:
        case "bin_majority":
            return BinaryMajorityDataset(length_range, max_test_length, add_positional_offset=add_positional_offset)
        case "majority":
            return MajorityDataset(length_range, max_test_length, add_positional_offset=add_positional_offset)
        case "bin_majority_interleave":
            return BinaryMajorityInterleaveDataset(
                length_range, max_test_length, period=3, add_positional_offset=add_positional_offset
            )
        case "unique_copy":
            return UniqueCopyDataset(length_range, max_test_length, add_positional_offset=add_positional_offset)
        case "repeat_copy":
            return RepeatCopyDataset(length_range, max_test_length, add_positional_offset=add_positional_offset)
        case "sort":
            return SortDataset(
                length_range,
                max_test_length,
                add_positional_offset=add_positional_offset,
                vocab_size=run_config.sort_vocab_size,
                # cover_vocab=add_positional_offset and run_config.sort_vocab_size is not None,
            )
        case "parity":
            return ParityDataset(length_range, max_test_length, add_positional_offset=add_positional_offset)
        case "addition":
            return AdditionDataset(length_range, max_test_length, add_positional_offset=add_positional_offset)
        case "mqar":
            return MQARWordProblemDataset(
                length_range,
                max_test_length,
                add_positional_offset=add_positional_offset,
                key_size=run_config.key_size,
                query_fraction_upper=run_config.query_fraction_upper,
                query_fraction_lower=run_config.query_fraction_lower,
                monoid_type=run_config.monoid,
                monoid_n=run_config.monoid_n,
            )
        case "flipflop":
            return FlipFlopDataset(length_range, max_test_length, add_positional_offset=add_positional_offset)
        case "selective_copy":
            return SelectiveCopyDataset(
                length_range,
                max_test_length,
                add_positional_offset=add_positional_offset,
                marker_vocab_size=run_config.marker_vocab_size,
            )
        case "mkar":
            return MKARDataset(
                length_range,
                max_test_length,
                add_positional_offset=add_positional_offset,
                key_len=run_config.key_len,
            )
        case "dyck_2":
            return Dyck2Dataset(length_range, max_test_length, add_positional_offset=add_positional_offset)
        case _:
            raise ValueError(f"Unknown task {task!r}")


def build_datasets(run_config: RunConfig, corpus_size_limit: int | None = None):
    """Build the train dataset and the dict of per-length-bin eval datasets.

    Args:
        corpus_size_limit: cap on the number of examples generated per split. Only
            affects formal-language tasks, whose corpora are pre-generated up front
            (the algorithmic tasks stream indefinitely, and their eval bins are sized
            by ``run_config.test_num``). Leave at ``None`` for training; inspection
            tools set it to avoid spending minutes generating a full corpus -- e.g.
            `tomita_6` -- just to look at a handful of examples.
    """
    if is_formal_task(run_config.task):
        return _build_formal_datasets(run_config, corpus_size_limit=corpus_size_limit)

    train_length_range = run_config.train_length_range
    test_length_ranges = run_config.test_length_ranges
    max_test_length = test_length_ranges[-1][1]
    test_num = run_config.test_num

    train_dataset = _make_task_dataset(run_config, train_length_range, max_test_length, add_positional_offset=True)
    test_dataset = {
        f"len{r[0]}-{r[1]}": EvalDataset(
            _make_task_dataset(run_config, r, max_test_length, add_positional_offset=False),
            test_num,
        )
        for r in test_length_ranges
    }

    return train_dataset, test_dataset, train_length_range, test_length_ranges


def build_curriculum_datasets(run_config: RunConfig):
    """Build datasets for curriculum learning (see ``utils.CurriculumConfig``).

    Only supported for algorithmic tasks: formal-language tasks are backed by
    fixed-size, pre-generated corpora (see ``_build_formal_datasets``) rather than
    an ``IterableDataset`` whose ``range_min``/``range_max`` can be grown in place
    as curriculum stages advance.

    Returns:
        train_dataset: a single dataset instance whose ``range_max`` is mutated in
            place by ``CurriculumTrainCallback`` as training advances through stages.
            ``range_min`` never changes across stages (curriculum stages always start
            at length 0, clamped by each task's own minimum length at construction).
        stage_eval_datasets: list indexed by stage (0-indexed), each a dict of three
            ``EvalDataset``s keyed like ``"len{a}-{b}"`` for that stage's 1x/2x/3x bins.
            Precomputed once up front (mirrors ``build_datasets``' ``test_dataset``)
            since the stage schedule doesn't depend on architecture or seed.
    """
    curriculum = run_config.curriculum
    assert curriculum is not None, "build_curriculum_datasets requires run_config.curriculum to be set"
    assert not is_formal_task(run_config.task), (
        f"curriculum learning is not supported for formal-language task {run_config.task!r}"
    )

    max_test_length = curriculum.max_test_length
    test_num = run_config.test_num

    train_dataset = _make_task_dataset(
        run_config, curriculum.stage_train_range(0), max_test_length, add_positional_offset=True
    )

    stage_eval_datasets: list[dict[str, EvalDataset]] = []
    for stage_idx in range(curriculum.num_steps):
        stage_eval_datasets.append(
            {
                f"len{r[0]}-{r[1]}": EvalDataset(
                    _make_task_dataset(run_config, r, max_test_length, add_positional_offset=False),
                    test_num,
                )
                for r in curriculum.stage_test_ranges(stage_idx)
            }
        )

    return train_dataset, stage_eval_datasets


# ── Formal-language dataset construction ────────────────────────────────────
#
# Per-task corpus specs mirroring formal_lang_suite/configs/dataset/*.yaml, so that
# this on-the-fly pipeline (algorithmic/task_datasets.py's duplicated generators)
# trains/evaluates on the same length windows and corpus sizes as formal_lang_suite
# instead of RunConfig's generic (0, 50) / 3-equal-bins default. `an_star_a2` has no
# formal_lang_suite counterpart and keeps using RunConfig-derived defaults below.
@dataclass(frozen=True)
class _FormalTaskSpec:
    lower_window: int
    upper_window: int
    len_incr: int
    num_bins: int  # total eval bins: bin0 = in-distribution (train window), rest = OOD
    train_size: int
    test_size: int


_FORMAL_TASK_SPECS: dict[str, _FormalTaskSpec] = {
    "tomita_1": _FormalTaskSpec(1, 100, 50, 3, 1000, 50),
    "tomita_2": _FormalTaskSpec(1, 100, 50, 3, 1000, 25),
    "tomita_3": _FormalTaskSpec(1, 100, 50, 3, 10000, 2000),
    "tomita_4": _FormalTaskSpec(1, 100, 50, 3, 10000, 2000),
    "tomita_5": _FormalTaskSpec(1, 100, 50, 4, 10000, 2000),
    "tomita_6": _FormalTaskSpec(1, 100, 50, 4, 10000, 10000),
    "tomita_7": _FormalTaskSpec(1, 100, 50, 3, 10000, 2000),
    "d_2": _FormalTaskSpec(2, 50, 50, 3, 200, 25),
    "d_3": _FormalTaskSpec(2, 50, 50, 3, 10000, 2000),
    "d_4": _FormalTaskSpec(2, 100, 100, 3, 10000, 2000),
    "d_12": _FormalTaskSpec(2, 100, 100, 3, 10000, 2000),
    # formal_lang_suite's "AAStar" dataset (num_par=2); its "AAAAStar" (num_par=4)
    # variant has no corresponding algorithmic task name.
    "aa_star": _FormalTaskSpec(2, 100, 50, 3, 40, 10),
    "abab_star": _FormalTaskSpec(4, 500, 100, 3, 125, 25),
    "aa_star_bb_star": _FormalTaskSpec(5, 200, 100, 3, 10000, 2000),
    "ab_star_d_bc_star": _FormalTaskSpec(2, 50, 50, 3, 10000, 2000),
    "012_star_0_2_star": _FormalTaskSpec(2, 50, 50, 3, 10000, 2000),
}

# `is_formal_task` also accepts un-tabulated `tomita_*`/`d_*` names (which then fall
# back to RunConfig's generic length windows), so the enumerable set is the tabulated
# ones plus `an_star_a2`, the one formal task deliberately left out of the table.
FORMAL_TASKS: tuple[str, ...] = tuple(_FORMAL_TASK_SPECS) + ("an_star_a2",)

ALL_TASKS: tuple[str, ...] = ALGORITHMIC_TASKS + FORMAL_TASKS


def _formal_task_length_ranges(spec: _FormalTaskSpec) -> tuple[tuple[int, int], list[tuple[int, int]]]:
    """bin0 = in-distribution (train) window; later bins grow by `len_incr` each,
    non-overlapping. Mirrors formal_lang_suite's Tomita/D_n/NonStarFree bin
    construction -- deliberately NOT its buggy StarFreeSpecial one (used for
    `ab_star_d_bc_star`/`012_star_0_2_star` there), which never advances
    `lower_window` and so re-includes short, in-distribution lengths in the
    later "OOD" bins.
    """
    train_range = (spec.lower_window, spec.upper_window)
    bins = [train_range]
    lo, hi = train_range
    for _ in range(spec.num_bins - 1):
        lo, hi = hi + 1, hi + spec.len_incr
        bins.append((lo, hi))
    return train_range, bins


def _infer_target_chunk_size(corpora) -> int:
    """How many target characters each source position is labelled with.

    What a position's label means varies by language: prefix membership for `aa_star`
    and `abab_star` (1 char), the set of symbols that may legally follow written as one
    char per alphabet symbol for Tomita 1-4/7 (2) and `ab_star_d_bc_star` (4), and the
    DFA state reached so far for D_n (2). This mirrors `chunk_size` in
    formal_lang_suite's dataset yamls, but is derived from the corpora rather than
    tabulated so it cannot drift out of sync with the generators.
    """
    widths = set()
    for corpus in corpora:
        for src, tgt in zip(corpus.source, corpus.target):
            if not src:
                continue
            if len(tgt) % len(src) != 0:
                raise ValueError(
                    f"target of length {len(tgt)} does not label a source of length {len(src)} "
                    "an equal number of characters per position"
                )
            widths.add(len(tgt) // len(src))
    if len(widths) != 1:
        raise ValueError(f"corpora disagree on target characters per position: {sorted(widths)}")
    return widths.pop()


def _chunk_targets(targets, chunk_size: int) -> list[list[str]]:
    """Group each target into one token per source position.

    Multi-character chunks (e.g. `"10"`) become vocabulary entries in their own
    right, which keeps `FormalLanguageDataset`'s per-token lookup unchanged and
    keeps decoded sequences readable in the summary logs.
    """
    return [
        [tgt[i : i + chunk_size] for i in range(0, len(tgt), chunk_size)]
        for tgt in targets
    ]


def _serialized_length(src, tgt, aligned: bool) -> int:
    if aligned:
        return 1 + len(src) + 1  # <bos> src <eos>
    return 1 + len(src) + 1 + len(tgt) + 1  # <bos> src <sep> tgt <eos>


def _make_tokenizer_and_n_positions(sources_per_split, targets_per_split, aligned: bool):
    """Tokenizer and position budget shared by the train split and every eval bin.

    Sharing both means an OOD bin can never introduce a token (or, for multi-character
    targets, a chunk) the embedding has not seen, nor overflow the position embedding.
    """
    tokens: set[str] = set()
    n_positions = 0
    for sources, targets in zip(sources_per_split, targets_per_split):
        for src, tgt in zip(sources, targets):
            tokens.update(src)
            tokens.update(tgt)
            n_positions = max(n_positions, _serialized_length(src, tgt, aligned))
    return customTokenizer(sorted(tokens)), n_positions


def _build_formal_datasets(run_config: RunConfig, corpus_size_limit: int | None = None):
    task = run_config.task
    spec = _FORMAL_TASK_SPECS.get(task)
    if spec is not None:
        train_length_range, test_length_ranges = _formal_task_length_ranges(spec)
        train_num = spec.train_size
        test_num = spec.test_size
    else:
        train_length_range = run_config.train_length_range
        test_length_ranges = run_config.test_length_ranges
        test_num = run_config.test_num
        train_num = max(4 * test_num, 1000)
    if corpus_size_limit is not None:
        train_num = min(train_num, corpus_size_limit)
        test_num = min(test_num, corpus_size_limit)
    lower_window, upper_window = train_length_range

    if task.startswith("tomita_"):
        n = int(task.split("_")[1])
        train_corpus = TomitaCorpus(n, lower_window, upper_window, train_num, unique=False, leak=True)
        test_bins = [TomitaCorpus(n, r[0], r[1], test_num, unique=True, leak=True) for r in test_length_ranges]
    elif task in _NONSTARFREE_LANG_MAP:
        lang_name, num_par = _NONSTARFREE_LANG_MAP[task]
        train_corpus = NonStarFreeCorpus(lang_name, num_par, lower_window, upper_window, train_num)
        test_bins = [NonStarFreeCorpus(lang_name, num_par, r[0], r[1], test_num, unique=True) for r in test_length_ranges]
    elif task.startswith("d_"):
        n = int(task.split("_")[1])
        train_corpus = StarFreeCorpus("D_n", n, lower_window, upper_window, train_num, unique=False)
        test_bins = [StarFreeCorpus("D_n", n, r[0], r[1], test_num, unique=False) for r in test_length_ranges]
    elif task == "aa_star_bb_star":
        train_corpus = StarFreeCorpus("AAStarBBStar", 5, lower_window, upper_window, train_num, unique=True)
        test_bins = [StarFreeCorpus("AAStarBBStar", 5, r[0], r[1], test_num, unique=True) for r in test_length_ranges]
    elif task == "ab_star_d_bc_star":
        train_corpus = StarFreePostLanguageCorpus("d", "ab", "bc", lower_window, upper_window, train_num)
        test_bins = [StarFreePostLanguageCorpus("d", "ab", "bc", r[0], r[1], test_num) for r in test_length_ranges]
    elif task == "012_star_0_2_star":
        train_corpus = StarFreePostLanguageCorpus("0", "012", "2", lower_window, upper_window, train_num)
        test_bins = [StarFreePostLanguageCorpus("0", "012", "2", r[0], r[1], test_num) for r in test_length_ranges]
    else:
        raise ValueError(f"Unknown formal task {task!r}")

    aligned = run_config.formal_aligned_targets
    if aligned:
        chunk_size = _infer_target_chunk_size([train_corpus, *test_bins])
        train_target = _chunk_targets(train_corpus.target, chunk_size)
        test_targets = [_chunk_targets(bin_corpus.target, chunk_size) for bin_corpus in test_bins]
    else:
        train_target = list(train_corpus.target)
        test_targets = [list(bin_corpus.target) for bin_corpus in test_bins]

    tokenizer, n_positions = _make_tokenizer_and_n_positions(
        [train_corpus.source, *(bin_corpus.source for bin_corpus in test_bins)],
        [train_target, *test_targets],
        aligned,
    )
    train_dataset = FormalLanguageDataset(
        train_corpus.source, train_target, tokenizer, n_positions, aligned=aligned
    )
    test_dataset = {
        f"len{r[0]}-{r[1]}": EvalDataset(
            FormalLanguageDataset(
                bin_corpus.source,
                bin_target,
                tokenizer,
                n_positions,
                add_positional_offset=False,
                aligned=aligned,
            ),
            min(test_num, len(bin_corpus.source)),
        )
        for r, bin_corpus, bin_target in zip(test_length_ranges, test_bins, test_targets)
    }
    return train_dataset, test_dataset, train_length_range, test_length_ranges
