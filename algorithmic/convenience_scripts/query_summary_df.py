#!/usr/bin/env python3
"""Query summary CSV via pandas and output resulting datapoints across tasks."""

from __future__ import annotations

import argparse
from pathlib import Path

from dataframe_query_utils import (
    apply_keep_remove_filters,
    apply_query_filters,
    require_columns,
)
from plot_utils import load_summary_dataframe


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Query summary CSV with DataFrame filters. "
            "Outputs rows with task/spec columns, bins, and accuracies."
        )
    )
    repo_root = Path(__file__).resolve().parents[2]
    default_csv = repo_root / "exports" / "all_summary_results.csv"

    parser.add_argument("--input-csv", "--csv", dest="input_csv", type=Path, default=default_csv)
    parser.add_argument(
        "--keep",
        action="append",
        default=[],
        help="Keep rows by exact column values: column=v1,v2 (repeatable).",
    )
    parser.add_argument(
        "--remove",
        action="append",
        default=[],
        help="Drop rows by exact column values: column=v1,v2 (repeatable).",
    )
    parser.add_argument(
        "--query",
        action="append",
        default=[],
        help="Keep rows matching pandas query expression (repeatable, AND across uses).",
    )
    parser.add_argument(
        "--exclude-query",
        action="append",
        default=[],
        help="Drop rows matching pandas query expression (repeatable).",
    )
    parser.add_argument(
        "--datapoint-cols",
        action="append",
        default=[],
        help=(
            "Columns that define one datapoint (repeatable). "
            "Default: task,task-spec-columns,model,learning_rate,bucket,accuracy"
        ),
    )
    parser.add_argument(
        "--show-cols",
        action="append",
        default=[],
        help=(
            "Additional columns to include in the output (repeatable). "
            "If omitted, only datapoint columns are shown."
        ),
    )
    args = parser.parse_args()

    df = load_summary_dataframe(args.input_csv)
    df = apply_keep_remove_filters(df, args.keep, args.remove)
    df = apply_query_filters(df, args.query, args.exclude_query)
    if df.empty:
        raise SystemExit("No rows left after DataFrame filtering.")

    # default_datapoint_cols = ["task", *TASK_PARAM_COLUMNS, "model", "learning_rate", "bucket", "accuracy"]
    default_datapoint_cols = ["task", "model", "num_bins", "train_range", "bucket", "accuracy"]
    datapoint_cols = args.datapoint_cols or [c for c in default_datapoint_cols if c in df.columns]
    require_columns(df, datapoint_cols, "--datapoint-cols")
    require_columns(df, args.show_cols, "--show-cols")

    out_cols: list[str] = []
    for c in [*datapoint_cols, *args.show_cols]:
        if c not in out_cols:
            out_cols.append(c)

    sort_cols = [c for c in ("task", "bucket", "model", "learning_rate", "accuracy") if c in out_cols]
    if not sort_cols:
        sort_cols = list(datapoint_cols)
    out_df = df[out_cols].drop_duplicates().sort_values(sort_cols).reset_index(drop=True)

    # if args.output_csv is not None:
    #     args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    #     out_df.to_csv(args.output_csv, index=False)
    #     print(f"Wrote datapoints CSV: {args.output_csv} ({len(out_df)} rows)")
    # else:
    print(out_df.to_string(index=False))
    print(f"\nDatapoints: {len(out_df)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
