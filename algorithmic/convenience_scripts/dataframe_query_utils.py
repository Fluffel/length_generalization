#!/usr/bin/env python3
"""Shared DataFrame filtering helpers for convenience scripts."""

from __future__ import annotations


def parse_col_values(raw: str) -> tuple[str, list[str]]:
    if "=" not in raw:
        raise SystemExit(
            f"Invalid filter {raw!r}. Expected format: column=value1,value2"
        )
    col, vals = raw.split("=", 1)
    col = col.strip()
    values = [v.strip() for v in vals.split(",") if v.strip()]
    if not col:
        raise SystemExit(f"Invalid filter {raw!r}: empty column name.")
    if not values:
        raise SystemExit(f"Invalid filter {raw!r}: provide at least one value.")
    return col, values


def require_columns(df, cols: list[str], flag_name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise SystemExit(
            f"{flag_name}: unknown column(s): {', '.join(sorted(missing))}. "
            f"Available columns: {', '.join(df.columns)}"
        )


def apply_keep_remove_filters(df, keeps: list[str], removes: list[str]):
    for raw in keeps:
        col, vals = parse_col_values(raw)
        require_columns(df, [col], "--keep")
        df = df[df[col].astype(str).isin(vals)]
    for raw in removes:
        col, vals = parse_col_values(raw)
        require_columns(df, [col], "--remove")
        df = df[~df[col].astype(str).isin(vals)]
    return df


def apply_query_filters(df, queries: list[str], excludes: list[str]):
    for q in queries:
        q = q.strip()
        if not q:
            continue
        df = df.query(q, engine="python")
    for q in excludes:
        q = q.strip()
        if not q:
            continue
        idx = df.query(q, engine="python").index
        df = df.drop(index=idx)
    return df
