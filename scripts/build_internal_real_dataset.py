#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


BOOLISH_COLUMNS = [
    'model_answer_supported_by_context',
    'exact_match_any',
    'substring_match_any',
    'has_uncertainty_marker',
    'needs_manual_review',
    'is_hallucination_heuristic',
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build internal dataset from real-generated rows plus base negatives")
    parser.add_argument("--real-path", type=str, required=True)
    parser.add_argument("--base-negative-path", type=str, default="artifacts_poc/internal_large_v1.parquet")
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("--target-negative-positive-ratio", type=float, default=4.0)
    parser.add_argument("--validation-share", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def normalize_boolish_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in BOOLISH_COLUMNS:
        if col not in out.columns:
            continue
        numeric = pd.to_numeric(out[col], errors="coerce")
        if numeric.notna().any():
            out[col] = numeric
            continue
        lowered = out[col].astype(str).str.strip().str.lower()
        out[col] = lowered.map({"true": 1, "false": 0, "1": 1, "0": 0})
    return out


def normalize_identifier_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ["id", "source_id"]:
        if col not in out.columns:
            continue
        out[col] = out[col].where(out[col].isna(), out[col].astype(str))
    return out


def add_real_generated_labels(df: pd.DataFrame) -> pd.DataFrame:
    out = normalize_boolish_columns(df)
    out["is_hallucination"] = pd.to_numeric(out["is_hallucination"], errors="coerce")
    out = out.loc[out["is_hallucination"].isin([0, 1])].copy()
    out["weak_label"] = out["is_hallucination"]
    out["weak_label_reason"] = "real_generated_auto"
    out["source"] = "real_generated"
    return out


def select_base_negatives(df: pd.DataFrame) -> pd.DataFrame:
    out = normalize_boolish_columns(df)
    if "weak_label" in out.columns:
        out["is_hallucination"] = pd.to_numeric(out["weak_label"], errors="coerce")
    elif "is_hallucination" in out.columns:
        out["is_hallucination"] = pd.to_numeric(out["is_hallucination"], errors="coerce")
    else:
        raise RuntimeError("base negative dataset must contain weak_label or is_hallucination")
    out = out.loc[out["is_hallucination"] == 0].copy()
    out["source"] = "base_negative"
    return out


def assign_stratified_split(df: pd.DataFrame, validation_share: float, seed: int) -> pd.DataFrame:
    labels = pd.to_numeric(df["is_hallucination"], errors="coerce")
    if labels.isna().any():
        raise RuntimeError("dataset contains rows without is_hallucination labels")
    _, valid_idx = train_test_split(
        df.index,
        test_size=validation_share,
        random_state=seed,
        stratify=labels.astype(int),
    )
    out = df.copy()
    out["split_name"] = "train"
    out.loc[valid_idx, "split_name"] = "validation"
    out["is_hallucination"] = labels.astype(int)
    return out.reset_index(drop=True)


def main() -> None:
    args = parse_args()
    real_df = pd.read_parquet(Path(args.real_path))
    base_df = pd.read_parquet(Path(args.base_negative_path))

    real_df = add_real_generated_labels(real_df)
    base_df = select_base_negatives(base_df)

    positive_count = int((real_df["is_hallucination"] == 1).sum())
    negative_count = int((real_df["is_hallucination"] == 0).sum())
    target_negative_count = int(round(positive_count * args.target_negative_positive_ratio))
    extra_negative_needed = max(0, target_negative_count - negative_count)

    if extra_negative_needed > len(base_df):
        raise RuntimeError(
            f"Requested {extra_negative_needed} base negatives, but only {len(base_df)} are available"
        )

    if extra_negative_needed > 0:
        base_df = base_df.sample(n=extra_negative_needed, random_state=args.seed).copy()
    else:
        base_df = base_df.head(0).copy()

    out = pd.concat([real_df, base_df], ignore_index=True, sort=False)
    out = normalize_boolish_columns(out)
    out = normalize_identifier_columns(out)
    out = assign_stratified_split(out, validation_share=args.validation_share, seed=args.seed)

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(output_path, index=False)

    label_counts = out["is_hallucination"].value_counts(dropna=False).to_dict()
    split_label_counts = (
        out.groupby(["split_name", "is_hallucination"]).size().rename("rows").reset_index().to_dict(orient="records")
    )
    source_counts = out["source"].value_counts(dropna=False).to_dict() if "source" in out.columns else {}

    print(f"saved: {output_path}")
    print(f"rows: {len(out)}")
    print(f"label_counts: {label_counts}")
    print(f"source_counts: {source_counts}")
    print(f"split_label_counts: {split_label_counts}")


if __name__ == "__main__":
    main()
