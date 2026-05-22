#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from build_training_dataset_with_synthetic import prepare_synthetic_df, read_table, write_table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a small PoC dataset for cheap vs internal feature A/B testing")
    parser.add_argument(
        "--base-dataset-path",
        type=str,
        default="artifacts_v3/gigachat_sberquad_labeling_dataset_with_manual.parquet",
    )
    parser.add_argument(
        "--synthetic-path",
        type=str,
        default="synthetic_hallucinations_v2.parquet",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="artifacts_poc/internal_poc_dataset.parquet",
    )
    parser.add_argument("--base-negative-rows", type=int, default=1500)
    parser.add_argument("--synthetic-rows", type=int, default=500)
    parser.add_argument("--validation-share", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def assign_split(df: pd.DataFrame, validation_share: float, seed: int) -> pd.DataFrame:
    if df.empty:
        out = df.copy()
        out["split_name"] = []
        return out

    out = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    valid_n = max(1, int(round(len(out) * validation_share))) if len(out) > 1 else 0
    out["split_name"] = "train"
    if valid_n > 0:
        out.loc[out.index[:valid_n], "split_name"] = "validation"
    return out


def main() -> None:
    args = parse_args()
    base_df = read_table(Path(args.base_dataset_path))
    synthetic_df = read_table(Path(args.synthetic_path))

    base_neg = base_df.loc[base_df["weak_label"].fillna(-1).astype(float) == 0.0].copy()
    if "is_hallucination_manual" in base_neg.columns:
        manual_str = base_neg["is_hallucination_manual"].astype(str).str.lower()
        base_neg = base_neg.loc[~manual_str.isin(["true", "1"])].copy()

    if len(base_neg) < args.base_negative_rows:
        raise RuntimeError(f"Not enough base negatives: requested {args.base_negative_rows}, have {len(base_neg)}")
    if len(synthetic_df) < args.synthetic_rows:
        raise RuntimeError(f"Not enough synthetic rows: requested {args.synthetic_rows}, have {len(synthetic_df)}")

    base_sample = base_neg.sample(n=args.base_negative_rows, random_state=args.seed).copy()
    base_sample = assign_split(base_sample, validation_share=args.validation_share, seed=args.seed)
    base_sample["poc_source"] = "base_negative"

    synthetic_sample = synthetic_df.sample(n=args.synthetic_rows, random_state=args.seed).copy()
    synthetic_prepared = prepare_synthetic_df(base_df=base_df, synthetic_df=synthetic_sample, synthetic_split="train")
    synthetic_prepared = assign_split(synthetic_prepared, validation_share=args.validation_share, seed=args.seed + 1)
    synthetic_prepared["poc_source"] = "synthetic_positive"

    out = pd.concat([base_sample, synthetic_prepared], ignore_index=True, sort=False)
    out = out.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    write_table(out, Path(args.output_path))

    print(f"saved: {args.output_path}")
    print(f"rows: {len(out)}")
    print("label summary:")
    print(out["poc_source"].value_counts(dropna=False).to_string())
    print("split summary:")
    print(out.groupby(["split_name", "poc_source"]).size().to_string())


if __name__ == "__main__":
    main()
