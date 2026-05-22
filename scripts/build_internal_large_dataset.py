#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from build_training_dataset_with_synthetic import prepare_synthetic_df, read_table, write_table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a larger dataset for internal-feature training")
    parser.add_argument("--base-dataset-path", type=str, default="artifacts_v3/gigachat_sberquad_labeling_dataset_with_manual.parquet")
    parser.add_argument("--synthetic-path", type=str, required=True)
    parser.add_argument("--output-path", type=str, default="artifacts_poc/internal_large_v1.parquet")
    parser.add_argument("--base-negative-rows", type=int, default=20000)
    parser.add_argument("--synthetic-rows", type=int, default=6000)
    parser.add_argument("--manual-positive-rows", type=int, default=0)
    parser.add_argument("--manual-negative-rows", type=int, default=0)
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


def sample_or_raise(df: pd.DataFrame, n: int, seed: int, name: str) -> pd.DataFrame:
    if n <= 0:
        return df.head(0).copy()
    if len(df) < n:
        raise RuntimeError(f"Not enough rows for {name}: requested {n}, have {len(df)}")
    return df.sample(n=n, random_state=seed).copy()


def main() -> None:
    args = parse_args()
    base_df = read_table(Path(args.base_dataset_path))
    synthetic_df = read_table(Path(args.synthetic_path))

    base_neg = base_df.loc[base_df["weak_label"].fillna(-1).astype(float) == 0.0].copy()
    if "is_hallucination_manual" in base_neg.columns:
        manual_str = base_neg["is_hallucination_manual"].astype(str).str.lower()
        base_neg = base_neg.loc[~manual_str.isin(["true", "1"])].copy()

    manual_pos = base_df.head(0).copy()
    manual_neg = base_df.head(0).copy()
    if "is_hallucination_manual" in base_df.columns:
        manual_str = base_df["is_hallucination_manual"].astype(str).str.lower()
        manual_pos = base_df.loc[manual_str.isin(["true", "1"])].copy()
        manual_neg = base_df.loc[manual_str.isin(["false", "0"])].copy()

    base_sample = sample_or_raise(base_neg, args.base_negative_rows, args.seed, "base_negative")
    base_sample = assign_split(base_sample, validation_share=args.validation_share, seed=args.seed)
    base_sample["poc_source"] = "base_negative"

    synth_sample = sample_or_raise(synthetic_df, args.synthetic_rows, args.seed + 1, "synthetic_positive")
    synth_prepared = prepare_synthetic_df(base_df=base_df, synthetic_df=synth_sample, synthetic_split="train")
    synth_prepared = assign_split(synth_prepared, validation_share=args.validation_share, seed=args.seed + 1)
    synth_prepared["poc_source"] = "synthetic_positive"

    parts = [base_sample, synth_prepared]

    if args.manual_positive_rows > 0:
        manual_pos_sample = sample_or_raise(manual_pos, args.manual_positive_rows, args.seed + 2, "manual_positive")
        manual_pos_sample = assign_split(manual_pos_sample, validation_share=args.validation_share, seed=args.seed + 2)
        manual_pos_sample["poc_source"] = "manual_positive"
        parts.append(manual_pos_sample)

    if args.manual_negative_rows > 0:
        manual_neg_sample = sample_or_raise(manual_neg, args.manual_negative_rows, args.seed + 3, "manual_negative")
        manual_neg_sample = assign_split(manual_neg_sample, validation_share=args.validation_share, seed=args.seed + 3)
        manual_neg_sample["poc_source"] = "manual_negative"
        parts.append(manual_neg_sample)

    out = pd.concat(parts, ignore_index=True, sort=False)
    out = out.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    write_table(out, Path(args.output_path))

    print(f"saved: {args.output_path}")
    print(f"rows: {len(out)}")
    print("label summary:")
    if "is_hallucination_manual" in out.columns:
        manual_str = out["is_hallucination_manual"].astype(str).str.lower()
        label_proxy = pd.Series(pd.NA, index=out.index, dtype='object')
        label_proxy.loc[out["weak_label"].fillna(-1).astype(float) == 0.0] = "negative"
        label_proxy.loc[out["weak_label"].fillna(-1).astype(float) == 1.0] = "positive"
        label_proxy.loc[manual_str.isin(["false", "0"])] = "negative"
        label_proxy.loc[manual_str.isin(["true", "1"])] = "positive"
        print(label_proxy.value_counts(dropna=False).to_string())
    print("source summary:")
    print(out["poc_source"].value_counts(dropna=False).to_string())
    print("split summary:")
    print(out.groupby(["split_name", "poc_source"]).size().to_string())


if __name__ == "__main__":
    main()
