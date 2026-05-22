#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from build_training_dataset_with_synthetic import prepare_synthetic_df, read_table, write_table


DEFAULT_HARD_NEGATIVE_REASONS = ",".join(
    [
        "low_similarity_unsupported",
        "paraphrase_or_inflection",
        "high_char_similarity",
        "single_token_inflection",
        "high_token_f1",
        "high_content_f1",
        "content_overlap_paraphrase",
    ]
)

DEFAULT_SYNTHETIC_TYPES = "cause_fact_swap,false_refusal,entity_swap"
DEFAULT_EASY_NEGATIVE_REASONS = "exact_match,substring_match"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build datasets for internal-feature experiments")
    parser.add_argument("--mode", type=str, choices=["full", "hardneg"], required=True)
    parser.add_argument(
        "--base-dataset-path",
        type=str,
        default="artifacts_v3/gigachat_sberquad_labeling_dataset_with_manual.parquet",
    )
    parser.add_argument("--synthetic-path", type=str, default="synthetic_hallucinations_v2.parquet")
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("--synthetic-types", type=str, default=DEFAULT_SYNTHETIC_TYPES)
    parser.add_argument("--max-synthetic-rows", type=int, default=None)
    parser.add_argument("--dedup-synthetic", action="store_true")
    parser.add_argument("--validation-share", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hard-negative-reasons", type=str, default=DEFAULT_HARD_NEGATIVE_REASONS)
    parser.add_argument("--easy-negative-reasons", type=str, default=DEFAULT_EASY_NEGATIVE_REASONS)
    parser.add_argument("--easy-negative-rows", type=int, default=8000)
    return parser.parse_args()


def parse_csv_set(raw: str) -> set[str]:
    return {part.strip() for part in str(raw).split(",") if part.strip()}


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


def sample_total_by_split(df: pd.DataFrame, total_rows: int, seed: int) -> pd.DataFrame:
    if total_rows <= 0 or df.empty:
        return df.head(0).copy()
    if total_rows >= len(df):
        return df.copy()
    if "split_name" not in df.columns:
        return df.sample(n=total_rows, random_state=seed).copy()

    counts = df["split_name"].value_counts()
    raw_targets = {split: total_rows * count / len(df) for split, count in counts.items()}
    targets = {split: min(int(raw_targets[split]), int(counts[split])) for split in counts.index}
    assigned = sum(targets.values())
    remainders = sorted(
        ((raw_targets[split] - targets[split], split) for split in counts.index),
        reverse=True,
    )
    for _, split in remainders:
        if assigned >= total_rows:
            break
        if targets[split] < counts[split]:
            targets[split] += 1
            assigned += 1

    parts = []
    for offset, split in enumerate(counts.index):
        n = targets.get(split, 0)
        if n <= 0:
            continue
        split_df = df.loc[df["split_name"] == split]
        parts.append(split_df.sample(n=n, random_state=seed + offset).copy())
    return pd.concat(parts, ignore_index=True, sort=False)


def add_effective_label(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["effective_label"] = pd.NA
    if "weak_label" in out.columns:
        weak = pd.to_numeric(out["weak_label"], errors="coerce")
        out.loc[weak == 0.0, "effective_label"] = 0
        out.loc[weak == 1.0, "effective_label"] = 1
    if "is_hallucination_manual" in out.columns:
        manual_str = out["is_hallucination_manual"].astype(str).str.lower()
        out.loc[manual_str.isin(["false", "0"]), "effective_label"] = 0
        out.loc[manual_str.isin(["true", "1"]), "effective_label"] = 1
    return out


def load_filtered_synthetic(
    base_df: pd.DataFrame,
    synthetic_path: Path,
    synthetic_types: set[str],
    max_synthetic_rows: int | None,
    dedup_synthetic: bool,
    validation_share: float,
    seed: int,
) -> pd.DataFrame:
    synthetic_df = read_table(synthetic_path)
    if synthetic_types:
        synthetic_df = synthetic_df.loc[synthetic_df["hallucination_type"].isin(sorted(synthetic_types))].copy()
    if dedup_synthetic and "model_answer" in synthetic_df.columns:
        synthetic_df["__dedup_answer"] = synthetic_df["model_answer"].fillna("").astype(str).str.strip().str.lower()
        synthetic_df = synthetic_df.drop_duplicates(subset=["hallucination_type", "__dedup_answer"]).copy()
        synthetic_df = synthetic_df.drop(columns="__dedup_answer")
    if max_synthetic_rows is not None:
        synthetic_df = synthetic_df.head(max_synthetic_rows).copy()
    prepared = prepare_synthetic_df(base_df=base_df, synthetic_df=synthetic_df, synthetic_split="train")
    prepared = assign_split(prepared, validation_share=validation_share, seed=seed)
    prepared["dataset_source"] = "synthetic_positive"
    return prepared


def build_full_dataset(args: argparse.Namespace) -> pd.DataFrame:
    base_df = read_table(Path(args.base_dataset_path)).copy()
    base_df["dataset_source"] = "base"
    synthetic_df = load_filtered_synthetic(
        base_df=base_df,
        synthetic_path=Path(args.synthetic_path),
        synthetic_types=parse_csv_set(args.synthetic_types),
        max_synthetic_rows=args.max_synthetic_rows,
        dedup_synthetic=args.dedup_synthetic,
        validation_share=args.validation_share,
        seed=args.seed,
    )
    out = pd.concat([base_df, synthetic_df], ignore_index=True, sort=False)
    return out.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)


def build_hardneg_dataset(args: argparse.Namespace) -> pd.DataFrame:
    base_df = add_effective_label(read_table(Path(args.base_dataset_path)))
    hard_negative_reasons = parse_csv_set(args.hard_negative_reasons)
    easy_negative_reasons = parse_csv_set(args.easy_negative_reasons)

    positives = base_df.loc[base_df["effective_label"] == 1].copy()
    positives["dataset_source"] = "base_positive"

    hard_negatives = base_df.loc[
        (base_df["effective_label"] == 0)
        & (base_df["weak_label_reason"].fillna("").isin(sorted(hard_negative_reasons)))
    ].copy()
    hard_negatives["dataset_source"] = "hard_negative"

    manual_negatives = base_df.loc[
        base_df["is_hallucination_manual"].astype(str).str.lower().isin(["false", "0"])
    ].copy()
    manual_negatives["dataset_source"] = "manual_negative"

    easy_pool = base_df.loc[
        (base_df["effective_label"] == 0)
        & (base_df["weak_label_reason"].fillna("").isin(sorted(easy_negative_reasons)))
    ].copy()
    easy_negatives = sample_total_by_split(easy_pool, total_rows=args.easy_negative_rows, seed=args.seed + 10)
    easy_negatives["dataset_source"] = "easy_negative"

    base_subset = pd.concat(
        [positives, hard_negatives, manual_negatives, easy_negatives],
        ignore_index=True,
        sort=False,
    )
    if "id" in base_subset.columns:
        base_subset = base_subset.drop_duplicates(subset=["id"], keep="first").copy()

    synthetic_df = load_filtered_synthetic(
        base_df=base_df.drop(columns=["effective_label"]),
        synthetic_path=Path(args.synthetic_path),
        synthetic_types=parse_csv_set(args.synthetic_types),
        max_synthetic_rows=args.max_synthetic_rows,
        dedup_synthetic=args.dedup_synthetic,
        validation_share=args.validation_share,
        seed=args.seed + 20,
    )

    out = pd.concat([base_subset.drop(columns=["effective_label"], errors="ignore"), synthetic_df], ignore_index=True, sort=False)
    return out.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)


def print_summary(df: pd.DataFrame) -> None:
    print(f"rows: {len(df)}")
    if "dataset_source" in df.columns:
        print("source summary:")
        print(df["dataset_source"].value_counts(dropna=False).to_string())
    if "split_name" in df.columns:
        print("split summary:")
        print(df["split_name"].value_counts(dropna=False).to_string())

    labeled = add_effective_label(df)
    if "effective_label" in labeled.columns:
        print("label summary:")
        printable = labeled["effective_label"].map({0: "negative", 1: "positive"}).fillna("unlabeled")
        print(printable.value_counts(dropna=False).to_string())

    if "weak_label_reason" in df.columns:
        print("weak_label_reason top:")
        print(df["weak_label_reason"].fillna("NA").value_counts(dropna=False).head(12).to_string())


def main() -> None:
    args = parse_args()
    if args.mode == "full":
        out = build_full_dataset(args)
    else:
        out = build_hardneg_dataset(args)

    write_table(out, Path(args.output_path))
    print(f"saved: {args.output_path}")
    print_summary(out)


if __name__ == "__main__":
    main()
