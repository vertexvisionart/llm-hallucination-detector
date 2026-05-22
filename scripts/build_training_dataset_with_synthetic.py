#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


REQUIRED_SYNTHETIC_COLUMNS = [
    "source_id",
    "split_name",
    "title",
    "question",
    "context",
    "gold_answers_text",
    "best_gold_answer",
    "model_answer",
    "hallucination_type",
    "label",
    "synthetic_source",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build training dataset with synthetic hallucinations")
    parser.add_argument(
        "--base-dataset-path",
        type=str,
        default="artifacts_v3/gigachat_sberquad_labeling_dataset_with_manual.parquet",
    )
    parser.add_argument(
        "--synthetic-path",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="artifacts_v3/gigachat_sberquad_training_with_synthetic.parquet",
    )
    parser.add_argument(
        "--synthetic-split",
        type=str,
        default="train",
        help="Какой split_name оставить у synthetic строк.",
    )
    parser.add_argument(
        "--max-synthetic-rows",
        type=int,
        default=None,
    )
    return parser.parse_args()



def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported format: {path}")



def write_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".parquet":
        df.to_parquet(path, index=False)
    elif path.suffix.lower() == ".csv":
        df.to_csv(path, index=False)
    else:
        raise ValueError(f"Unsupported format: {path}")



def prepare_synthetic_df(base_df: pd.DataFrame, synthetic_df: pd.DataFrame, synthetic_split: str) -> pd.DataFrame:
    missing = [col for col in REQUIRED_SYNTHETIC_COLUMNS if col not in synthetic_df.columns]
    if missing:
        raise RuntimeError(f"Synthetic file missing columns: {missing}")

    template_columns = list(base_df.columns)
    out = pd.DataFrame(columns=template_columns)
    out = out.reindex(range(len(synthetic_df)))

    for col in template_columns:
        if col in synthetic_df.columns:
            out[col] = synthetic_df[col].values
        else:
            out[col] = pd.NA

    synthetic_ids = [f"synthetic_{row.source_id}_{idx}" for idx, row in enumerate(synthetic_df.itertuples(index=False), start=1)]
    if "id" in base_df.columns:
        try:
            pd.to_numeric(base_df["id"])
            out["id"] = range(-1, -len(synthetic_ids) - 1, -1)
            out["source_id_original"] = synthetic_ids
        except Exception:
            out["id"] = synthetic_ids
    else:
        out["id"] = synthetic_ids

    out["split_name"] = synthetic_split
    out["is_hallucination_manual"] = True
    out["reviewer_comment"] = (
        "synthetic_hallucination: " + synthetic_df["hallucination_type"].astype(str)
    )
    out["weak_label"] = 1
    out["weak_label_reason"] = "synthetic_hallucination"
    out["needs_manual_review"] = False
    out["is_hallucination_heuristic"] = True
    out["source_file"] = synthetic_df.get("synthetic_source", "").astype(str)
    out["generated_file"] = synthetic_df.get("synthetic_source", "").astype(str)
    out["prompt_text"] = synthetic_df.get("generator_prompt", "")
    return out



def main() -> None:
    args = parse_args()
    base_path = Path(args.base_dataset_path)
    synthetic_path = Path(args.synthetic_path)
    output_path = Path(args.output_path)

    base_df = read_table(base_path)
    synthetic_df = read_table(synthetic_path)

    if args.max_synthetic_rows is not None:
        synthetic_df = synthetic_df.head(args.max_synthetic_rows).copy()

    prepared_synth = prepare_synthetic_df(base_df, synthetic_df, synthetic_split=args.synthetic_split)
    merged = pd.concat([base_df, prepared_synth], ignore_index=True, sort=False)

    write_table(merged, output_path)

    print(f"saved: {output_path}")
    print(f"base_rows: {len(base_df)}")
    print(f"synthetic_rows: {len(prepared_synth)}")
    print(f"total_rows: {len(merged)}")
    if "weak_label_reason" in merged.columns:
        print("synthetic weak_label_reason count:")
        print(merged["weak_label_reason"].value_counts(dropna=False).head(10).to_string())


if __name__ == "__main__":
    main()
