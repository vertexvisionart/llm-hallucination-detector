#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from datasets import load_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build factual QA train candidates from public Russian QA datasets")
    parser.add_argument(
        "--output-path",
        type=str,
        default="artifacts_poc/factual_qa_train_candidates.parquet",
    )
    parser.add_argument("--max-answer-chars", type=int, default=40)
    parser.add_argument("--max-answer-words", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def normalize_whitespace(text: str) -> str:
    return " ".join(str(text or "").split()).strip()


def prepare_rubq() -> pd.DataFrame:
    ds = load_dataset("d0rj/RuBQ_2.0")
    parts = []
    for split_name, split_ds in ds.items():
        df = split_ds.to_pandas().copy()
        df["question"] = df["question_text"].map(normalize_whitespace)
        df["gold_answer"] = df["answer_text"].map(normalize_whitespace)
        df["gold_answers_text"] = df["gold_answer"].map(lambda x: [x] if x else [])
        df["source_dataset"] = "d0rj/RuBQ_2.0"
        df["source_split"] = split_name
        df["source_id"] = df["uid"].astype(str)
        parts.append(df[["question", "gold_answer", "gold_answers_text", "source_dataset", "source_split", "source_id"]])
    return pd.concat(parts, ignore_index=True)


def prepare_russian_facts(max_answer_chars: int, max_answer_words: int) -> pd.DataFrame:
    ds = load_dataset("Expotion/russian-facts-qa")
    df = ds["train"].to_pandas().copy()
    df["question"] = df["q"].map(normalize_whitespace)
    df["gold_answer"] = df["a"].map(normalize_whitespace)
    answer_chars = df["gold_answer"].str.len()
    answer_words = df["gold_answer"].str.split().str.len()
    df = df.loc[(answer_chars <= max_answer_chars) & (answer_words <= max_answer_words)].copy()
    df["gold_answers_text"] = df["gold_answer"].map(lambda x: [x] if x else [])
    df["source_dataset"] = "Expotion/russian-facts-qa"
    df["source_split"] = "train"
    df["source_id"] = df.index.astype(str)
    return df[["question", "gold_answer", "gold_answers_text", "source_dataset", "source_split", "source_id"]]


def main() -> None:
    args = parse_args()

    rubq = prepare_rubq()
    facts = prepare_russian_facts(
        max_answer_chars=args.max_answer_chars,
        max_answer_words=args.max_answer_words,
    )

    out = pd.concat([rubq, facts], ignore_index=True)
    out["question"] = out["question"].map(normalize_whitespace)
    out["gold_answer"] = out["gold_answer"].map(normalize_whitespace)
    out = out.loc[out["question"].astype(bool) & out["gold_answer"].astype(bool)].copy()
    out = out.drop_duplicates(subset=["question", "gold_answer"]).reset_index(drop=True)

    out["id"] = [
        f"factual_{idx}_{src}_{sid}"
        for idx, (src, sid) in enumerate(zip(out["source_dataset"], out["source_id"]), start=1)
    ]
    out["split_name"] = "train"
    out = out[
        [
            "id",
            "split_name",
            "question",
            "gold_answer",
            "gold_answers_text",
            "source_dataset",
            "source_split",
            "source_id",
        ]
    ]

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(output_path, index=False)

    print(f"saved: {output_path}")
    print(f"rows: {len(out)}")
    print(out["source_dataset"].value_counts(dropna=False).to_string())
    print(out.head(5).to_dict(orient="records"))


if __name__ == "__main__":
    main()
