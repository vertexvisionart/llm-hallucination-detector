#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm.auto import tqdm

from generate_answers_from_sberchallenge import (
    MODEL_ID,
    build_prompt,
    generate_answer,
    load_model_and_tokenizer,
)
from prepare_hallucination_labeling import (
    answer_supported_by_context,
    best_gold_metrics,
    ensure_list,
    normalize_text,
)


HARD_PATTERN = re.compile(r"\d{4}|\d+[.,]\d+|\d+ (?:года|лет|км|млн|млрд|тысяч|человек)", re.IGNORECASE)
QUESTION_HARD_PATTERN = re.compile(
    r"(?:сколько|когда|какой год|в каком|кто основал|кто первый|сколько раз|какова|каков)",
    re.IGNORECASE,
)
CONTEXT_LONG_NUMBER_PATTERN = re.compile(r"\d{4,}")
GOLD_PROPER_NAME_PATTERN = re.compile(r"[А-ЯЁ][а-яё]+ [А-ЯЁ][а-яё]+")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate real positives from hard train rows")
    parser.add_argument("--dataset-path", type=str, default="artifacts_v3/gigachat_sberquad_labeling_dataset_with_manual.parquet")
    parser.add_argument("--output-path", type=str, default="artifacts_poc/real_positives_v1.parquet")
    parser.add_argument("--exclude-id-paths", nargs="*", default=None)
    parser.add_argument("--cache-dir", type=str, default="hf_cache")
    parser.add_argument("--model-id", type=str, default=MODEL_ID)
    parser.add_argument("--limit", type=int, default=5000)
    parser.add_argument("--save-every", type=int, default=20)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--prompt-max-length", type=int, default=4096)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def save_rows(rows: list[dict[str, Any]], save_path: Path) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(save_path, index=False)


def gold_contains_hard_pattern(gold_answers: list[str]) -> bool:
    joined = " | ".join(str(x) for x in gold_answers if str(x).strip())
    return bool(HARD_PATTERN.search(joined))


def question_matches_hard_pattern(question: str) -> bool:
    return bool(QUESTION_HARD_PATTERN.search(str(question or "")))


def context_contains_long_number(context: str) -> bool:
    return bool(CONTEXT_LONG_NUMBER_PATTERN.search(str(context or "")))


def gold_contains_proper_name(gold_answers: list[str]) -> bool:
    joined = " | ".join(str(x) for x in gold_answers if str(x).strip())
    return bool(GOLD_PROPER_NAME_PATTERN.search(joined))


def row_is_hard(row_s: pd.Series) -> bool:
    gold_answers = ensure_list(row_s.get("gold_answers_text"))
    return any(
        [
            gold_contains_hard_pattern(gold_answers),
            question_matches_hard_pattern(row_s.get("question", "")),
            context_contains_long_number(row_s.get("context", "")),
            gold_contains_proper_name(gold_answers),
        ]
    )


def load_excluded_ids(paths: list[str] | None) -> set[str]:
    excluded_ids: set[str] = set()
    for raw_path in paths or []:
        path = Path(raw_path)
        if not path.exists():
            print(f"warn: exclude path not found, skip: {path}")
            continue
        df = pd.read_parquet(path) if path.suffix.lower() == ".parquet" else pd.read_csv(path)
        if "id" not in df.columns:
            print(f"warn: exclude path has no id column, skip: {path}")
            continue
        ids = df["id"].dropna().astype(str)
        excluded_ids.update(ids.tolist())
        print(f"loaded exclude ids: {path} -> {len(ids)} rows")
    return excluded_ids


def build_output_row(row_s: pd.Series, prompt_text: str, model_answer: str) -> dict[str, Any] | None:
    gold_answers = ensure_list(row_s.get("gold_answers_text"))
    normalized_model_answer = normalize_text(model_answer)
    normalized_gold_answers = [normalize_text(answer) for answer in gold_answers]
    exact_match_any = normalized_model_answer in set(normalized_gold_answers)

    metrics = best_gold_metrics(model_answer, gold_answers)
    supported = answer_supported_by_context(model_answer, row_s.get("context", ""))

    is_hallucination: int | None
    if metrics["token_f1_max"] < 0.25 and not supported and len(model_answer.strip()) > 10:
        is_hallucination = 1
    elif metrics["token_f1_max"] > 0.7 or exact_match_any:
        is_hallucination = 0
    else:
        return None

    return {
        "id": row_s.get("id"),
        "question": row_s.get("question", ""),
        "context": row_s.get("context", ""),
        "gold_answers_text": gold_answers,
        "model_answer": model_answer,
        "prompt_text": prompt_text,
        "is_hallucination": is_hallucination,
        "token_f1_max": metrics["token_f1_max"],
        "char_similarity_max": metrics["char_similarity_max"],
        "model_answer_supported_by_context": int(bool(supported)),
        "source": "real_generated",
    }


def main() -> None:
    args = parse_args()
    dataset_path = Path(args.dataset_path)
    output_path = Path(args.output_path)

    df = pd.read_parquet(dataset_path)
    if "gold_answers_text" not in df.columns:
        raise RuntimeError("dataset missing gold_answers_text")

    df = df.copy()
    df["gold_answers_text"] = df["gold_answers_text"].apply(ensure_list)
    df = df.loc[df["split_name"] == "train"].copy()

    old_mask = df["gold_answers_text"].apply(gold_contains_hard_pattern)
    expanded_mask = df.apply(row_is_hard, axis=1)

    old_candidates = int(old_mask.sum())
    expanded_candidates = int(expanded_mask.sum())
    print(f"old hard train candidates: {old_candidates}")
    print(f"expanded hard train candidates: {expanded_candidates}")
    print(f"new candidates added by expanded filter: {expanded_candidates - old_candidates}")

    df = df.loc[expanded_mask].copy()

    excluded_ids = load_excluded_ids(args.exclude_id_paths)
    if excluded_ids:
        before_exclude = len(df)
        df = df.loc[~df["id"].astype(str).isin(excluded_ids)].copy()
        print(f"remaining candidates after exclude ids: {len(df)} (removed {before_exclude - len(df)})")

    if df.empty:
        raise RuntimeError("no hard train rows found")

    sample_n = min(args.limit, len(df))
    df = df.sample(n=sample_n, random_state=args.seed).reset_index(drop=True)

    total_rows = len(df)
    rows: list[dict[str, Any]] = []
    start_idx = 0

    if output_path.exists() and not args.overwrite:
        existing_df = pd.read_parquet(output_path)
        existing_count = len(existing_df)
        if existing_count >= total_rows:
            print(f"skip: already exists complete output {output_path}")
            return
        rows = existing_df.to_dict(orient="records")
        start_idx = existing_count
        print(f"resume: {output_path} from row {start_idx}/{total_rows}")
    else:
        print(f"processing: {dataset_path}")
        print(f"hard train candidates sampled: {total_rows}")

    model, tokenizer = load_model_and_tokenizer(args.model_id, Path(args.cache_dir))

    iterator_df = df.iloc[start_idx:] if start_idx > 0 else df
    processed_since_save = 0
    kept_rows = len(rows)

    for row in tqdm(iterator_df.itertuples(index=False), total=len(iterator_df), initial=0):
        row_s = pd.Series(row._asdict())
        prompt_text = build_prompt(row_s)
        model_answer = generate_answer(
            prompt=prompt_text,
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            do_sample=args.do_sample,
            prompt_max_length=args.prompt_max_length,
        )

        out_row = build_output_row(row_s, prompt_text, model_answer)
        if out_row is not None:
            rows.append(out_row)
            kept_rows += 1

        processed_since_save += 1
        if args.save_every > 0 and processed_since_save >= args.save_every:
            save_rows(rows, output_path)
            print(f"checkpoint saved: {output_path} (processed={start_idx + processed_since_save}/{total_rows}, kept={kept_rows})")
            processed_since_save = 0

    save_rows(rows, output_path)
    print(f"saved: {output_path} (processed={total_rows}/{total_rows}, kept={len(rows)})")


if __name__ == "__main__":
    main()
