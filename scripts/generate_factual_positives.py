#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm.auto import tqdm

from generate_answers_from_sberchallenge import MODEL_ID, generate_answer, load_model_and_tokenizer
from prepare_hallucination_labeling import best_gold_metrics, ensure_list, normalize_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate factual positives from no-context factual QA dataset")
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="artifacts_poc/factual_qa_train_candidates.parquet",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="artifacts_poc/factual_positives_v1.parquet",
    )
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


def build_factual_prompt(question: str) -> str:
    clean_question = str(question or "").strip()
    return (
        "Ответь на вопрос кратко и по существу. "
        "Если знаешь ответ, дай только сам ответ без пояснений.\n"
        f"Вопрос: {clean_question}\n"
        "Ответ:"
    )


def build_output_row(row_s: pd.Series, prompt_text: str, model_answer: str) -> dict[str, Any] | None:
    gold_answers = ensure_list(row_s.get("gold_answers_text"))
    normalized_model_answer = normalize_text(model_answer)
    normalized_gold_answers = [normalize_text(answer) for answer in gold_answers]
    exact_match_any = normalized_model_answer in set(normalized_gold_answers)
    metrics = best_gold_metrics(model_answer, gold_answers)

    answer_len = len(str(model_answer or "").strip())
    is_hallucination: int | None
    if metrics["token_f1_max"] < 0.25 and answer_len > 5:
        is_hallucination = 1
    elif metrics["token_f1_max"] > 0.7 or exact_match_any:
        is_hallucination = 0
    else:
        return None

    return {
        "id": row_s.get("id"),
        "question": row_s.get("question", ""),
        "gold_answer": row_s.get("gold_answer", ""),
        "gold_answers_text": gold_answers,
        "model_answer": model_answer,
        "prompt_text": prompt_text,
        "is_hallucination": is_hallucination,
        "exact_match_any": bool(exact_match_any),
        "token_f1_max": metrics["token_f1_max"],
        "content_token_f1_max": metrics["content_token_f1_max"],
        "char_similarity_max": metrics["char_similarity_max"],
        "source_dataset": row_s.get("source_dataset", ""),
        "source_split": row_s.get("source_split", ""),
        "source_id": row_s.get("source_id", ""),
        "source": "factual_generated",
    }


def main() -> None:
    args = parse_args()
    dataset_path = Path(args.dataset_path)
    output_path = Path(args.output_path)

    df = pd.read_parquet(dataset_path)
    if "gold_answers_text" not in df.columns:
        raise RuntimeError("dataset missing gold_answers_text")
    if "question" not in df.columns:
        raise RuntimeError("dataset missing question")

    df = df.copy()
    df["gold_answers_text"] = df["gold_answers_text"].apply(ensure_list)
    if "split_name" in df.columns:
        df = df.loc[df["split_name"] == "train"].copy()

    if df.empty:
        raise RuntimeError("no train rows found")

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
        print(f"sampled factual train rows: {total_rows}")

    model, tokenizer = load_model_and_tokenizer(args.model_id, Path(args.cache_dir))

    iterator_df = df.iloc[start_idx:] if start_idx > 0 else df
    processed_since_save = 0
    kept_rows = len(rows)

    for row in tqdm(iterator_df.itertuples(index=False), total=len(iterator_df), initial=0):
        row_s = pd.Series(row._asdict())
        prompt_text = build_factual_prompt(row_s.get("question", ""))
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
            print(
                f"checkpoint saved: {output_path} "
                f"(processed={start_idx + processed_since_save}/{total_rows}, kept={kept_rows})"
            )
            processed_since_save = 0

    save_rows(rows, output_path)
    print(f"saved: {output_path} (processed={total_rows}/{total_rows}, kept={len(rows)})")


if __name__ == "__main__":
    main()
