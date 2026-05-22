#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Подготовка таблицы для сравнения ответов GigaChat с эталоном и последующей разметки.

Вход: parquet-файлы после generate_answers_from_sberchallenge.py
Выход: единый parquet/csv с колонками для анализа и ручной валидации.

Добавляем:
- normalized_model_answer
- normalized_gold_answers
- exact_match_any
- substring_match_any
- token_f1_max
- content_token_f1_max
- char_similarity_max
- has_uncertainty_marker
- weak_label
- weak_label_reason
- needs_manual_review
- is_hallucination_manual
- reviewer_comment

Пример:
    python prepare_hallucination_labeling.py --inputs-dir outputs --output-dir artifacts
"""

from __future__ import annotations

import argparse
import ast
import re
import string
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import pandas as pd


FUNCTION_WORDS = {
    "а", "без", "более", "бы", "в", "во", "все", "для", "до", "его", "ее", "её", "же",
    "за", "и", "из", "или", "им", "их", "к", "как", "ко", "ли", "на", "над", "не",
    "но", "о", "об", "обо", "около", "он", "она", "оно", "они", "от", "по", "под",
    "при", "про", "с", "со", "так", "то", "у", "это", "эта", "этот", "эти",
}

UNCERTAINTY_PATTERNS = [
    r"\bне знаю\b",
    r"\bне могу\b",
    r"\bне удалось\b",
    r"\bне указано\b",
    r"\bне сказано\b",
    r"\bне упоминается\b",
    r"\bнет информации\b",
    r"\bнедостаточно информации\b",
    r"\bне могу ответить\b",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs-dir", type=str, default="outputs")
    parser.add_argument("--output-dir", type=str, default="artifacts")
    parser.add_argument("--only-split", type=str, default=None)
    return parser.parse_args()


def normalize_text(text: str) -> str:
    text = "" if text is None else str(text)
    text = text.lower().strip()
    text = text.replace("ё", "е")
    text = re.sub(r"\s+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation + "«»“”„…"))
    return text.strip()


def ensure_list(value: Any) -> list[str]:
    if value is None:
        return []

    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return ensure_list(tolist())

    if isinstance(value, dict):
        if "text" in value:
            return ensure_list(value["text"])
        return [str(value)]

    if isinstance(value, (list, tuple, set)):
        result: list[str] = []
        for item in value:
            result.extend(ensure_list(item))
        return result

    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            parsed = ast.literal_eval(text)
            if parsed != value:
                return ensure_list(parsed)
        except Exception:
            pass
        return [text]

    return [str(value)]


def token_f1(prediction: str, target: str) -> float:
    pred_tokens = normalize_text(prediction).split()
    target_tokens = normalize_text(target).split()
    if not pred_tokens or not target_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(target_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0

    precision = overlap / len(pred_tokens)
    recall = overlap / len(target_tokens)
    return 2 * precision * recall / (precision + recall)


def content_tokens(text: str) -> list[str]:
    return [tok for tok in normalize_text(text).split() if tok and tok not in FUNCTION_WORDS]


def content_token_f1(prediction: str, target: str) -> float:
    pred_tokens = content_tokens(prediction)
    target_tokens = content_tokens(target)
    if not pred_tokens or not target_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(target_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0

    precision = overlap / len(pred_tokens)
    recall = overlap / len(target_tokens)
    return 2 * precision * recall / (precision + recall)


def strip_leading_function_words(text: str) -> str:
    tokens = normalize_text(text).split()
    while tokens and tokens[0] in FUNCTION_WORDS:
        tokens = tokens[1:]
    return " ".join(tokens)


def char_similarity(a: str, b: str) -> float:
    a_norm = strip_leading_function_words(a)
    b_norm = strip_leading_function_words(b)
    if not a_norm or not b_norm:
        return 0.0
    return SequenceMatcher(None, a_norm, b_norm).ratio()


def extract_number_tokens(text: str) -> tuple[str, ...]:
    clean = "" if text is None else str(text).replace(",", ".")
    return tuple(re.findall(r"\d+(?:\.\d+)?", clean))


def has_uncertainty_marker(text: str) -> bool:
    norm = normalize_text(text)
    return any(re.search(pattern, norm) for pattern in UNCERTAINTY_PATTERNS)


def answer_supported_by_context(answer: str, context: str) -> bool:
    answer_norm = strip_leading_function_words(answer)
    context_norm = normalize_text(context)
    if not answer_norm or not context_norm:
        return False
    if answer_norm in context_norm:
        return True

    answer_tokens = [tok for tok in answer_norm.split() if len(tok) >= 4]
    if not answer_tokens:
        return False

    return sum(tok in context_norm for tok in answer_tokens) >= max(1, len(answer_tokens) - 1)


def best_gold_metrics(model_answer: str, gold_answers: list[str]) -> dict[str, Any]:
    answer_numbers = extract_number_tokens(model_answer)
    if not gold_answers:
        return {
            "best_gold_answer": "",
            "token_f1_max": 0.0,
            "content_token_f1_max": 0.0,
            "char_similarity_max": 0.0,
            "gold_numbers_best": tuple(),
            "answer_numbers": answer_numbers,
        }

    scored = []
    for gold in gold_answers:
        scored.append(
            {
                "gold": gold,
                "token_f1": token_f1(model_answer, gold),
                "content_token_f1": content_token_f1(model_answer, gold),
                "char_similarity": char_similarity(model_answer, gold),
                "gold_numbers": extract_number_tokens(gold),
            }
        )

    best = max(scored, key=lambda x: (x["token_f1"], x["content_token_f1"], x["char_similarity"]))
    return {
        "best_gold_answer": best["gold"],
        "token_f1_max": best["token_f1"],
        "content_token_f1_max": best["content_token_f1"],
        "char_similarity_max": best["char_similarity"],
        "gold_numbers_best": best["gold_numbers"],
        "answer_numbers": answer_numbers,
    }


def infer_weak_label(row: pd.Series) -> tuple[int | None, str]:
    if row["has_uncertainty_marker"]:
        return 1, "uncertainty_marker"
    if row["answer_numbers"] and row["gold_numbers_best"] and row["answer_numbers"] != row["gold_numbers_best"]:
        return 1, "numeric_mismatch"

    answer_token_count = len(strip_leading_function_words(row["model_answer"]).split())
    gold_token_count = len(strip_leading_function_words(row["best_gold_answer"]).split())

    if row["exact_match_any"]:
        return 0, "exact_match"
    if row["substring_match_any"]:
        return 0, "substring_match"
    if row["token_f1_max"] >= 0.85:
        return 0, "high_token_f1"
    if row["content_token_f1_max"] >= 0.9:
        return 0, "high_content_f1"
    if row["char_similarity_max"] >= 0.92:
        return 0, "high_char_similarity"
    if answer_token_count == 1 and gold_token_count == 1 and row["char_similarity_max"] >= 0.78:
        return 0, "single_token_inflection"
    if row["token_f1_max"] >= 0.55 and row["char_similarity_max"] >= 0.72:
        return 0, "paraphrase_or_inflection"
    if row["content_token_f1_max"] >= 0.6 and row["char_similarity_max"] >= 0.72:
        return 0, "content_overlap_paraphrase"
    if (
        row["token_f1_max"] <= 0.15
        and row["content_token_f1_max"] <= 0.2
        and row["char_similarity_max"] <= 0.35
        and not row["model_answer_supported_by_context"]
    ):
        return 1, "low_similarity_unsupported"

    return None, "manual_review"


def load_generated_tables(inputs_dir: Path) -> list[Path]:
    return sorted(inputs_dir.rglob("*_with_answers.parquet"))


def main() -> None:
    args = parse_args()
    inputs_dir = Path(args.inputs_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = load_generated_tables(inputs_dir)
    if args.only_split:
        files = [path for path in files if f"/{args.only_split}/" in str(path)]

    if not files:
        raise RuntimeError(f"В {inputs_dir} не найдено parquet-файлов с ответами")

    frames = []
    for path in files:
        df = pd.read_parquet(path)
        df["generated_file"] = str(path)
        frames.append(df)

    df = pd.concat(frames, ignore_index=True)

    if "gold_answers_text" not in df.columns and "answers" in df.columns:
        df["gold_answers_text"] = df["answers"].apply(ensure_list)
    else:
        df["gold_answers_text"] = df["gold_answers_text"].apply(ensure_list)

    df["model_answer"] = df["model_answer"].fillna("").astype(str)
    df["normalized_model_answer"] = df["model_answer"].apply(normalize_text)
    df["normalized_gold_answers"] = df["gold_answers_text"].apply(
        lambda answers: [normalize_text(answer) for answer in answers]
    )

    df["exact_match_any"] = df.apply(
        lambda row: row["normalized_model_answer"] in set(row["normalized_gold_answers"]),
        axis=1,
    )
    df["substring_match_any"] = df.apply(
        lambda row: any(
            gold and (gold in row["normalized_model_answer"] or row["normalized_model_answer"] in gold)
            for gold in row["normalized_gold_answers"]
        ),
        axis=1,
    )

    metrics_df = df.apply(
        lambda row: pd.Series(best_gold_metrics(row["model_answer"], row["gold_answers_text"])),
        axis=1,
    )
    df = pd.concat([df, metrics_df], axis=1)

    df["has_uncertainty_marker"] = df["model_answer"].apply(has_uncertainty_marker)
    df["model_answer_supported_by_context"] = df.apply(
        lambda row: answer_supported_by_context(row["model_answer"], row.get("context", "")),
        axis=1,
    )

    weak_labels = df.apply(infer_weak_label, axis=1)
    df["weak_label"] = [label for label, _ in weak_labels]
    df["weak_label_reason"] = [reason for _, reason in weak_labels]
    df["needs_manual_review"] = df["weak_label"].isna()
    df["is_hallucination_heuristic"] = df["weak_label"].fillna(0).astype(int) == 1
    df["is_hallucination_manual"] = pd.NA
    df["reviewer_comment"] = ""

    preferred_columns = [
        "id",
        "split_name",
        "title",
        "question",
        "context",
        "gold_answers_text",
        "best_gold_answer",
        "model_answer",
        "exact_match_any",
        "substring_match_any",
        "token_f1_max",
        "content_token_f1_max",
        "char_similarity_max",
        "has_uncertainty_marker",
        "model_answer_supported_by_context",
        "weak_label",
        "weak_label_reason",
        "needs_manual_review",
        "is_hallucination_heuristic",
        "is_hallucination_manual",
        "reviewer_comment",
        "prompt_text",
        "source_file",
        "generated_file",
    ]
    tail_columns = [column for column in df.columns if column not in preferred_columns]
    df = df[preferred_columns + tail_columns]

    parquet_path = output_dir / "gigachat_sberquad_labeling_dataset.parquet"
    csv_path = output_dir / "gigachat_sberquad_labeling_dataset.csv"
    manual_review_path = output_dir / "manual_review_candidates.parquet"
    manual_review_csv_path = output_dir / "manual_review_candidates.csv"

    df.to_parquet(parquet_path, index=False)
    df.to_csv(csv_path, index=False)
    review_df = df[df["needs_manual_review"]].copy()
    review_df.to_parquet(manual_review_path, index=False)
    review_df.to_csv(manual_review_csv_path, index=False)

    print(f"saved: {parquet_path}")
    print(f"saved: {csv_path}")
    print(f"saved: {manual_review_path}")
    print(f"saved: {manual_review_csv_path}")
    print(f"rows: {len(df)}")
    print(f"weak_label_non_null: {int(df['weak_label'].notna().sum())}")
    print(f"heuristic_positive: {int(df['is_hallucination_heuristic'].sum())}")
    print(f"needs_manual_review: {int(df['needs_manual_review'].sum())}")


if __name__ == "__main__":
    main()
