#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from prepare_hallucination_labeling import content_token_f1, ensure_list, extract_number_tokens, normalize_text, token_f1


QUESTION_PATTERNS = {
    "expects_person": [
        r"\bкто\b",
        r"\bкем\b",
        r"\bкого\b",
        r"\bкому\b",
    ],
    "expects_date": [
        r"\bкогда\b",
        r"\bв каком году\b",
        r"\bв каком веке\b",
        r"\bкакого года\b",
        r"\bв каком месяце\b",
    ],
    "expects_number": [
        r"\bсколько\b",
        r"\bкакое количество\b",
        r"\bкакова численность\b",
        r"\bкакой процент\b",
    ],
    "expects_location": [
        r"\bгде\b",
        r"\bв какой стране\b",
        r"\bв каком городе\b",
        r"\bна каком острове\b",
        r"\bв каком регионе\b",
    ],
    "expects_yesno": [
        r"^верно ли\b",
        r"^правда ли\b",
        r"^является ли\b",
        r"^был ли\b",
        r"^была ли\b",
        r"^было ли\b",
    ],
}


def get_question_text(row: pd.Series) -> str:
    if "question" in row and pd.notna(row.get("question")):
        return str(row.get("question"))
    if "prompt" in row and pd.notna(row.get("prompt")):
        return str(row.get("prompt"))
    return ""


def get_answer_text(row: pd.Series) -> str:
    return "" if pd.isna(row.get("model_answer")) else str(row.get("model_answer"))


def question_type_flags(question: str) -> dict[str, float]:
    q = normalize_text(question)
    out = {}
    for key, patterns in QUESTION_PATTERNS.items():
        out[key] = float(any(re.search(pattern, q) for pattern in patterns))
    return out


def answer_shape_features(answer: str) -> dict[str, float]:
    raw = str(answer or "").strip()
    norm = normalize_text(raw)
    words = [w for w in norm.split() if w]
    year_like = bool(re.search(r"\b(1[0-9]{3}|20[0-9]{2}|21[0-9]{2})\b", raw))
    return {
        "answer_char_len": float(len(raw)),
        "answer_word_len": float(len(words)),
        "answer_digit_count": float(len(re.findall(r"\d", raw))),
        "answer_has_digit": float(bool(re.search(r"\d", raw))),
        "answer_has_year_like": float(year_like),
        "answer_has_yes_token": float(any(tok in {"да", "нет"} for tok in words[:2])),
        "answer_is_short": float(len(words) <= 3),
        "answer_is_long": float(len(words) >= 12),
        "answer_upper_ratio": float(sum(ch.isupper() for ch in raw) / max(len(raw), 1)),
        "answer_punct_ratio": float(sum(not ch.isalnum() and not ch.isspace() for ch in raw) / max(len(raw), 1)),
    }


def type_mismatch_features(question: str, answer: str) -> dict[str, float]:
    q_flags = question_type_flags(question)
    a_shape = answer_shape_features(answer)

    answer_numbers = extract_number_tokens(answer)
    raw = str(answer or "")
    first_char_upper = float(raw[:1].isupper()) if raw else 0.0

    return {
        **q_flags,
        **a_shape,
        "mismatch_person_but_has_digit": float(q_flags["expects_person"] and a_shape["answer_has_digit"]),
        "mismatch_date_but_no_digit": float(q_flags["expects_date"] and not a_shape["answer_has_digit"]),
        "mismatch_number_but_no_digit": float(q_flags["expects_number"] and not a_shape["answer_has_digit"]),
        "mismatch_location_but_has_year": float(q_flags["expects_location"] and a_shape["answer_has_year_like"]),
        "mismatch_yesno_but_not_yesno": float(q_flags["expects_yesno"] and not a_shape["answer_has_yes_token"]),
        "match_number_expected_and_present": float(q_flags["expects_number"] and bool(answer_numbers)),
        "match_date_expected_and_present": float(q_flags["expects_date"] and a_shape["answer_has_year_like"]),
        "match_person_expected_and_titlecase": float(q_flags["expects_person"] and first_char_upper),
    }


def lexical_pair_features(question: str, answer: str) -> dict[str, float]:
    q_norm = normalize_text(question)
    a_norm = normalize_text(answer)
    q_words = set(q_norm.split())
    a_words = set(a_norm.split())
    overlap = len(q_words & a_words)
    union = len(q_words | a_words)
    return {
        "qa_token_overlap_count": float(overlap),
        "qa_token_jaccard": float(overlap / union) if union else 0.0,
        "qa_answer_in_question": float(bool(a_norm) and a_norm in q_norm),
        "qa_question_in_answer": float(bool(q_norm) and q_norm[: min(20, len(q_norm))] in a_norm) if q_norm else 0.0,
    }


def gold_comparison_features(row: pd.Series) -> dict[str, float]:
    answer = get_answer_text(row)
    gold_answers = ensure_list(row.get("gold_answers_text"))
    if not gold_answers and pd.notna(row.get("correct_answer")):
        gold_answers = [str(row.get("correct_answer"))]
    if not gold_answers:
        return {
            "gold_token_f1_max": 0.0,
            "gold_content_f1_max": 0.0,
            "gold_char_match_hint": 0.0,
        }
    token_scores = [token_f1(answer, gold) for gold in gold_answers]
    content_scores = [content_token_f1(answer, gold) for gold in gold_answers]
    char_scores = [float(normalize_text(answer) == normalize_text(gold)) for gold in gold_answers]
    return {
        "gold_token_f1_max": float(max(token_scores)) if token_scores else 0.0,
        "gold_content_f1_max": float(max(content_scores)) if content_scores else 0.0,
        "gold_char_match_hint": float(max(char_scores)) if char_scores else 0.0,
    }


def extract_qa_consistency_dense_features_one(row: pd.Series) -> dict[str, float]:
    question = get_question_text(row)
    answer = get_answer_text(row)
    feats: dict[str, float] = {}
    feats.update(type_mismatch_features(question, answer))
    feats.update(lexical_pair_features(question, answer))
    feats.update(gold_comparison_features(row))
    return feats


def build_text_views(df: pd.DataFrame) -> dict[str, list[str]]:
    questions = [get_question_text(pd.Series(rec)) for rec in df.to_dict(orient="records")]
    answers = [get_answer_text(pd.Series(rec)) for rec in df.to_dict(orient="records")]
    qa_concat = [f"Q: {q}\nA: {a}" for q, a in zip(questions, answers)]
    return {
        "question": questions,
        "answer": answers,
        "qa_concat": qa_concat,
    }


def fit_text_vectorizers(df: pd.DataFrame) -> dict[str, TfidfVectorizer]:
    views = build_text_views(df)
    return {
        "question_char": TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), lowercase=True, min_df=1, sublinear_tf=True).fit(views["question"]),
        "answer_char": TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), lowercase=True, min_df=1, sublinear_tf=True).fit(views["answer"]),
        "qa_word": TfidfVectorizer(analyzer="word", ngram_range=(1, 2), lowercase=True, min_df=1, sublinear_tf=True).fit(views["qa_concat"]),
    }


def transform_text_features(df: pd.DataFrame, vectorizers: dict[str, TfidfVectorizer]):
    views = build_text_views(df)
    dense_df = pd.DataFrame([extract_qa_consistency_dense_features_one(pd.Series(rec)) for rec in df.to_dict(orient="records")])
    mats = [
        vectorizers["question_char"].transform(views["question"]),
        vectorizers["answer_char"].transform(views["answer"]),
        vectorizers["qa_word"].transform(views["qa_concat"]),
        sparse.csr_matrix(dense_df.astype(np.float32).values),
    ]
    return sparse.hstack(mats).tocsr(), dense_df


def build_feature_name_list(vectorizers: dict[str, TfidfVectorizer], dense_columns: list[str]) -> list[str]:
    names = []
    names.extend([f"question_char::{x}" for x in vectorizers["question_char"].get_feature_names_out().tolist()])
    names.extend([f"answer_char::{x}" for x in vectorizers["answer_char"].get_feature_names_out().tolist()])
    names.extend([f"qa_word::{x}" for x in vectorizers["qa_word"].get_feature_names_out().tolist()])
    names.extend([f"dense::{x}" for x in dense_columns])
    return names
