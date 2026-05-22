#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import re

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from prepare_hallucination_labeling import extract_number_tokens, normalize_text


QUESTION_PATTERNS = {
    "expects_person": [
        r"\bкто\b",
        r"\bкем\b",
        r"\bкого\b",
        r"\bкому\b",
        r"\bчей\b",
    ],
    "expects_date": [
        r"\bкогда\b",
        r"\bв каком году\b",
        r"\bв каком веке\b",
        r"\bкакого года\b",
        r"\bв каком месяце\b",
        r"\bкакого числа\b",
        r"\bв каком десятилетии\b",
    ],
    "expects_number": [
        r"\bсколько\b",
        r"\bкакое количество\b",
        r"\bкакова численность\b",
        r"\bкакой процент\b",
        r"\bкакой номер\b",
        r"\bскольких\b",
    ],
    "expects_location": [
        r"\bгде\b",
        r"\bв какой стране\b",
        r"\bв каком городе\b",
        r"\bна каком острове\b",
        r"\bв каком регионе\b",
        r"\bв какой области\b",
    ],
    "expects_yesno": [
        r"^верно ли\b",
        r"^правда ли\b",
        r"^является ли\b",
        r"^был ли\b",
        r"^была ли\b",
        r"^было ли\b",
        r"^можно ли\b",
    ],
    "expects_reason": [
        r"\bпочему\b",
        r"\bзачем\b",
        r"\bпо какой причине\b",
    ],
}

SHORT_ANSWER_EXPECTED_KEYS = (
    "expects_person",
    "expects_date",
    "expects_number",
    "expects_location",
    "expects_yesno",
)

HEDGING_PATTERNS = [
    r"\bвозможно\b",
    r"\bвероятно\b",
    r"\bпредположительно\b",
    r"\bпо некоторым данным\b",
    r"\bсчитается\b",
    r"\bкак правило\b",
    r"\bобычно\b",
    r"\bможет\b",
]

EXPLANATION_PATTERNS = [
    r"\bпотому что\b",
    r"\bтак как\b",
    r"\bкоторый\b",
    r"\bкоторая\b",
    r"\bкоторые\b",
    r"\bпоскольку\b",
    r"\bэто связано\b",
]

MONTH_PATTERNS = [
    r"\bянвар",
    r"\bфеврал",
    r"\bмарт",
    r"\bапрел",
    r"\bма[йя]\b",
    r"\bиюн",
    r"\bиюл",
    r"\bавгуст",
    r"\bсентябр",
    r"\bоктябр",
    r"\bноябр",
    r"\bдекабр",
]

ORDINAL_HINTS = [
    r"\bперв",
    r"\bвтор",
    r"\bтрет",
    r"\bчетверт",
    r"\bпят",
    r"\bшест",
    r"\bседьм",
    r"\bвосьм",
    r"\bдевят",
    r"\bдесят",
]


def get_question_text(row: pd.Series) -> str:
    if "question" in row and pd.notna(row.get("question")):
        return str(row.get("question"))
    if "prompt" in row and pd.notna(row.get("prompt")):
        return str(row.get("prompt"))
    return ""


def get_answer_text(row: pd.Series) -> str:
    return "" if pd.isna(row.get("model_answer")) else str(row.get("model_answer"))


def _regex_any(text: str, patterns: list[str]) -> bool:
    return any(re.search(pattern, text) for pattern in patterns)


def _extract_years(text: str) -> list[str]:
    return re.findall(r"\b(?:1[0-9]{3}|20[0-9]{2}|21[0-9]{2})\b", text or "")


def _sentence_count(text: str) -> int:
    parts = [p for p in re.split(r"[.!?]+", text or "") if p.strip()]
    return len(parts)


def _first_tokens(text: str, n: int) -> list[str]:
    return [tok for tok in normalize_text(text).split() if tok][:n]


def question_type_flags(question: str) -> dict[str, float]:
    q = normalize_text(question)
    out = {}
    for key, patterns in QUESTION_PATTERNS.items():
        out[key] = float(_regex_any(q, patterns))
    out["expects_short_fact"] = float(any(out[key] > 0 for key in SHORT_ANSWER_EXPECTED_KEYS))
    out["question_has_year"] = float(bool(_extract_years(question)))
    out["question_has_digit"] = float(bool(re.search(r"\d", question or "")))
    out["question_has_month_name"] = float(_regex_any(q, MONTH_PATTERNS))
    out["question_has_ordinal_hint"] = float(_regex_any(q, ORDINAL_HINTS))
    return out


def answer_shape_features(answer: str) -> dict[str, float]:
    raw = str(answer or "").strip()
    norm = normalize_text(raw)
    words = [w for w in norm.split() if w]
    sentence_count = _sentence_count(raw)
    digit_tokens = extract_number_tokens(raw)
    year_tokens = _extract_years(raw)
    first_tokens = _first_tokens(raw, 4)
    return {
        "answer_char_len": float(len(raw)),
        "answer_word_len": float(len(words)),
        "answer_sentence_count": float(sentence_count),
        "answer_digit_count": float(len(re.findall(r"\d", raw))),
        "answer_numeric_token_count": float(len(digit_tokens)),
        "answer_year_token_count": float(len(year_tokens)),
        "answer_has_digit": float(bool(re.search(r"\d", raw))),
        "answer_has_year_like": float(bool(year_tokens)),
        "answer_has_month_name": float(_regex_any(norm, MONTH_PATTERNS)),
        "answer_has_yes_token": float(any(tok in {"да", "нет"} for tok in first_tokens)),
        "answer_is_short": float(len(words) <= 3),
        "answer_is_very_short": float(len(words) <= 1),
        "answer_is_long": float(len(words) >= 12),
        "answer_is_very_long": float(len(words) >= 24),
        "answer_upper_ratio": float(sum(ch.isupper() for ch in raw) / max(len(raw), 1)),
        "answer_punct_ratio": float(sum(not ch.isalnum() and not ch.isspace() for ch in raw) / max(len(raw), 1)),
        "answer_has_parentheses": float("(" in raw or ")" in raw),
        "answer_has_colon": float(":" in raw),
        "answer_has_comma": float("," in raw),
        "answer_starts_with_namecase": float(raw[:1].isupper()) if raw else 0.0,
        "answer_has_hedging": float(_regex_any(norm, HEDGING_PATTERNS)),
        "answer_has_explanation_marker": float(_regex_any(norm, EXPLANATION_PATTERNS)),
    }


def numeric_consistency_features(question: str, answer: str) -> dict[str, float]:
    q_nums = extract_number_tokens(question)
    a_nums = extract_number_tokens(answer)
    q_years = _extract_years(question)
    a_years = _extract_years(answer)
    q_num_set = set(q_nums)
    a_num_set = set(a_nums)
    shared_nums = q_num_set & a_num_set
    new_answer_nums = a_num_set - q_num_set
    missing_question_nums = q_num_set - a_num_set
    return {
        "qa_shared_number_count": float(len(shared_nums)),
        "qa_new_answer_number_count": float(len(new_answer_nums)),
        "qa_missing_question_number_count": float(len(missing_question_nums)),
        "qa_shared_year_count": float(len(set(q_years) & set(a_years))),
        "qa_new_answer_year_count": float(len(set(a_years) - set(q_years))),
        "answer_has_more_numbers_than_question": float(len(a_nums) > len(q_nums)),
        "answer_introduces_year_without_question_year": float(bool(a_years) and not bool(q_years)),
        "question_has_year_but_answer_not": float(bool(q_years) and not bool(a_years)),
    }


def verbosity_features(question: str, answer: str) -> dict[str, float]:
    q_words = [w for w in normalize_text(question).split() if w]
    a_words = [w for w in normalize_text(answer).split() if w]
    ratio = len(a_words) / max(len(q_words), 1)
    first_answer_tokens = _first_tokens(answer, 8)
    return {
        "answer_to_question_word_ratio": float(ratio),
        "answer_longer_than_question": float(len(a_words) > len(q_words)),
        "answer_starts_with_repeat_of_question": float(bool(first_answer_tokens) and " ".join(first_answer_tokens[:3]) in normalize_text(question)),
        "answer_has_multiple_sentences": float(_sentence_count(answer) >= 2),
    }


def type_mismatch_features(question: str, answer: str) -> dict[str, float]:
    q_flags = question_type_flags(question)
    a_shape = answer_shape_features(answer)
    answer_numbers = extract_number_tokens(answer)

    return {
        **q_flags,
        **a_shape,
        "mismatch_person_but_has_digit": float(q_flags["expects_person"] and a_shape["answer_has_digit"]),
        "mismatch_date_but_no_digit": float(q_flags["expects_date"] and not a_shape["answer_has_digit"]),
        "mismatch_number_but_no_digit": float(q_flags["expects_number"] and not a_shape["answer_has_digit"]),
        "mismatch_location_but_has_year": float(q_flags["expects_location"] and a_shape["answer_has_year_like"]),
        "mismatch_yesno_but_not_yesno": float(q_flags["expects_yesno"] and not a_shape["answer_has_yes_token"]),
        "mismatch_shortfact_but_very_long": float(q_flags["expects_short_fact"] and a_shape["answer_is_very_long"]),
        "mismatch_shortfact_but_multisentence": float(q_flags["expects_short_fact"] and a_shape["answer_sentence_count"] >= 2),
        "mismatch_reason_but_too_short": float(q_flags["expects_reason"] and a_shape["answer_is_short"]),
        "match_number_expected_and_present": float(q_flags["expects_number"] and bool(answer_numbers)),
        "match_date_expected_and_present": float(q_flags["expects_date"] and (a_shape["answer_has_year_like"] or a_shape["answer_has_month_name"])),
        "match_person_expected_and_titlecase": float(q_flags["expects_person"] and a_shape["answer_starts_with_namecase"]),
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
        "qa_question_prefix_in_answer": float(bool(q_norm) and q_norm[: min(25, len(q_norm))] in a_norm) if q_norm else 0.0,
    }


def extract_qa_consistency_dense_features_one(row: pd.Series) -> dict[str, float]:
    question = get_question_text(row)
    answer = get_answer_text(row)
    feats: dict[str, float] = {}
    feats.update(type_mismatch_features(question, answer))
    feats.update(lexical_pair_features(question, answer))
    feats.update(numeric_consistency_features(question, answer))
    feats.update(verbosity_features(question, answer))
    return feats


def build_text_views(df: pd.DataFrame) -> dict[str, list[str]]:
    questions = [get_question_text(pd.Series(rec)) for rec in df.to_dict(orient="records")]
    answers = [get_answer_text(pd.Series(rec)) for rec in df.to_dict(orient="records")]
    qa_concat = [f"Q: {q}\nA: {a}" for q, a in zip(questions, answers)]
    answer_firstline = [(a.splitlines()[0] if a else "") for a in answers]
    return {
        "question": questions,
        "answer": answers,
        "qa_concat": qa_concat,
        "answer_firstline": answer_firstline,
    }


def fit_text_vectorizers(df: pd.DataFrame) -> dict[str, TfidfVectorizer]:
    views = build_text_views(df)
    return {
        "question_char": TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), lowercase=True, min_df=2, sublinear_tf=True).fit(views["question"]),
        "answer_char": TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), lowercase=True, min_df=2, sublinear_tf=True).fit(views["answer"]),
        "qa_word": TfidfVectorizer(analyzer="word", ngram_range=(1, 2), lowercase=True, min_df=2, sublinear_tf=True).fit(views["qa_concat"]),
        "answer_firstline_word": TfidfVectorizer(analyzer="word", ngram_range=(1, 2), lowercase=True, min_df=2, sublinear_tf=True).fit(views["answer_firstline"]),
    }


def transform_text_features(df: pd.DataFrame, vectorizers: dict[str, TfidfVectorizer]):
    views = build_text_views(df)
    dense_df = pd.DataFrame([extract_qa_consistency_dense_features_one(pd.Series(rec)) for rec in df.to_dict(orient="records")])
    mats = [
        vectorizers["question_char"].transform(views["question"]),
        vectorizers["answer_char"].transform(views["answer"]),
        vectorizers["qa_word"].transform(views["qa_concat"]),
        vectorizers["answer_firstline_word"].transform(views["answer_firstline"]),
        sparse.csr_matrix(dense_df.astype(np.float32).values),
    ]
    return sparse.hstack(mats).tocsr(), dense_df


def build_feature_name_list(vectorizers: dict[str, TfidfVectorizer], dense_columns: list[str]) -> list[str]:
    names = []
    names.extend([f"question_char::{x}" for x in vectorizers["question_char"].get_feature_names_out().tolist()])
    names.extend([f"answer_char::{x}" for x in vectorizers["answer_char"].get_feature_names_out().tolist()])
    names.extend([f"qa_word::{x}" for x in vectorizers["qa_word"].get_feature_names_out().tolist()])
    names.extend([f"answer_firstline_word::{x}" for x in vectorizers["answer_firstline_word"].get_feature_names_out().tolist()])
    names.extend([f"dense::{x}" for x in dense_columns])
    return names
