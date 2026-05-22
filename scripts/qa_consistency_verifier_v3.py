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
        r"\bскольким\b",
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
    r"\bкакой по сч[её]ту\b",
    r"\bкакое место\b",
]

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


def _words(text: str) -> list[str]:
    return [w for w in normalize_text(text).split() if w]


def _first_tokens(text: str, n: int) -> list[str]:
    return _words(text)[:n]


def _first_sentence(text: str) -> str:
    raw = str(text or "").strip()
    if not raw:
        return ""
    parts = re.split(r"(?<=[.!?])\s+", raw, maxsplit=1)
    return parts[0].strip()


def _core_answer_span(text: str) -> str:
    raw = str(text or "").strip()
    if not raw:
        return ""
    first_sentence = _first_sentence(raw)
    clause_parts = re.split(r"\s*[,:;—-]\s+|\s+\b(?:потому что|так как|который|которая|которые|поскольку)\b\s+", first_sentence, maxsplit=1, flags=re.IGNORECASE)
    core = clause_parts[0].strip()
    words = _words(core)
    if len(words) > 8:
        core = " ".join(words[:8])
    return core


def _token_overlap_ratio(a: str, b: str) -> float:
    aw = set(_words(a))
    bw = set(_words(b))
    if not aw or not bw:
        return 0.0
    return len(aw & bw) / len(aw | bw)


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
    words = _words(raw)
    year_tokens = _extract_years(raw)
    digit_tokens = extract_number_tokens(raw)
    first_tokens = _first_tokens(raw, 4)
    return {
        "answer_char_len": float(len(raw)),
        "answer_word_len": float(len(words)),
        "answer_sentence_count": float(_sentence_count(raw)),
        "answer_digit_count": float(len(re.findall(r"\d", raw))),
        "answer_numeric_token_count": float(len(digit_tokens)),
        "answer_year_token_count": float(len(year_tokens)),
        "answer_has_digit": float(bool(re.search(r"\d", raw))),
        "answer_has_year_like": float(bool(year_tokens)),
        "answer_has_month_name": float(_regex_any(normalize_text(raw), MONTH_PATTERNS)),
        "answer_has_yes_token": float(any(tok in {"да", "нет"} for tok in first_tokens)),
        "answer_is_short": float(len(words) <= 3),
        "answer_is_very_short": float(len(words) <= 1),
        "answer_is_long": float(len(words) >= 12),
        "answer_is_very_long": float(len(words) >= 24),
        "answer_has_hedging": float(_regex_any(normalize_text(raw), HEDGING_PATTERNS)),
        "answer_has_explanation_marker": float(_regex_any(normalize_text(raw), EXPLANATION_PATTERNS)),
    }


def first_span_features(question: str, answer: str) -> dict[str, float]:
    core = _core_answer_span(answer)
    first_sentence = _first_sentence(answer)
    all_words = _words(answer)
    core_words = _words(core)
    tail_words = max(len(all_words) - len(core_words), 0)
    q_flags = question_type_flags(question)

    core_nums = extract_number_tokens(core)
    q_nums = extract_number_tokens(question)
    core_years = _extract_years(core)
    q_years = _extract_years(question)

    return {
        "core_word_len": float(len(core_words)),
        "core_is_short": float(len(core_words) <= 5),
        "core_sentence_overlap_jaccard": _token_overlap_ratio(core, question),
        "core_has_digit": float(bool(core_nums)),
        "core_has_year_like": float(bool(core_years)),
        "core_has_month_name": float(_regex_any(normalize_text(core), MONTH_PATTERNS)),
        "core_has_yes_token": float(any(tok in {"да", "нет"} for tok in _first_tokens(core, 3))),
        "core_number_count": float(len(core_nums)),
        "core_year_count": float(len(core_years)),
        "core_shared_question_number_count": float(len(set(core_nums) & set(q_nums))),
        "core_shared_question_year_count": float(len(set(core_years) & set(q_years))),
        "tail_word_len": float(tail_words),
        "tail_to_core_ratio": float(tail_words / max(len(core_words), 1)),
        "answer_has_long_tail_after_short_core": float(len(core_words) <= 5 and tail_words >= 8),
        "shortfact_core_good_shape": float(q_flags["expects_short_fact"] and len(core_words) <= 6),
        "shortfact_but_first_sentence_long": float(q_flags["expects_short_fact"] and len(_words(first_sentence)) >= 12),
        "shortfact_but_long_tail": float(q_flags["expects_short_fact"] and len(core_words) <= 6 and tail_words >= 10),
        "date_question_core_missing_date": float(q_flags["expects_date"] and not (core_years or _regex_any(normalize_text(core), MONTH_PATTERNS))),
        "number_question_core_missing_number": float(q_flags["expects_number"] and not bool(core_nums)),
        "yesno_question_core_missing_yesno": float(q_flags["expects_yesno"] and not any(tok in {"да", "нет"} for tok in _first_tokens(core, 2))),
        "location_question_core_has_year": float(q_flags["expects_location"] and bool(core_years)),
        "person_question_core_has_digit": float(q_flags["expects_person"] and bool(core_nums)),
        "ordinal_question_core_missing_number": float(q_flags["question_has_ordinal_hint"] and not bool(core_nums)),
    }


def numeric_consistency_features(question: str, answer: str) -> dict[str, float]:
    q_nums = extract_number_tokens(question)
    a_nums = extract_number_tokens(answer)
    q_years = _extract_years(question)
    a_years = _extract_years(answer)
    q_num_set = set(q_nums)
    a_num_set = set(a_nums)
    return {
        "qa_shared_number_count": float(len(q_num_set & a_num_set)),
        "qa_new_answer_number_count": float(len(a_num_set - q_num_set)),
        "qa_missing_question_number_count": float(len(q_num_set - a_num_set)),
        "qa_shared_year_count": float(len(set(q_years) & set(a_years))),
        "qa_new_answer_year_count": float(len(set(a_years) - set(q_years))),
        "answer_has_more_numbers_than_question": float(len(a_nums) > len(q_nums)),
        "answer_introduces_year_without_question_year": float(bool(a_years) and not bool(q_years)),
        "question_has_year_but_answer_not": float(bool(q_years) and not bool(a_years)),
    }


def verbosity_features(question: str, answer: str) -> dict[str, float]:
    q_words = _words(question)
    a_words = _words(answer)
    return {
        "answer_to_question_word_ratio": float(len(a_words) / max(len(q_words), 1)),
        "answer_longer_than_question": float(len(a_words) > len(q_words)),
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
    }


def lexical_pair_features(question: str, answer: str) -> dict[str, float]:
    q_words = set(_words(question))
    a_words = set(_words(answer))
    overlap = len(q_words & a_words)
    union = len(q_words | a_words)
    return {
        "qa_token_overlap_count": float(overlap),
        "qa_token_jaccard": float(overlap / union) if union else 0.0,
    }


def extract_qa_consistency_dense_features_one(row: pd.Series) -> dict[str, float]:
    question = get_question_text(row)
    answer = get_answer_text(row)
    feats: dict[str, float] = {}
    feats.update(type_mismatch_features(question, answer))
    feats.update(lexical_pair_features(question, answer))
    feats.update(numeric_consistency_features(question, answer))
    feats.update(verbosity_features(question, answer))
    feats.update(first_span_features(question, answer))
    return feats


def build_text_views(df: pd.DataFrame) -> dict[str, list[str]]:
    questions = [get_question_text(pd.Series(rec)) for rec in df.to_dict(orient="records")]
    answers = [get_answer_text(pd.Series(rec)) for rec in df.to_dict(orient="records")]
    answer_core = [_core_answer_span(a) for a in answers]
    answer_first_sentence = [_first_sentence(a) for a in answers]
    qa_concat = [f"Q: {q}\nA: {a}" for q, a in zip(questions, answers)]
    qa_core_concat = [f"Q: {q}\nA_CORE: {a}" for q, a in zip(questions, answer_core)]
    return {
        "question": questions,
        "answer": answers,
        "answer_core": answer_core,
        "answer_first_sentence": answer_first_sentence,
        "qa_concat": qa_concat,
        "qa_core_concat": qa_core_concat,
    }


def fit_text_vectorizers(df: pd.DataFrame) -> dict[str, TfidfVectorizer]:
    views = build_text_views(df)
    return {
        "question_char": TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), lowercase=True, min_df=2, sublinear_tf=True).fit(views["question"]),
        "answer_char": TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), lowercase=True, min_df=2, sublinear_tf=True).fit(views["answer"]),
        "answer_core_word": TfidfVectorizer(analyzer="word", ngram_range=(1, 2), lowercase=True, min_df=2, sublinear_tf=True).fit(views["answer_core"]),
        "answer_first_sentence_word": TfidfVectorizer(analyzer="word", ngram_range=(1, 2), lowercase=True, min_df=2, sublinear_tf=True).fit(views["answer_first_sentence"]),
        "qa_word": TfidfVectorizer(analyzer="word", ngram_range=(1, 2), lowercase=True, min_df=2, sublinear_tf=True).fit(views["qa_concat"]),
        "qa_core_word": TfidfVectorizer(analyzer="word", ngram_range=(1, 2), lowercase=True, min_df=2, sublinear_tf=True).fit(views["qa_core_concat"]),
    }


def transform_text_features(df: pd.DataFrame, vectorizers: dict[str, TfidfVectorizer]):
    views = build_text_views(df)
    dense_df = pd.DataFrame([extract_qa_consistency_dense_features_one(pd.Series(rec)) for rec in df.to_dict(orient="records")])
    mats = [
        vectorizers["question_char"].transform(views["question"]),
        vectorizers["answer_char"].transform(views["answer"]),
        vectorizers["answer_core_word"].transform(views["answer_core"]),
        vectorizers["answer_first_sentence_word"].transform(views["answer_first_sentence"]),
        vectorizers["qa_word"].transform(views["qa_concat"]),
        vectorizers["qa_core_word"].transform(views["qa_core_concat"]),
        sparse.csr_matrix(dense_df.astype(np.float32).values),
    ]
    return sparse.hstack(mats).tocsr(), dense_df


def build_feature_name_list(vectorizers: dict[str, TfidfVectorizer], dense_columns: list[str]) -> list[str]:
    names = []
    names.extend([f"question_char::{x}" for x in vectorizers["question_char"].get_feature_names_out().tolist()])
    names.extend([f"answer_char::{x}" for x in vectorizers["answer_char"].get_feature_names_out().tolist()])
    names.extend([f"answer_core_word::{x}" for x in vectorizers["answer_core_word"].get_feature_names_out().tolist()])
    names.extend([f"answer_first_sentence_word::{x}" for x in vectorizers["answer_first_sentence_word"].get_feature_names_out().tolist()])
    names.extend([f"qa_word::{x}" for x in vectorizers["qa_word"].get_feature_names_out().tolist()])
    names.extend([f"qa_core_word::{x}" for x in vectorizers["qa_core_word"].get_feature_names_out().tolist()])
    names.extend([f"dense::{x}" for x in dense_columns])
    return names
