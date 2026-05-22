#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from tqdm.auto import tqdm

from generate_answers_from_sberchallenge import MODEL_ID, load_model_and_tokenizer, parse_answers_field, read_table


HALLUCINATION_INSTRUCTIONS = {
    "entity_swap": "Сгенерируй краткий, уверенный, но неверный ответ, подменив ключевую сущность, имя, страну, город, организацию или объект на похожий.",
    "date_number_swap": "Сгенерируй краткий, уверенный, но неверный ответ, подменив дату, год, число, количество или сумму на правдоподобное, но неверное.",
    "cause_fact_swap": "Сгенерируй краткий, уверенный, но неверный ответ, подменив причину, факт, характеристику или итог на соседний, но неправильный.",
    "off_context_confident": "Сгенерируй краткий, уверенный, но неверный ответ, который звучит правдоподобно, но не подтверждается данным контекстом.",
    "false_refusal": "Сгенерируй ответ в стиле 'не знаю' или 'в контексте нет ответа', даже если правильный ответ в контексте есть.",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic hallucinations with local GigaChat")
    parser.add_argument("--input-path", type=str, default="artifacts_v3/gigachat_sberquad_labeling_dataset.parquet")
    parser.add_argument("--output-path", type=str, default="synthetic_hallucinations_v1.parquet")
    parser.add_argument("--cache-dir", type=str, default="hf_cache")
    parser.add_argument("--model-id", type=str, default=MODEL_ID)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--types", type=str, default="entity_swap,date_number_swap,cause_fact_swap,false_refusal")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--prompt-max-length", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


@torch.no_grad()
def generate_text(
    prompt: str,
    model,
    tokenizer,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    prompt_max_length: int,
) -> str:
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=prompt_max_length, return_token_type_ids=False)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    generated = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    new_tokens = generated[0, inputs["input_ids"].shape[1] :]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    text = text.replace("\r\n", "\n")
    text = re.sub(r"^Ответ\s*[:\-]\s*", "", text, flags=re.IGNORECASE).strip()
    text = text.split("\n\n")[0].strip()
    text = text.split("\n")[0].strip()
    return text


def build_hallucination_prompt(row: pd.Series, hallucination_type: str) -> str:
    title = "" if pd.isna(row.get("title")) else str(row.get("title"))
    question = "" if pd.isna(row.get("question")) else str(row.get("question"))
    context = "" if pd.isna(row.get("context")) else str(row.get("context"))
    gold_answers = row.get("gold_answers_text")
    if gold_answers is None and "answers" in row:
        gold_answers = parse_answers_field(row.get("answers"))
    if isinstance(gold_answers, str):
        gold_answers = [gold_answers]
    if not isinstance(gold_answers, list):
        gold_answers = [] if gold_answers is None else list(gold_answers)
    gold_text = json.dumps(gold_answers, ensure_ascii=False)
    instruction = HALLUCINATION_INSTRUCTIONS[hallucination_type]

    return (
        "Ты создаёшь синтетический негативный пример для детектора галлюцинаций.\n"
        "Нужно выдать один краткий ответ на вопрос, который звучит правдоподобно, но фактически неверен относительно данного контекста.\n"
        "Не пиши объяснений, списков, JSON и служебного текста. Верни только сам ответ одной строкой.\n"
        f"Тип ошибки: {hallucination_type}. {instruction}\n\n"
        f"Заголовок: {title}\n\n"
        f"Контекст:\n{context}\n\n"
        f"Вопрос: {question}\n"
        f"Правильные ответы: {gold_text}\n\n"
        "Сгенерируй только неверный ответ:"
    )


def select_rows(df: pd.DataFrame, split: str, limit: int) -> pd.DataFrame:
    out = df.copy()
    if "split_name" in out.columns and split:
        out = out[out["split_name"] == split].copy()
    if "weak_label" in out.columns:
        out = out[out["weak_label"].fillna(-1).astype(float) == 0.0].copy()
    if "is_hallucination_manual" in out.columns:
        manual = out["is_hallucination_manual"]
        manual_false = manual.astype(str).str.lower().isin(["false", "0"]) | (manual == False)  # noqa: E712
        out = out[manual.isna() | manual_false].copy()
    return out.head(limit).reset_index(drop=True)


def save_rows(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, index=False)


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_path)
    output_path = Path(args.output_path)
    cache_dir = Path(args.cache_dir)

    hallucination_types = [item.strip() for item in args.types.split(",") if item.strip()]
    invalid = [item for item in hallucination_types if item not in HALLUCINATION_INSTRUCTIONS]
    if invalid:
        raise RuntimeError(f"Unknown hallucination types: {invalid}")

    df = read_table(input_path)
    selected = select_rows(df, split=args.split, limit=args.limit)
    if selected.empty:
        raise RuntimeError("No rows selected for synthetic hallucination generation")

    if output_path.exists() and not args.overwrite:
        existing = pd.read_parquet(output_path)
        rows = existing.to_dict(orient="records")
        done_keys = {(row["source_id"], row["hallucination_type"]) for row in rows}
        print(f"resume from existing file: {output_path}, rows={len(rows)}")
    else:
        rows = []
        done_keys = set()

    model, tokenizer = load_model_and_tokenizer(args.model_id, cache_dir)

    processed_since_save = 0
    total = len(selected) * len(hallucination_types)
    progress = tqdm(total=total)
    if done_keys:
        progress.update(sum((int(row.get("source_id") is not None and row.get("hallucination_type") is not None and (row["source_id"], row["hallucination_type"]) in done_keys) for row in rows)))

    for row in selected.itertuples(index=False):
        row_s = pd.Series(row._asdict())
        source_id = row_s.get("id")
        for hallucination_type in hallucination_types:
            key = (source_id, hallucination_type)
            if key in done_keys:
                progress.update(1)
                continue

            prompt = build_hallucination_prompt(row_s, hallucination_type)
            synthetic_answer = generate_text(
                prompt=prompt,
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                prompt_max_length=args.prompt_max_length,
            )

            out_row = {
                "source_id": source_id,
                "split_name": row_s.get("split_name", args.split),
                "title": row_s.get("title", ""),
                "question": row_s.get("question", ""),
                "context": row_s.get("context", ""),
                "gold_answers_text": row_s.get("gold_answers_text", []),
                "best_gold_answer": row_s.get("best_gold_answer", ""),
                "model_answer": synthetic_answer,
                "hallucination_type": hallucination_type,
                "label": 1,
                "synthetic_source": args.model_id,
                "generator_prompt": prompt,
            }
            rows.append(out_row)
            done_keys.add(key)
            processed_since_save += 1
            progress.update(1)

            if args.save_every > 0 and processed_since_save >= args.save_every:
                save_rows(rows, output_path)
                print(f"checkpoint saved: {output_path} rows={len(rows)}")
                processed_since_save = 0

    progress.close()
    save_rows(rows, output_path)
    print(f"saved: {output_path}")
    print(f"rows: {len(rows)}")
    print("by_type:")
    print(pd.DataFrame(rows)["hallucination_type"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()
