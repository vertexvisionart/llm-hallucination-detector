#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm.auto import tqdm
from vllm import LLM, SamplingParams

from generate_answers_from_sberchallenge import parse_answers_field, read_table


HALLUCINATION_INSTRUCTIONS = {
    "entity_swap": (
        "Сгенерируй краткий, уверенный, но неверный ответ, подменив ключевую сущность, имя, страну, город, организацию или объект на похожий. "
        "Не меняй тип ответа: если правильный ответ состоит из сущности или короткой фразы, сохрани такой же формат."
    ),
    "date_number_swap": (
        "Сгенерируй краткий, уверенный, но неверный ответ, обязательно подменив дату, год, число, количество или сумму на правдоподобное, но неверное значение. "
        "В ответе обязательно должна присутствовать хотя бы одна цифра. Не заменяй numeric error на общий ложный факт."
    ),
    "cause_fact_swap": (
        "Сгенерируй краткий, уверенный, но неверный ответ, подменив причину, факт, характеристику или итог на соседний, но неправильный. "
        "Ответ должен звучать как обычный факт без пояснений."
    ),
    "off_context_confident": (
        "Сгенерируй краткий, уверенный, но неверный ответ, который звучит правдоподобно, но не подтверждается данным контекстом."
    ),
    "false_refusal": (
        "Сгенерируй именно ложный отказ, как если бы модель не нашла ответ в контексте. "
        "Ответ должен быть одной короткой фразой в стиле: не знаю, в контексте нет ответа, это не указано в контексте. "
        "Нельзя придумывать альтернативный факт. Нельзя писать ничего, кроме отказа."
    ),
}


EXTRA_RULES = {
    "entity_swap": "Замени сущность, но сохрани длину и формат ответа близкими к правильному.",
    "date_number_swap": "В ответе обязательно используй цифры. Не пиши пояснение о том, что число было изменено.",
    "cause_fact_swap": "Не добавляй кавычки, комментарии и мета-объяснения.",
    "off_context_confident": "Дай уверенный ложный факт одной строкой без оговорок.",
    "false_refusal": "Ответ должен быть именно отказом и содержать одну из формулировок: не знаю, в контексте нет ответа, это не указано в контексте.",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic hallucinations with vLLM")
    parser.add_argument("--input-path", type=str, default="artifacts_v3/gigachat_sberquad_labeling_dataset_with_manual.parquet")
    parser.add_argument("--output-path", type=str, default="synthetic_hallucinations_vllm_v1.parquet")
    parser.add_argument("--model-path", type=str, default="hf_cache/gigachat3_local")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--types", type=str, default="entity_swap,date_number_swap,cause_fact_swap,false_refusal")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--prompt-max-length", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--enforce-eager", action="store_true")
    return parser.parse_args()


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
    extra_rule = EXTRA_RULES[hallucination_type]

    return (
        "Ты создаёшь синтетический негативный пример для детектора галлюцинаций.\n"
        "Нужно выдать один краткий ответ на вопрос, который звучит правдоподобно, но фактически неверен относительно данного контекста.\n"
        "Не пиши объяснений, списков, JSON, кавычек и служебного текста. Верни только сам ответ одной строкой.\n"
        "Нельзя объяснять, почему ответ неверен. Нельзя ссылаться на инструкцию. Нельзя писать комментарии в скобках.\n"
        f"Тип ошибки: {hallucination_type}. {instruction}\n"
        f"Дополнительное правило: {extra_rule}\n\n"
        f"Заголовок: {title}\n\n"
        f"Контекст:\n{context}\n\n"
        f"Вопрос: {question}\n"
        f"Правильные ответы: {gold_text}\n\n"
        "Верни только сам ответ:"
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


def cleanup_generated_text(text: str) -> str:
    text = (text or "").strip().replace("\r\n", "\n")
    text = re.sub(r"^Ответ\s*[:\-]\s*", "", text, flags=re.IGNORECASE).strip()
    text = text.strip('"').strip("'").strip()
    text = text.split("\n\n")[0].strip()
    text = text.split("\n")[0].strip()
    return text


def save_rows(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, index=False)


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_path)
    output_path = Path(args.output_path)
    model_path = str(Path(args.model_path))

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

    jobs: list[dict[str, Any]] = []
    for row in selected.itertuples(index=False):
        row_s = pd.Series(row._asdict())
        source_id = row_s.get("id")
        for hallucination_type in hallucination_types:
            key = (source_id, hallucination_type)
            if key in done_keys:
                continue
            jobs.append({
                "source_id": source_id,
                "hallucination_type": hallucination_type,
                "row": row_s,
                "prompt": build_hallucination_prompt(row_s, hallucination_type),
            })

    print(f"selected base rows: {len(selected)}")
    print(f"pending generations: {len(jobs)}")

    if not jobs:
        print(f"nothing to do: {output_path}")
        return

    llm = LLM(
        model=model_path,
        tokenizer=model_path,
        trust_remote_code=True,
        dtype="bfloat16",
        max_model_len=args.prompt_max_length,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=args.enforce_eager,
    )
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_new_tokens,
        seed=args.seed,
    )

    prompts = [job["prompt"] for job in jobs]
    outputs = llm.generate(prompts, sampling_params, use_tqdm=True)

    for job, output in tqdm(list(zip(jobs, outputs)), total=len(jobs), desc="collect"):
        text = output.outputs[0].text if output.outputs else ""
        synthetic_answer = cleanup_generated_text(text)
        row_s = job["row"]
        out_row = {
            "source_id": job["source_id"],
            "split_name": row_s.get("split_name", args.split),
            "title": row_s.get("title", ""),
            "question": row_s.get("question", ""),
            "context": row_s.get("context", ""),
            "gold_answers_text": row_s.get("gold_answers_text", []),
            "best_gold_answer": row_s.get("best_gold_answer", ""),
            "model_answer": synthetic_answer,
            "hallucination_type": job["hallucination_type"],
            "label": 1,
            "synthetic_source": f"vllm::{model_path}",
            "generator_prompt": job["prompt"],
        }
        rows.append(out_row)

    save_rows(rows, output_path)
    print(f"saved: {output_path}")
    print(f"rows: {len(rows)}")
    print("by_type:")
    print(pd.DataFrame(rows)["hallucination_type"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()
