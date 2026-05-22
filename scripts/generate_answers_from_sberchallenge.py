#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Генерация ответов GigaChat3 на датасетах в формате SQuAD-подобного QA.

Поддерживает два варианта входной структуры:
1. datasets/train/*.parquet и datasets/validation/*.parquet
2. datasets/train-00000-of-00001.parquet и datasets/validation-00000-of-00001.parquet

Для каждой строки формирует prompt из title/context/question и сохраняет parquet
с дополнительными колонками:
- gold_answers_text
- prompt_text
- model_answer
- source_file
- split_name

Примеры запуска:
    python generate_answers_from_sberchallenge.py --limit 100
    python generate_answers_from_sberchallenge.py --datasets-dir datasets --outputs-dir outputs
    HF_TOKEN=hf_xxx python generate_answers_from_sberchallenge.py
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from huggingface_hub import login, snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm.auto import tqdm


MODEL_ID = "ai-sage/GigaChat3-10B-A1.8B-bf16"
LOCAL_MODEL_DIRNAME = "gigachat3_local"
SUPPORTED_EXTS = {".csv", ".parquet", ".jsonl", ".json"}


def can_use_cuda() -> bool:
    if not torch.cuda.is_available():
        return False

    try:
        major, minor = torch.cuda.get_device_capability(0)
        current_arch = f"sm_{major}{minor}"
        return current_arch in set(torch.cuda.get_arch_list())
    except Exception:
        return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", type=str, default=MODEL_ID)
    parser.add_argument("--datasets-dir", type=str, default="datasets")
    parser.add_argument("--cache-dir", type=str, default="hf_cache")
    parser.add_argument("--outputs-dir", type=str, default="outputs")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--prompt-max-length", type=int, default=4096)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--save-every",
        type=int,
        default=50,
        help="Сохранять промежуточный parquet каждые N обработанных строк.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Перезаписывать готовые parquet в outputs.",
    )
    return parser.parse_args()


def hf_login_if_needed() -> str | None:
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    if token:
        login(token=token)
    return token


def ensure_local_model_dir(model_id: str, cache_dir: Path, token: str | None) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    local_dir = cache_dir / LOCAL_MODEL_DIRNAME

    if (local_dir / "config.json").exists() and (local_dir / "tokenizer.json").exists():
        return local_dir

    if not token:
        raise RuntimeError(
            "Локальная копия модели не найдена и HF_TOKEN не задан. "
            "Либо положите модель в hf_cache/gigachat3_local, либо экспортируйте HF_TOKEN."
        )

    snapshot_download(
        repo_id=model_id,
        local_dir=str(local_dir),
        token=token,
        local_dir_use_symlinks=False,
    )
    return local_dir


def patch_config_if_needed(local_dir: Path) -> None:
    config_path = local_dir / "config.json"
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    changed = False
    if "routed_scaling_factor" in cfg and isinstance(cfg["routed_scaling_factor"], int):
        cfg["routed_scaling_factor"] = float(cfg["routed_scaling_factor"])
        changed = True

    if changed:
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)


def load_model_and_tokenizer(model_id: str, cache_dir: Path):
    token = hf_login_if_needed()
    local_dir = ensure_local_model_dir(model_id=model_id, cache_dir=cache_dir, token=token)
    patch_config_if_needed(local_dir)

    tokenizer = AutoTokenizer.from_pretrained(
        str(local_dir),
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    if can_use_cuda():
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        device = torch.device("cuda")
    else:
        dtype = torch.float32
        device = torch.device("cpu")
        if torch.cuda.is_available():
            print("CUDA доступна, но текущая сборка PyTorch не поддерживает compute capability этого GPU. Использую CPU.")

    model_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": dtype,
    }

    try:
        import accelerate  # noqa: F401
        model_kwargs["device_map"] = "auto"
    except ImportError:
        pass

    model = AutoModelForCausalLM.from_pretrained(
        str(local_dir),
        **model_kwargs,
    )

    if "device_map" not in model_kwargs:
        model.to(device)

    model.eval()

    loaded_device = next(model.parameters()).device
    print(f"Model loaded from: {local_dir}")
    print(f"Model dtype: {dtype}")
    print(f"Model loaded on: {loaded_device}")
    return model, tokenizer


def infer_split_name(path: Path) -> str:
    parts = [part.lower() for part in path.parts]
    for name in ("train", "validation", "valid", "val", "test"):
        if name in parts or path.name.lower().startswith(f"{name}-"):
            return "validation" if name in {"validation", "valid", "val"} else name
    return "unspecified"


def collect_dataset_files(datasets_dir: Path) -> list[tuple[str, Path]]:
    files: list[tuple[str, Path]] = []
    for path in sorted(datasets_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTS:
            files.append((infer_split_name(path), path))
    return files


def read_table(path: Path) -> pd.DataFrame:
    ext = path.suffix.lower()
    if ext == ".csv":
        return pd.read_csv(path)
    if ext == ".parquet":
        return pd.read_parquet(path)
    if ext == ".jsonl":
        return pd.read_json(path, lines=True)
    if ext == ".json":
        try:
            return pd.read_json(path)
        except ValueError:
            with open(path, "r", encoding="utf-8") as f:
                obj = json.load(f)
            return pd.DataFrame(obj)
    raise ValueError(f"Неподдерживаемый формат: {path}")


def flatten_to_strings(value: Any) -> list[str]:
    if value is None:
        return []

    if isinstance(value, (list, tuple, set)):
        result: list[str] = []
        for item in value:
            result.extend(flatten_to_strings(item))
        return result

    shape = getattr(value, "shape", None)
    tolist = getattr(value, "tolist", None)
    if callable(tolist) and shape is not None:
        return flatten_to_strings(tolist())

    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []

    return [str(value)]


def parse_answers_field(x: Any) -> list[str]:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return []

    if isinstance(x, dict):
        return flatten_to_strings(x.get("text", []))

    if isinstance(x, (list, tuple, set)):
        return flatten_to_strings(x)

    if isinstance(x, str):
        s = x.strip()
        if not s:
            return []
        try:
            obj = json.loads(s)
            if isinstance(obj, dict):
                return flatten_to_strings(obj.get("text", []))
            if isinstance(obj, list):
                return flatten_to_strings(obj)
        except Exception:
            pass
        try:
            obj = ast.literal_eval(s)
            if isinstance(obj, dict):
                return flatten_to_strings(obj.get("text", []))
            if isinstance(obj, list):
                return flatten_to_strings(obj)
        except Exception:
            pass
        return [s]

    return flatten_to_strings(x)


def build_prompt(row: pd.Series) -> str:
    title = "" if pd.isna(row.get("title")) else str(row.get("title"))
    context = "" if pd.isna(row.get("context")) else str(row.get("context"))
    question = "" if pd.isna(row.get("question")) else str(row.get("question"))

    return (
        "Ответь на вопрос по данному контексту кратко и строго по фактам из контекста. "
        "Если в контексте нет ответа, напиши 'не знаю'.\n\n"
        f"Заголовок: {title}\n\n"
        f"Контекст:\n{context}\n\n"
        f"Вопрос: {question}\n"
        "Ответ:"
    )


@torch.no_grad()
def generate_answer(
    prompt: str,
    model,
    tokenizer,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    do_sample: bool,
    prompt_max_length: int,
) -> str:
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=prompt_max_length, return_token_type_ids=False)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    generated = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature if do_sample else None,
        top_p=top_p if do_sample else None,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    new_tokens = generated[0, inputs["input_ids"].shape[1] :]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    text = text.replace("\r\n", "\n")

    cleanup_patterns = [
        r"\n+не знаю\.?$",
        r"\n+не могу ответить\.?$",
        r"\n+в контексте нет ответа\.?$",
    ]
    for pattern in cleanup_patterns:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE).strip()

    return text


def save_rows(rows: list[dict[str, Any]], save_path: Path) -> None:
    pd.DataFrame(rows).to_parquet(save_path, index=False)


def process_file(
    split_name: str,
    file_path: Path,
    outputs_dir: Path,
    model,
    tokenizer,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    do_sample: bool,
    prompt_max_length: int,
    limit: int | None,
    save_every: int,
    overwrite: bool,
) -> Path | None:
    df = read_table(file_path)

    required_cols = {"context", "question"}
    missing = required_cols - set(df.columns)
    if missing:
        print(f"skip {file_path}: нет обязательных колонок {missing}")
        return None

    if limit is not None:
        df = df.head(limit).copy()

    split_out_dir = outputs_dir / split_name
    split_out_dir.mkdir(parents=True, exist_ok=True)
    out_name = file_path.stem + "_with_answers.parquet"
    save_path = split_out_dir / out_name

    total_rows = len(df)
    rows: list[dict[str, Any]] = []
    start_idx = 0

    if save_path.exists() and not overwrite:
        existing_df = pd.read_parquet(save_path)
        existing_count = len(existing_df)
        if existing_count >= total_rows:
            print(f"skip {file_path}: уже существует полный результат {save_path}")
            return save_path
        rows = existing_df.to_dict(orient="records")
        start_idx = existing_count
        print(f"resume [{split_name}]: {file_path} from row {start_idx}/{total_rows}")
    else:
        print(f"processing [{split_name}]: {file_path}")

    if start_idx > 0:
        iterator_df = df.iloc[start_idx:]
    else:
        iterator_df = df

    processed_since_save = 0
    for row in tqdm(iterator_df.itertuples(index=False), total=len(iterator_df), initial=0):
        row_s = pd.Series(row._asdict())
        prompt_text = build_prompt(row_s)
        model_answer = generate_answer(
            prompt=prompt_text,
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            prompt_max_length=prompt_max_length,
        )
        gold_answers = parse_answers_field(row_s.get("answers"))

        out_row = row_s.to_dict()
        out_row["gold_answers_text"] = gold_answers
        out_row["prompt_text"] = prompt_text
        out_row["model_answer"] = model_answer
        out_row["source_file"] = str(file_path)
        out_row["split_name"] = split_name
        rows.append(out_row)
        processed_since_save += 1

        if save_every > 0 and processed_since_save >= save_every:
            save_rows(rows, save_path)
            print(f"checkpoint saved: {save_path} ({len(rows)}/{total_rows})")
            processed_since_save = 0

    save_rows(rows, save_path)
    print(f"saved: {save_path} ({len(rows)}/{total_rows})")
    return save_path


def main() -> None:
    args = parse_args()

    datasets_dir = Path(args.datasets_dir)
    cache_dir = Path(args.cache_dir)
    outputs_dir = Path(args.outputs_dir)

    dataset_files = collect_dataset_files(datasets_dir)
    if not dataset_files:
        raise RuntimeError(f"В {datasets_dir} не найдено файлов датасета")

    print("Found dataset files:")
    for split_name, file_path in dataset_files:
        print(f"- [{split_name}] {file_path}")

    model, tokenizer = load_model_and_tokenizer(args.model_id, cache_dir)

    saved_files: list[Path] = []
    for split_name, file_path in dataset_files:
        result = process_file(
            split_name=split_name,
            file_path=file_path,
            outputs_dir=outputs_dir,
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            do_sample=args.do_sample,
            prompt_max_length=args.prompt_max_length,
            limit=args.limit,
            save_every=args.save_every,
            overwrite=args.overwrite,
        )
        if result is not None:
            saved_files.append(result)

    print("\nDone.")
    print("Saved files:")
    for path in saved_files:
        print(f"- {path}")


if __name__ == "__main__":
    main()
