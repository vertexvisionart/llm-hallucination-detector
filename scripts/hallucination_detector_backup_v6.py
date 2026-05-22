#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import os
import re
import string
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from huggingface_hub import login, snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "ai-sage/GigaChat3-10B-A1.8B-bf16"
LOCAL_MODEL_DIRNAME = "gigachat3_local"


def can_use_cuda() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        major, minor = torch.cuda.get_device_capability(0)
        current_arch = f"sm_{major}{minor}"
        return current_arch in set(torch.cuda.get_arch_list())
    except Exception:
        return False


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

    tokenizer = AutoTokenizer.from_pretrained(str(local_dir), trust_remote_code=True)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    if can_use_cuda():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        device = torch.device("cuda")
    else:
        dtype = torch.float32
        device = torch.device("cpu")

    model_kwargs = {"trust_remote_code": True, "dtype": dtype, "low_cpu_mem_usage": True}
    try:
        import accelerate  # noqa: F401
        model_kwargs["device_map"] = "auto"
    except Exception:
        pass

    model = AutoModelForCausalLM.from_pretrained(str(local_dir), **model_kwargs)
    if "device_map" not in model_kwargs:
        model.to(device)
    model.eval()
    return model, tokenizer


def normalize_text(text: str) -> str:
    text = "" if text is None else str(text)
    text = text.lower().strip().replace("ё", "е")
    text = re.sub(r"\s+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation + "«»“”„…"))
    return text.strip()


def simple_tokens(text: str) -> list[str]:
    return [tok for tok in normalize_text(text).split() if tok]


def count_numbers(text: str) -> int:
    return len(re.findall(r"\d+", text or ""))


def count_upper_tokens(text: str) -> int:
    if not text:
        return 0
    return sum(1 for tok in re.findall(r"\b\w+\b", text) if tok[:1].isupper())


def repeated_unigram_ratio(tokens: list[int]) -> float:
    if not tokens:
        return 0.0
    return 1.0 - len(set(tokens)) / max(len(tokens), 1)


def repeated_bigram_ratio(tokens: list[int]) -> float:
    if len(tokens) < 2:
        return 0.0
    bigrams = list(zip(tokens[:-1], tokens[1:]))
    return 1.0 - len(set(bigrams)) / max(len(bigrams), 1)


def overlap_ratio(a: list[str], b: list[str]) -> float:
    if not a or not b:
        return 0.0
    a_set = set(a)
    b_set = set(b)
    return len(a_set & b_set) / max(len(a_set), 1)


def jaccard_ratio(a: list[str], b: list[str]) -> float:
    if not a and not b:
        return 0.0
    a_set = set(a)
    b_set = set(b)
    union = a_set | b_set
    if not union:
        return 0.0
    return len(a_set & b_set) / len(union)


def build_prompt_text(row: pd.Series) -> str:
    if "prompt_text" in row and pd.notna(row.get("prompt_text")):
        return str(row.get("prompt_text"))
    if "prompt" in row and pd.notna(row.get("prompt")):
        return str(row.get("prompt"))
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


def build_full_text(prompt_text: str, response: str) -> str:
    if not prompt_text:
        return response
    if prompt_text.endswith((" ", "\n", "\t")):
        return prompt_text + response
    return prompt_text + " " + response


def _safe_stats(values: np.ndarray, prefix: str) -> dict[str, float]:
    if values.size == 0:
        return {
            f"{prefix}_mean": 0.0,
            f"{prefix}_std": 0.0,
            f"{prefix}_min": 0.0,
            f"{prefix}_max": 0.0,
        }
    return {
        f"{prefix}_mean": float(values.mean()),
        f"{prefix}_std": float(values.std()),
        f"{prefix}_min": float(values.min()),
        f"{prefix}_max": float(values.max()),
    }


def _safe_percentile(values: np.ndarray, q: float) -> float:
    if values.size == 0:
        return 0.0
    return float(np.percentile(values, q))


def _safe_max_window_mean(values: np.ndarray, window: int) -> float:
    if values.size == 0 or window <= 0:
        return 0.0
    if values.size < window:
        return float(values.mean())
    kernel = np.ones(window, dtype=np.float32) / float(window)
    window_means = np.convolve(values.astype(np.float32), kernel, mode="valid")
    return float(window_means.max()) if window_means.size else 0.0


def _safe_max_drop(values: np.ndarray) -> float:
    if values.size < 2:
        return 0.0
    diffs = values[:-1] - values[1:]
    return float(np.maximum(diffs, 0.0).max())


def _safe_variance(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    return float(values.var())


def _safe_tail_degradation(values: np.ndarray, frac: float = 0.3) -> float:
    if values.size == 0:
        return 0.0
    span = max(1, int(np.ceil(values.size * frac)))
    head = values[:span]
    tail = values[-span:]
    return float(head.mean() - tail.mean())


def _safe_worst_span_mean(values: np.ndarray, window: int, mode: str) -> float:
    if values.size == 0:
        return 0.0
    if values.size < window:
        return float(values.mean())
    kernel = np.ones(window, dtype=np.float32) / float(window)
    window_means = np.convolve(values.astype(np.float32), kernel, mode="valid")
    if window_means.size == 0:
        return 0.0
    if mode == "min":
        return float(window_means.min())
    return float(window_means.max())


def _safe_first_k_mean(values: np.ndarray, k: int) -> float:
    if values.size == 0 or k <= 0:
        return 0.0
    return float(values[: min(k, values.size)].mean())


def _resolve_hidden_state_idx(layer_num: int, hidden_states: tuple[torch.Tensor, ...]) -> int:
    if not hidden_states:
        return 0
    return max(0, min(layer_num, len(hidden_states) - 1))


def _safe_tensor_mean(values: torch.Tensor) -> float:
    if values.numel() == 0:
        return 0.0
    return float(values.mean().item())


def _safe_tensor_std(values: torch.Tensor) -> float:
    if values.numel() == 0:
        return 0.0
    return float(values.std(unbiased=False).item())


def _layer_distribution_stats(
    model,
    hidden_states: tuple[torch.Tensor, ...],
    layer_idx: int,
    start_idx: int,
) -> tuple[torch.Tensor, float, float]:
    layer_hidden = hidden_states[layer_idx][:, :-1, :]
    resp_hidden = layer_hidden[:, start_idx:, :]
    if resp_hidden.numel() == 0:
        empty = torch.empty(0, device=layer_hidden.device, dtype=torch.float32)
        return empty, 0.0, 0.0

    layer_logits = model.lm_head(resp_hidden).float()
    layer_log_probs = F.log_softmax(layer_logits, dim=-1)
    layer_probs = layer_log_probs.exp()
    layer_entropy = -(layer_probs * layer_log_probs).sum(dim=-1)
    layer_top1_probs = layer_probs.max(dim=-1).values
    return layer_log_probs, _safe_tensor_mean(layer_entropy), _safe_tensor_mean(layer_top1_probs)


def _mean_token_kl(log_probs_p: torch.Tensor, log_probs_q: torch.Tensor) -> float:
    if log_probs_p.numel() == 0 or log_probs_q.numel() == 0:
        return 0.0
    token_kl = F.kl_div(log_probs_p, log_probs_q.exp(), reduction="none").sum(dim=-1)
    return _safe_tensor_mean(token_kl)


def _cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return 0.0
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def _safe_alignment_stats(response_hidden: np.ndarray, context_hidden: np.ndarray) -> dict[str, float]:
    if response_hidden.size == 0 or context_hidden.size == 0:
        return {
            "ctx_align_maxcos_mean": 0.0,
            "ctx_align_maxcos_std": 0.0,
            "ctx_align_maxcos_min": 0.0,
            "ctx_align_maxcos_p10": 0.0,
            "ctx_align_low_support_rate": 0.0,
        }

    response_norm = response_hidden / np.clip(np.linalg.norm(response_hidden, axis=1, keepdims=True), 1e-12, None)
    context_norm = context_hidden / np.clip(np.linalg.norm(context_hidden, axis=1, keepdims=True), 1e-12, None)
    sim = response_norm @ context_norm.T
    max_per_response = sim.max(axis=1)
    return {
        "ctx_align_maxcos_mean": float(max_per_response.mean()),
        "ctx_align_maxcos_std": float(max_per_response.std()),
        "ctx_align_maxcos_min": float(max_per_response.min()),
        "ctx_align_maxcos_p10": _safe_percentile(max_per_response, 10),
        "ctx_align_low_support_rate": float((max_per_response < 0.3).mean()),
    }


@torch.inference_mode()
def extract_features_one(
    row: pd.Series,
    model=None,
    tokenizer=None,
    include_internal_features: bool = True,
    max_length: int = 4096,
) -> dict[str, Any]:
    prompt_text = build_prompt_text(row)
    response = "" if pd.isna(row.get("model_answer")) else str(row.get("model_answer"))
    question = "" if pd.isna(row.get("question")) else str(row.get("question"))
    context = "" if pd.isna(row.get("context")) else str(row.get("context"))

    question_tokens = simple_tokens(question)
    context_tokens = simple_tokens(context)
    response_tokens_simple = simple_tokens(response)

    features: dict[str, Any] = {
        "answer_char_len": len(response),
        "answer_word_len": len(response_tokens_simple),
        "answer_line_count": response.count("\n") + 1 if response else 0,
        "answer_digit_count": count_numbers(response),
        "answer_upper_token_count": count_upper_tokens(response),
        "question_digit_count": count_numbers(question),
        "context_digit_count": count_numbers(context),
        "question_answer_overlap": overlap_ratio(response_tokens_simple, question_tokens),
        "context_answer_overlap": overlap_ratio(response_tokens_simple, context_tokens),
        "question_answer_jaccard": jaccard_ratio(response_tokens_simple, question_tokens),
        "context_answer_jaccard": jaccard_ratio(response_tokens_simple, context_tokens),
        "answer_in_context": float(normalize_text(response) in normalize_text(context)) if response and context else 0.0,
        "context_len_chars": len(context),
        "question_len_chars": len(question),
    }

    if not include_internal_features:
        return features

    if model is None or tokenizer is None:
        raise ValueError("model/tokenizer required when include_internal_features=True")

    full_text = build_full_text(prompt_text, response)
    full_enc = tokenizer(
        full_text,
        return_tensors="pt",
        add_special_tokens=True,
        truncation=True,
        max_length=max_length,
    )
    resp_enc = tokenizer(
        response,
        return_tensors="pt",
        add_special_tokens=False,
        truncation=True,
        max_length=max_length,
    )

    input_ids = full_enc["input_ids"].to(model.device)
    attention_mask = full_enc["attention_mask"].to(model.device)
    total_tokens = input_ids.shape[1]
    response_len = int(resp_enc["input_ids"].shape[1]) if response else 0
    response_token_ids = resp_enc["input_ids"][0].tolist() if response_len else []

    context_token_ids = []
    if context:
        context_enc = tokenizer(
            context,
            return_tensors="pt",
            add_special_tokens=False,
            truncation=True,
            max_length=max_length,
        )
        context_token_ids = context_enc["input_ids"][0].tolist()
    prompt_prefix_token_ids = input_ids[0, : max(0, total_tokens - response_len)].tolist()

    features["response_token_len"] = response_len
    features["response_repeated_unigram_ratio"] = repeated_unigram_ratio(response_token_ids)
    features["response_repeated_bigram_ratio"] = repeated_bigram_ratio(response_token_ids)

    if response_len == 0:
        features.update(_safe_stats(np.array([]), "token_logprob"))
        features.update(_safe_stats(np.array([]), "token_entropy"))
        features.update({
            "logprob_p05": 0.0,
            "entropy_max_window_3": 0.0,
            "entropy_max_window_5": 0.0,
            "logprob_max_drop": 0.0,
            "entropy_variance": 0.0,
            "logprob_tail_degradation": 0.0,
            "worst_span_logprob_mean": 0.0,
            "worst_span_entropy_mean": 0.0,
            "first_5_tokens_entropy": 0.0,
            "top1_match_rate": 0.0,
            "top1_prob_mean": 0.0,
            "last_hidden_norm_mean": 0.0,
            "last_hidden_norm_std": 0.0,
            "mid_hidden_norm_mean": 0.0,
            "early_hidden_norm_mean": 0.0,
            "last_mid_cos": 0.0,
            "last_early_cos": 0.0,
            "mid_early_cos": 0.0,
            "hidden_state_cosine_diff": 0.0,
            "layer_disagreement_score": 0.0,
            "logit_lens_l10_entropy_mean": 0.0,
            "logit_lens_l20_entropy_mean": 0.0,
            "logit_lens_l30_entropy_mean": 0.0,
            "logit_lens_early_late_entropy_gap": 0.0,
            "logit_lens_top1_prob_trajectory_std": 0.0,
            "layer_kl_10_20_mean": 0.0,
            "layer_kl_20_30_mean": 0.0,
            "layer_kl_30_last_mean": 0.0,
            "layer_kl_total_mean": 0.0,
            "ctx_align_maxcos_mean": 0.0,
            "ctx_align_maxcos_std": 0.0,
            "ctx_align_maxcos_min": 0.0,
            "ctx_align_maxcos_p10": 0.0,
            "ctx_align_low_support_rate": 0.0,
        })
        return features

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
        use_cache=False,
    )

    logits = outputs.logits
    hidden_states = outputs.hidden_states

    resp_start = max(1, total_tokens - response_len)

    shift_logits = logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]
    log_probs = F.log_softmax(shift_logits, dim=-1)
    probs = torch.softmax(shift_logits, dim=-1)

    token_logprobs = log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)
    entropy = -(probs * log_probs).sum(dim=-1)
    top1 = shift_logits.argmax(dim=-1)
    top1_match = (top1 == shift_labels).float()
    top1_probs = probs.max(dim=-1).values

    start_idx = max(0, resp_start - 1)
    resp_token_logprobs = token_logprobs[:, start_idx:]
    resp_entropy = entropy[:, start_idx:]
    resp_top1_match = top1_match[:, start_idx:]
    resp_top1_probs = top1_probs[:, start_idx:]

    resp_token_logprobs_np = resp_token_logprobs[0].float().cpu().numpy()
    resp_entropy_np = resp_entropy[0].float().cpu().numpy()

    features.update(_safe_stats(resp_token_logprobs_np, "token_logprob"))
    features.update(_safe_stats(resp_entropy_np, "token_entropy"))
    features["logprob_min"] = float(resp_token_logprobs_np.min()) if resp_token_logprobs_np.size else 0.0
    features["logprob_p05"] = _safe_percentile(resp_token_logprobs_np, 5)
    features["logprob_p10"] = _safe_percentile(resp_token_logprobs_np, 10)
    features["entropy_max_window_3"] = _safe_max_window_mean(resp_entropy_np, 3)
    features["entropy_max_window_5"] = _safe_max_window_mean(resp_entropy_np, 5)
    features["logprob_max_drop"] = _safe_max_drop(resp_token_logprobs_np)
    features["entropy_variance"] = _safe_variance(resp_entropy_np)
    features["logprob_tail_degradation"] = _safe_tail_degradation(resp_token_logprobs_np)
    features["worst_span_logprob_mean"] = _safe_worst_span_mean(resp_token_logprobs_np, 5, mode="min")
    features["worst_span_entropy_mean"] = _safe_worst_span_mean(resp_entropy_np, 5, mode="max")
    features["first_5_tokens_entropy"] = _safe_first_k_mean(resp_entropy_np, 5)
    features["top1_match_rate"] = float(resp_top1_match.mean().item()) if resp_top1_match.numel() else 0.0
    features["top1_prob_mean"] = float(resp_top1_probs.mean().item()) if resp_top1_probs.numel() else 0.0

    layer_count = len(hidden_states)
    early_idx = min(8, layer_count - 1)
    mid_idx = layer_count // 2
    lens_l10_idx = _resolve_hidden_state_idx(10, hidden_states)
    lens_l20_idx = _resolve_hidden_state_idx(20, hidden_states)
    lens_l30_idx = _resolve_hidden_state_idx(30, hidden_states)
    lens_last_idx = layer_count - 1

    last_hidden = hidden_states[-1][0, resp_start:, :].float().cpu().numpy()
    mid_hidden = hidden_states[mid_idx][0, resp_start:, :].float().cpu().numpy()
    early_hidden = hidden_states[early_idx][0, resp_start:, :].float().cpu().numpy()

    log_probs_l10, entropy_l10_mean, top1_prob_l10_mean = _layer_distribution_stats(
        model=model,
        hidden_states=hidden_states,
        layer_idx=lens_l10_idx,
        start_idx=start_idx,
    )
    log_probs_l20, entropy_l20_mean, top1_prob_l20_mean = _layer_distribution_stats(
        model=model,
        hidden_states=hidden_states,
        layer_idx=lens_l20_idx,
        start_idx=start_idx,
    )
    log_probs_l30, entropy_l30_mean, top1_prob_l30_mean = _layer_distribution_stats(
        model=model,
        hidden_states=hidden_states,
        layer_idx=lens_l30_idx,
        start_idx=start_idx,
    )
    log_probs_last, entropy_last_mean, top1_prob_last_mean = _layer_distribution_stats(
        model=model,
        hidden_states=hidden_states,
        layer_idx=lens_last_idx,
        start_idx=start_idx,
    )

    top1_prob_trajectory = torch.tensor(
        [top1_prob_l10_mean, top1_prob_l20_mean, top1_prob_l30_mean, top1_prob_last_mean],
        dtype=torch.float32,
    )
    layer_kl_10_20_mean = _mean_token_kl(log_probs_l10, log_probs_l20)
    layer_kl_20_30_mean = _mean_token_kl(log_probs_l20, log_probs_l30)
    layer_kl_30_last_mean = _mean_token_kl(log_probs_l30, log_probs_last)

    features.update({
        "logit_lens_l10_entropy_mean": entropy_l10_mean,
        "logit_lens_l20_entropy_mean": entropy_l20_mean,
        "logit_lens_l30_entropy_mean": entropy_l30_mean,
        "logit_lens_early_late_entropy_gap": entropy_l10_mean - entropy_last_mean,
        "logit_lens_top1_prob_trajectory_std": _safe_tensor_std(top1_prob_trajectory),
        "layer_kl_10_20_mean": layer_kl_10_20_mean,
        "layer_kl_20_30_mean": layer_kl_20_30_mean,
        "layer_kl_30_last_mean": layer_kl_30_last_mean,
        "layer_kl_total_mean": float(np.mean([
            layer_kl_10_20_mean,
            layer_kl_20_30_mean,
            layer_kl_30_last_mean,
        ])),
    })

    if last_hidden.size == 0:
        features.update({
            "last_hidden_norm_mean": 0.0,
            "last_hidden_norm_std": 0.0,
            "mid_hidden_norm_mean": 0.0,
            "early_hidden_norm_mean": 0.0,
            "last_mid_cos": 0.0,
            "last_early_cos": 0.0,
            "mid_early_cos": 0.0,
            "hidden_state_cosine_diff": 0.0,
            "layer_disagreement_score": 0.0,
            "logit_lens_l10_entropy_mean": 0.0,
            "logit_lens_l20_entropy_mean": 0.0,
            "logit_lens_l30_entropy_mean": 0.0,
            "logit_lens_early_late_entropy_gap": 0.0,
            "logit_lens_top1_prob_trajectory_std": 0.0,
            "layer_kl_10_20_mean": 0.0,
            "layer_kl_20_30_mean": 0.0,
            "layer_kl_30_last_mean": 0.0,
            "layer_kl_total_mean": 0.0,
            "ctx_align_maxcos_mean": 0.0,
            "ctx_align_maxcos_std": 0.0,
            "ctx_align_maxcos_min": 0.0,
            "ctx_align_maxcos_p10": 0.0,
            "ctx_align_low_support_rate": 0.0,
        })
        return features

    last_norms = np.linalg.norm(last_hidden, axis=1)
    mid_norms = np.linalg.norm(mid_hidden, axis=1)
    early_norms = np.linalg.norm(early_hidden, axis=1)

    last_mean_vec = last_hidden.mean(axis=0)
    mid_mean_vec = mid_hidden.mean(axis=0)
    early_mean_vec = early_hidden.mean(axis=0)

    features.update({
        "last_hidden_norm_mean": float(last_norms.mean()),
        "last_hidden_norm_std": float(last_norms.std()),
        "mid_hidden_norm_mean": float(mid_norms.mean()),
        "early_hidden_norm_mean": float(early_norms.mean()),
        "last_mid_cos": _cos_sim(last_mean_vec, mid_mean_vec),
        "last_early_cos": _cos_sim(last_mean_vec, early_mean_vec),
        "mid_early_cos": _cos_sim(mid_mean_vec, early_mean_vec),
    })
    pairwise_cos = [
        features["last_mid_cos"],
        features["last_early_cos"],
        features["mid_early_cos"],
    ]
    features["hidden_state_cosine_diff"] = max(pairwise_cos) - min(pairwise_cos)
    features["layer_disagreement_score"] = 1.0 - float(np.mean(pairwise_cos))

    context_token_span_start = -1
    context_token_span_end = -1
    if context_token_ids and len(context_token_ids) <= len(prompt_prefix_token_ids):
        for start in range(0, len(prompt_prefix_token_ids) - len(context_token_ids) + 1):
            if prompt_prefix_token_ids[start:start + len(context_token_ids)] == context_token_ids:
                context_token_span_start = start
                context_token_span_end = start + len(context_token_ids)
                break

    if context_token_span_start >= 0:
        context_hidden = hidden_states[-1][0, context_token_span_start:context_token_span_end, :].float().cpu().numpy()
    else:
        context_hidden = hidden_states[-1][0, :resp_start, :].float().cpu().numpy()

    features.update(_safe_alignment_stats(last_hidden, context_hidden))
    return features
