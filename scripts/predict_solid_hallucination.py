#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from catboost import CatBoostClassifier

from hallucination_detector import MODEL_ID, extract_features_one, load_model_and_tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch-independent solid hallucination predictor")
    parser.add_argument("--input-path", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("--manifest-path", type=str, default="solidification_clean_manifest_v1.json")
    parser.add_argument("--cache-dir", type=str, default="hf_cache")
    parser.add_argument("--model-id", type=str, default=MODEL_ID)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--include-component-scores", action="store_true")
    return parser.parse_args()


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported input format: {path}")


def _load_catboost_models(manifest: dict[str, Any]):
    models = {}
    for spec in manifest["atomic_catboost_models"]:
        clf = CatBoostClassifier()
        clf.load_model(spec["model_path"])
        cols = json.loads(Path(spec["feature_columns_path"]).read_text(encoding="utf-8"))
        models[spec["name"]] = {
            "model": clf,
            "feature_columns": cols,
        }
    return models


def _load_qa_models(manifest: dict[str, Any]):
    models = {}
    for spec in manifest["qa_models"]:
        bundle = joblib.load(spec["model_path"])
        module = importlib.import_module(spec["module"])
        models[spec["name"]] = {
            "model": bundle["model"],
            "vectorizers": bundle["vectorizers"],
            "transform": module.transform_text_features,
        }
    return models


def _load_meta_model(manifest: dict[str, Any]):
    spec = manifest.get("meta_model")
    if not spec:
        return None
    model_type = spec.get("type", "catboost")
    if model_type == "catboost":
        clf = CatBoostClassifier()
        clf.load_model(spec["model_path"])
    elif model_type == "joblib_sklearn":
        bundle = joblib.load(spec["model_path"])
        clf = bundle["model"]
    else:
        raise ValueError(f"Unsupported meta model type: {model_type}")
    feature_columns = json.loads(Path(spec["feature_columns_path"]).read_text(encoding="utf-8"))
    return {
        "model": clf,
        "feature_columns": feature_columns,
        "type": model_type,
    }


def _prepare_cat_features(feats: dict[str, Any], feature_columns: list[str]) -> pd.DataFrame:
    X = pd.DataFrame([feats])
    missing = [col for col in feature_columns if col not in X.columns]
    for col in missing:
        X[col] = 0.0
    return X[feature_columns]


def _blend_scores(score_map: dict[str, float], blend_spec: list[dict[str, Any]]) -> float:
    num = 0.0
    den = 0.0
    for spec in blend_spec:
        score = float(score_map[spec["name"]])
        if spec.get("invert", False):
            score = 1.0 - score
        weight = float(spec["weight"])
        num += weight * score
        den += weight
    return num / den if den else 0.0


def _build_meta_features(score_map: dict[str, float]) -> dict[str, float]:
    feat = {f"score::{name}": float(score) for name, score in score_map.items()}
    vals = list(feat.values())
    if vals:
        series = pd.Series(vals, dtype=float)
        feat["meta::mean"] = float(series.mean())
        feat["meta::std"] = float(series.std(ddof=0))
        feat["meta::min"] = float(series.min())
        feat["meta::max"] = float(series.max())
        feat["meta::range"] = feat["meta::max"] - feat["meta::min"]
    else:
        feat["meta::mean"] = 0.0
        feat["meta::std"] = 0.0
        feat["meta::min"] = 0.0
        feat["meta::max"] = 0.0
        feat["meta::range"] = 0.0
    if "qa_consistency_v1" in score_map and "qa_consistency_v2" in score_map:
        feat["meta::qa_gap_v1_v2"] = float(score_map["qa_consistency_v1"] - score_map["qa_consistency_v2"])
    if "internal_mid_v3" in score_map and "internal_real_v1_nocontext" in score_map:
        feat["meta::mid3_realv1_gap"] = float(score_map["internal_mid_v3"] - score_map["internal_real_v1_nocontext"])
    return feat


def main() -> None:
    args = parse_args()
    manifest = json.loads(Path(args.manifest_path).read_text(encoding="utf-8"))
    df = read_table(Path(args.input_path)).reset_index(drop=True)

    model, tokenizer = load_model_and_tokenizer(model_id=args.model_id, cache_dir=Path(args.cache_dir))
    cat_models = _load_catboost_models(manifest)
    qa_models = _load_qa_models(manifest)
    meta_model = _load_meta_model(manifest)

    out_rows = []
    for idx, row in enumerate(df.itertuples(index=False), start=1):
        row_s = pd.Series(row._asdict())
        feats = extract_features_one(
            row_s,
            model=model,
            tokenizer=tokenizer,
            include_internal_features=True,
            max_length=args.max_length,
        )

        score_map: dict[str, float] = {}
        for name, spec in cat_models.items():
            X = _prepare_cat_features(feats, spec["feature_columns"])
            score_map[name] = float(spec["model"].predict_proba(X)[:, 1][0])

        row_df = pd.DataFrame([row_s.to_dict()])
        for name, spec in qa_models.items():
            X, _ = spec["transform"](row_df, spec["vectorizers"])
            score_map[name] = float(spec["model"].predict_proba(X)[:, 1][0])

        if meta_model is not None:
            meta_feats = _build_meta_features(score_map)
            X_meta = pd.DataFrame([meta_feats])
            for col in meta_model["feature_columns"]:
                if col not in X_meta.columns:
                    X_meta[col] = 0.0
            X_meta = X_meta[meta_model["feature_columns"]]
            final_score = float(meta_model["model"].predict_proba(X_meta)[:, 1][0])
        else:
            final_score = _blend_scores(score_map, manifest["solid_batch_independent_blend"])
        out = row_s.to_dict()
        out["hallucination_prob"] = final_score
        if args.include_component_scores:
            for name, score in score_map.items():
                out[f"score::{name}"] = score
        out_rows.append(out)

        if idx % 50 == 0:
            print(f"predicted {idx}/{len(df)}")

    out_df = pd.DataFrame(out_rows)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() == ".parquet":
        out_df.to_parquet(output_path, index=False)
    elif output_path.suffix.lower() == ".csv":
        out_df.to_csv(output_path, index=False)
    else:
        raise ValueError(f"Unsupported output format: {output_path}")

    print(f"saved predictions: {output_path}")


if __name__ == "__main__":
    main()
