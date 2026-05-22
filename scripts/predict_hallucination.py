#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from catboost import CatBoostClassifier

from hallucination_detector import MODEL_ID, extract_features_one, load_model_and_tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("--model-path", type=str, default="detector_artifacts/catboost_hallucination_detector.cbm")
    parser.add_argument("--feature-columns-path", type=str, default="detector_artifacts/feature_columns.json")
    parser.add_argument("--cache-dir", type=str, default="hf_cache")
    parser.add_argument("--model-id", type=str, default=MODEL_ID)
    parser.add_argument("--include-internal-features", action="store_true")
    parser.add_argument("--max-length", type=int, default=4096)
    return parser.parse_args()


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported input format: {path}")


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_path)
    output_path = Path(args.output_path)

    df = read_table(input_path).reset_index(drop=True)

    model = None
    tokenizer = None
    if args.include_internal_features:
        model, tokenizer = load_model_and_tokenizer(model_id=args.model_id, cache_dir=Path(args.cache_dir))

    feature_rows = []
    for idx, row in enumerate(df.itertuples(index=False), start=1):
        row_s = pd.Series(row._asdict())
        feats = extract_features_one(
            row_s,
            model=model,
            tokenizer=tokenizer,
            include_internal_features=args.include_internal_features,
            max_length=args.max_length,
        )
        feature_rows.append(feats)
        if idx % 100 == 0:
            print(f"pred features: {idx}/{len(df)}")

    X = pd.DataFrame(feature_rows)
    feature_columns = json.loads(Path(args.feature_columns_path).read_text(encoding="utf-8"))
    missing = [col for col in feature_columns if col not in X.columns]
    for col in missing:
        X[col] = 0.0
    X = X[feature_columns]

    clf = CatBoostClassifier()
    clf.load_model(args.model_path)
    pred = clf.predict_proba(X)[:, 1]

    out = df.copy()
    out["hallucination_prob"] = pred
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() == ".parquet":
        out.to_parquet(output_path, index=False)
    elif output_path.suffix.lower() == ".csv":
        out.to_csv(output_path, index=False)
    else:
        raise ValueError(f"Unsupported output format: {output_path}")

    print(f"saved predictions: {output_path}")


if __name__ == "__main__":
    main()
