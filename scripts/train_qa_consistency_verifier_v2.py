#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

from qa_consistency_verifier_v2 import build_feature_name_list, fit_text_vectorizers, transform_text_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train no-retrieval QA consistency verifier v2")
    parser.add_argument(
        "--labels-paths",
        type=str,
        default="artifacts_poc/internal_real_v1.parquet,artifacts_poc/internal_factual_v1.parquet",
    )
    parser.add_argument("--artifacts-dir", type=str, default="detector_artifacts_qa_consistency_v2")
    parser.add_argument("--label-column", type=str, default="is_hallucination")
    parser.add_argument("--train-split", type=str, default="train")
    parser.add_argument("--valid-split", type=str, default="validation")
    parser.add_argument("--c", type=float, default=3.0)
    parser.add_argument("--max-iter", type=int, default=2500)
    return parser.parse_args()


def read_concat(paths: list[str]) -> pd.DataFrame:
    frames = []
    for raw_path in paths:
        path = Path(raw_path.strip())
        if not path.exists():
            raise FileNotFoundError(path)
        frames.append(pd.read_parquet(path) if path.suffix.lower() == ".parquet" else pd.read_csv(path))
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    args = parse_args()
    artifacts_dir = Path(args.artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    df = read_concat([p for p in args.labels_paths.split(",") if p.strip()]).reset_index(drop=True)
    if args.label_column not in df.columns:
        raise RuntimeError(f"Missing label column: {args.label_column}")
    if "split_name" not in df.columns:
        raise RuntimeError("Missing split_name")

    df = df.loc[df[args.label_column].notna()].copy()
    df["label"] = (pd.to_numeric(df[args.label_column], errors="coerce").fillna(0.0) > 0).astype(int)

    train_df = df.loc[df["split_name"] == args.train_split].reset_index(drop=True)
    valid_df = df.loc[df["split_name"] == args.valid_split].reset_index(drop=True)
    if train_df.empty or valid_df.empty:
        raise RuntimeError("Empty train/validation split")

    vectorizers = fit_text_vectorizers(train_df)
    X_train, dense_train = transform_text_features(train_df, vectorizers)
    X_valid, dense_valid = transform_text_features(valid_df, vectorizers)
    y_train = train_df["label"].astype(int).to_numpy()
    y_valid = valid_df["label"].astype(int).to_numpy()

    clf = LogisticRegression(
        C=args.c,
        max_iter=args.max_iter,
        solver="saga",
        n_jobs=-1,
        random_state=42,
        class_weight="balanced",
    )
    clf.fit(X_train, y_train)

    valid_pred = clf.predict_proba(X_valid)[:, 1]
    pr_auc = average_precision_score(y_valid, valid_pred)
    print(f"validation PR-AUC: {pr_auc:.6f}")

    model_bundle = {
        "model": clf,
        "vectorizers": vectorizers,
    }
    model_path = artifacts_dir / "qa_consistency_verifier.joblib"
    feature_columns_path = artifacts_dir / "feature_columns.json"
    metrics_path = artifacts_dir / "metrics.json"
    valid_pred_path = artifacts_dir / "validation_predictions.parquet"
    dense_valid_path = artifacts_dir / "validation_dense_features.parquet"

    joblib.dump(model_bundle, model_path)
    feature_names = build_feature_name_list(vectorizers, dense_train.columns.tolist())
    feature_columns_path.write_text(json.dumps(feature_names, ensure_ascii=False, indent=2), encoding="utf-8")
    metrics_path.write_text(json.dumps({
        "validation_pr_auc": float(pr_auc),
        "train_rows": int(len(train_df)),
        "valid_rows": int(len(valid_df)),
        "c": float(args.c),
        "max_iter": int(args.max_iter),
        "feature_count": int(len(feature_names)),
        "labels_paths": [p for p in args.labels_paths.split(",") if p.strip()],
        "uses_gold_features": False,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    valid_out = valid_df.copy()
    if "id" in valid_out.columns:
        valid_out["id"] = valid_out["id"].astype(str)
    valid_out["hallucination_prob"] = valid_pred
    valid_out.to_parquet(valid_pred_path, index=False)
    dense_valid.to_parquet(dense_valid_path, index=False)

    print(f"saved model: {model_path}")
    print(f"saved metrics: {metrics_path}")
    print(f"saved validation predictions: {valid_pred_path}")


if __name__ == "__main__":
    main()
