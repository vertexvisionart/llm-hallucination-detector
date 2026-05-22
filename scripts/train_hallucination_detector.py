#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import average_precision_score

from hallucination_detector import MODEL_ID, extract_features_one, load_model_and_tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels-path", type=str, default="artifacts/gigachat_sberquad_labeling_dataset.parquet")
    parser.add_argument("--artifacts-dir", type=str, default="detector_artifacts")
    parser.add_argument("--cache-dir", type=str, default="hf_cache")
    parser.add_argument("--model-id", type=str, default=MODEL_ID)
    parser.add_argument("--feature-cache-path", type=str, default=None)
    parser.add_argument("--overwrite-feature-cache", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--train-split", type=str, default="train")
    parser.add_argument("--valid-split", type=str, default="validation")
    parser.add_argument("--negative-f1-threshold", type=float, default=0.9)
    parser.add_argument("--include-internal-features", action="store_true")
    parser.add_argument("--exclude-features", type=str, default="")
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--cache-every", type=int, default=0)
    parser.add_argument("--resume-feature-cache", action="store_true")
    parser.add_argument("--iterations", type=int, default=700)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--random-seed", type=int, default=42)
    return parser.parse_args()


def build_labels(df: pd.DataFrame, negative_f1_threshold: float) -> pd.DataFrame:
    out = df.copy()
    manual = out.get("is_hallucination_manual")
    manual_mask = manual.notna() if manual is not None else pd.Series(False, index=out.index)

    if "weak_label" in out.columns:
        weak_label_mask = out["weak_label"].notna()
        selected = manual_mask | weak_label_mask
        out = out.loc[selected].copy()
        out["label"] = out["weak_label"].fillna(0).astype(int)

        if manual is not None:
            manual_str = manual.astype(str).str.lower()
            true_mask = manual_mask & manual_str.isin(["true", "1"])
            false_mask = manual_mask & manual_str.isin(["false", "0"])
            out.loc[true_mask.reindex(out.index, fill_value=False), "label"] = 1
            out.loc[false_mask.reindex(out.index, fill_value=False), "label"] = 0
        return out

    weak_positive = out["is_hallucination_heuristic"].fillna(False).astype(bool)
    weak_negative = (
        out["exact_match_any"].fillna(False).astype(bool)
        | out["substring_match_any"].fillna(False).astype(bool)
        | (out["token_f1_max"].fillna(0.0) >= negative_f1_threshold)
    )

    selected = manual_mask | weak_positive | weak_negative
    out = out.loc[selected].copy()
    out["label"] = 0
    out.loc[weak_positive.reindex(out.index, fill_value=False), "label"] = 1

    if manual is not None:
        manual_str = manual.astype(str).str.lower()
        true_mask = manual_mask & manual_str.isin(["true", "1"])
        false_mask = manual_mask & manual_str.isin(["false", "0"])
        out.loc[true_mask.reindex(out.index, fill_value=False), "label"] = 1
        out.loc[false_mask.reindex(out.index, fill_value=False), "label"] = 0

    return out


def compute_features(
    df: pd.DataFrame,
    include_internal_features: bool,
    cache_dir: Path,
    model_id: str,
    feature_cache_path: Path | None,
    overwrite_feature_cache: bool,
    max_length: int,
    progress_every: int,
    cache_every: int,
    resume_feature_cache: bool,
) -> pd.DataFrame:
    cached_df = None
    cached_ids: set[int] = set()
    if feature_cache_path and feature_cache_path.exists() and not overwrite_feature_cache:
        if resume_feature_cache:
            cached_df = pd.read_parquet(feature_cache_path)
            cached_ids = set(cached_df["_row_id"].astype(int).tolist()) if "_row_id" in cached_df.columns else set()
            print(f"Resuming feature cache: {feature_cache_path} ({len(cached_ids)} ready rows)")
        else:
            print(f"Loading feature cache: {feature_cache_path}")
            return pd.read_parquet(feature_cache_path)

    pending_records = [record for record in df.to_dict(orient="records") if int(record["_row_id"]) not in cached_ids]

    model = None
    tokenizer = None
    if include_internal_features and pending_records:
        model, tokenizer = load_model_and_tokenizer(model_id=model_id, cache_dir=cache_dir)
        print(f"Loaded model for feature extraction: {model_id}")

    rows = [] if cached_df is None else cached_df.to_dict(orient="records")
    start = time.perf_counter()
    completed = len(rows)
    total = len(df)
    for local_idx, record in enumerate(pending_records, start=1):
        row_s = pd.Series(record)
        feats = extract_features_one(
            row_s,
            model=model,
            tokenizer=tokenizer,
            include_internal_features=include_internal_features,
            max_length=max_length,
        )
        feats["_row_id"] = int(record["_row_id"])
        rows.append(feats)
        completed += 1

        if progress_every > 0 and completed % progress_every == 0:
            elapsed = time.perf_counter() - start
            print(f"features: {completed}/{total} rows, elapsed={elapsed:.1f}s")

        if feature_cache_path and cache_every > 0 and completed % cache_every == 0:
            tmp_df = pd.DataFrame(rows)
            feature_cache_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_df.to_parquet(feature_cache_path, index=False)
            print(f"Checkpointed feature cache: {feature_cache_path} ({completed}/{total})")

    feat_df = pd.DataFrame(rows)
    if feature_cache_path:
        feature_cache_path.parent.mkdir(parents=True, exist_ok=True)
        feat_df.to_parquet(feature_cache_path, index=False)
        print(f"Saved feature cache: {feature_cache_path}")
    return feat_df


def main() -> None:
    args = parse_args()
    labels_path = Path(args.labels_path)
    artifacts_dir = Path(args.artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir)

    df = pd.read_parquet(labels_path).reset_index(drop=True)
    df = build_labels(df, negative_f1_threshold=args.negative_f1_threshold).reset_index(drop=True)
    df["_row_id"] = df.index

    if args.limit is not None:
        df = df.groupby("split_name", group_keys=False).head(args.limit).reset_index(drop=True)
        df["_row_id"] = df.index

    if args.feature_cache_path:
        feature_cache_path = Path(args.feature_cache_path)
    else:
        suffix = "internal" if args.include_internal_features else "cheap"
        feature_cache_path = artifacts_dir / f"features_{suffix}.parquet"

    print(f"Rows selected for training: {len(df)}")
    print(df["label"].value_counts(dropna=False).to_string())
    print(df["split_name"].value_counts(dropna=False).to_string())

    feat_df = compute_features(
        df=df,
        include_internal_features=args.include_internal_features,
        cache_dir=cache_dir,
        model_id=args.model_id,
        feature_cache_path=feature_cache_path,
        overwrite_feature_cache=args.overwrite_feature_cache,
        max_length=args.max_length,
        progress_every=args.progress_every,
        cache_every=args.cache_every,
        resume_feature_cache=args.resume_feature_cache,
    )

    merged = df.merge(feat_df, on="_row_id", how="inner")
    excluded_features = [name.strip() for name in args.exclude_features.split(",") if name.strip()]
    excluded_feature_set = set(excluded_features)
    feature_columns = [
        col for col in feat_df.columns
        if col != "_row_id" and col not in excluded_feature_set
    ]
    missing_excluded = [name for name in excluded_features if name not in feat_df.columns]
    if missing_excluded:
        print(f"warning: excluded features not found in feature cache: {missing_excluded}")
    print(f"Using {len(feature_columns)} feature columns")
    if excluded_features:
        print(f"Excluded {len(excluded_features)} features: {excluded_features}")

    train_df = merged[merged["split_name"] == args.train_split].copy()
    valid_df = merged[merged["split_name"] == args.valid_split].copy()
    if train_df.empty or valid_df.empty:
        raise RuntimeError("Train/validation split empty after filtering. Проверь split_name и пороги weak labels.")

    X_train = train_df[feature_columns]
    y_train = train_df["label"].astype(int)
    X_valid = valid_df[feature_columns]
    y_valid = valid_df["label"].astype(int)

    clf = CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="PRAUC",
        iterations=args.iterations,
        learning_rate=args.learning_rate,
        depth=args.depth,
        l2_leaf_reg=5.0,
        random_seed=args.random_seed,
        verbose=100,
        allow_writing_files=False,
    )

    clf.fit(Pool(X_train, y_train), eval_set=Pool(X_valid, y_valid), use_best_model=True)

    valid_pred = clf.predict_proba(X_valid)[:, 1]
    pr_auc = average_precision_score(y_valid, valid_pred)
    print(f"validation PR-AUC: {pr_auc:.6f}")

    model_path = artifacts_dir / "catboost_hallucination_detector.cbm"
    feature_path = artifacts_dir / "feature_columns.json"
    metrics_path = artifacts_dir / "metrics.json"
    fi_path = artifacts_dir / "feature_importance.csv"
    valid_pred_path = artifacts_dir / "validation_predictions.parquet"

    clf.save_model(str(model_path))
    feature_path.write_text(json.dumps(feature_columns, ensure_ascii=False, indent=2), encoding="utf-8")
    metrics = {
        "validation_pr_auc": float(pr_auc),
        "train_rows": int(len(train_df)),
        "valid_rows": int(len(valid_df)),
        "feature_count": int(len(feature_columns)),
        "include_internal_features": bool(args.include_internal_features),
        "excluded_features": excluded_features,
        "negative_f1_threshold": float(args.negative_f1_threshold),
    }
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    fi = pd.DataFrame({
        "feature": feature_columns,
        "importance": clf.get_feature_importance(type="FeatureImportance"),
    }).sort_values("importance", ascending=False)
    fi.to_csv(fi_path, index=False)

    valid_out = valid_df[["id", "split_name", "question", "model_answer", "label"]].copy()
    valid_out["hallucination_prob"] = valid_pred
    valid_out.to_parquet(valid_pred_path, index=False)

    print(f"saved model: {model_path}")
    print(f"saved metrics: {metrics_path}")
    print(f"saved feature importances: {fi_path}")
    print(f"saved validation predictions: {valid_pred_path}")


if __name__ == "__main__":
    main()
