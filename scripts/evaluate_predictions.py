#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, precision_score, recall_score, roc_auc_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate prediction file with hallucination_prob")
    parser.add_argument("--input-path", type=str, required=True)
    parser.add_argument("--output-metrics-path", type=str, default=None)
    parser.add_argument("--label-column", type=str, default="is_hallucination")
    parser.add_argument("--score-column", type=str, default="hallucination_prob")
    parser.add_argument("--threshold", type=float, default=0.5)
    return parser.parse_args()


def to_binary(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(int)
    if pd.api.types.is_numeric_dtype(series):
        return (series.astype(float) > 0).astype(int)
    lowered = series.astype(str).str.lower()
    return lowered.isin(["true", "1", "yes"]).astype(int)


def main() -> None:
    args = parse_args()
    df = pd.read_parquet(args.input_path) if args.input_path.endswith(".parquet") else pd.read_csv(args.input_path)
    if args.label_column not in df.columns:
        raise RuntimeError(f"Missing label column: {args.label_column}")
    if args.score_column not in df.columns:
        raise RuntimeError(f"Missing score column: {args.score_column}")

    y_true = to_binary(df[args.label_column])
    y_score = pd.to_numeric(df[args.score_column], errors="coerce").fillna(0.0)
    y_pred = (y_score >= args.threshold).astype(int)

    metrics = {
        "rows": int(len(df)),
        "label_column": args.label_column,
        "score_column": args.score_column,
        "threshold": float(args.threshold),
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "accuracy_at_threshold": float(accuracy_score(y_true, y_pred)),
        "precision_at_threshold": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall_at_threshold": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_at_threshold": float(f1_score(y_true, y_pred, zero_division=0)),
    }

    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    if args.output_metrics_path:
        Path(args.output_metrics_path).write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"saved metrics: {args.output_metrics_path}")


if __name__ == "__main__":
    main()
