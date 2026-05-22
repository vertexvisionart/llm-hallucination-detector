#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, precision_score, recall_score, roc_auc_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rank-average ensemble for prediction files")
    parser.add_argument("--input-paths", nargs="+", required=True)
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("--output-metrics-path", type=str, default=None)
    parser.add_argument("--score-column", type=str, default="hallucination_prob")
    parser.add_argument("--label-column", type=str, default="is_hallucination")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--weights", type=str, default=None, help="Comma-separated weights, same length as input-paths")
    return parser.parse_args()


def to_binary(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(int)
    if pd.api.types.is_numeric_dtype(series):
        return (series.astype(float) > 0).astype(int)
    lowered = series.astype(str).str.lower()
    return lowered.isin(["true", "1", "yes"]).astype(int)


def parse_weights(raw: str | None, n: int) -> list[float]:
    if raw is None:
        return [1.0] * n
    weights = [float(part.strip()) for part in raw.split(",") if part.strip()]
    if len(weights) != n:
        raise RuntimeError(f"Expected {n} weights, got {len(weights)}")
    return weights


def main() -> None:
    args = parse_args()
    weights = parse_weights(args.weights, len(args.input_paths))

    frames = []
    for idx, path in enumerate(args.input_paths):
        df = pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)
        if args.score_column not in df.columns:
            raise RuntimeError(f"{path} missing score column {args.score_column}")
        ranked = df[args.score_column].rank(method="average", pct=True)
        local = df.copy()
        local[f"__rank_{idx}"] = ranked
        frames.append(local)

    base = frames[0].copy()
    rank_columns = ["__rank_0"]
    for idx, df in enumerate(frames[1:], start=1):
        rank_col = f"__rank_{idx}"
        rank_columns.append(rank_col)
        base[rank_col] = df[rank_col].values

    weight_sum = sum(weights)
    base["hallucination_prob"] = sum(base[col] * w for col, w in zip(rank_columns, weights)) / weight_sum
    out = base.drop(columns=rank_columns)

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() == ".parquet":
        out.to_parquet(output_path, index=False)
    else:
        out.to_csv(output_path, index=False)
    print(f"saved ensemble predictions: {output_path}")

    if args.label_column in out.columns:
        y_true = to_binary(out[args.label_column])
        y_score = pd.to_numeric(out["hallucination_prob"], errors="coerce").fillna(0.0)
        y_pred = (y_score >= args.threshold).astype(int)
        metrics = {
            "rows": int(len(out)),
            "blend_type": "rankavg_equal" if args.weights is None else "rankavg_weighted",
            "input_paths": args.input_paths,
            "weights": weights,
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
