#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from qa_consistency_verifier import transform_text_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict with no-retrieval QA consistency verifier")
    parser.add_argument("--input-path", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("--model-path", type=str, default="detector_artifacts_qa_consistency_v1/qa_consistency_verifier.joblib")
    return parser.parse_args()


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported table format: {path}")


def main() -> None:
    args = parse_args()
    df = read_table(Path(args.input_path)).reset_index(drop=True)
    bundle = joblib.load(args.model_path)
    clf = bundle["model"]
    vectorizers = bundle["vectorizers"]

    X, _ = transform_text_features(df, vectorizers)
    pred = clf.predict_proba(X)[:, 1]

    out = df.copy()
    out["hallucination_prob"] = pred
    output_path = Path(args.output_path)
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
