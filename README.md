# LLM Hallucination Detector

Ensemble approach to hallucination detection on Russian extractive QA: a QA-consistency verifier plus a feature-based hallucination scorer over hidden-state and decoding signals, combined with rank-average and meta-model ensembling. Originally built for the sberchallenge knowledge benchmark.

## Architecture

The pipeline runs three independent scorers over each `(question, context, model_answer)` triple, then blends their probabilities into a single hallucination score.

- **Hallucination scorer** (`hallucination_detector.py`, CatBoost) — extracts decoding-time signals from a generator LLM forward pass: token log-probs, attention dispersion, hidden-state norms, span-level NLL, length and surface features. Trained as a binary classifier over labeled positives/negatives.
- **QA-consistency verifier** (`qa_consistency_verifier{,_v2,_v3}.py`, sklearn + sparse text features) — pure post-hoc detector that looks at question shape (yes/no, numeric, year), answer shape, lexical overlap, gold-answer F1, and a TF-IDF model over question/answer text. No model forward pass required.
- **Factual-positive detector** — same backbone as the QA verifier, trained on a separate slice of factual positives generated via `generate_factual_positives.py` and `generate_real_positives.py`.

Final score is produced by `predict_solid_hallucination.py`, which loads a manifest describing the component models and either rank-averages their scores (`rankavg_ensemble.py`) or feeds them into a CatBoost meta-model with cross-component features (mean, std, min/max, pairwise gaps).

## Stack

Python 3.11. PyTorch, transformers (HF), CatBoost, scikit-learn, scipy.sparse, joblib, pandas, datasets, tqdm. vLLM optional for synthetic data generation.

## Reported metric

On the sberchallenge public eval, the honest, batch-independent `solid_v1` configuration (one allowed model forward pass, no retrieval, no external knowledge) reports:

> **PR-AUC = 0.7204** on the public knowledge benchmark eval (`solid_v1`, file `results/knowledge_bench_public_eval_solid_v1_metrics.json` in the original submission bundle).

Stronger numbers existed during research but were public-tuned or offline-ensemble based and are not the honest production line. See `SUBMISSION_NOTES.md`.

## Quickstart

The pipeline expects QA records with the schema documented in [`data/README.md`](data/README.md): `question`, `context`, `gold_answers`, `model_answer`. Adapt to your own dataset by writing those four columns into a parquet file.

```bash
# 1. generate model answers (if you don't have them yet)
python scripts/generate_answers_from_sberchallenge.py \
    --datasets-dir datasets --outputs-dir outputs

# 2. label them by gold-answer overlap
python scripts/prepare_hallucination_labeling.py \
    --inputs-dir outputs --output-dir artifacts

# 3. train the per-component detectors
python scripts/train_hallucination_detector.py \
    --labels-path artifacts/labeling_dataset.parquet \
    --artifacts-dir detector_artifacts
python scripts/train_qa_consistency_verifier_v3.py \
    --train-paths artifacts_poc/internal_real_v1.parquet \
    --artifacts-dir detector_artifacts_qa_consistency_v3

# 4. score with the ensemble
python scripts/predict_solid_hallucination.py \
    --input-path my_eval.parquet \
    --output-path my_predictions.parquet \
    --manifest-path solidification_clean_manifest_v1.json
```

The base generator model (`ai-sage/GigaChat3-10B-A1.8B-bf16`) is required for components that need a forward pass. See `MODEL_INSTALL.txt`.

## Why this matters

Hallucination detection without retrieval, without external knowledge, and within a fixed compute budget per answer is the realistic production setting. This codebase is the honest line, and the schema is reusable for any extractive QA dataset.

## Data

Training data is **not** included. The original training corpus derives from the sberchallenge knowledge benchmark and its license does not allow redistribution. The pipeline is reusable on any QA-pair dataset following the [documented schema](data/README.md).

## License

MIT, see `LICENSE`.
