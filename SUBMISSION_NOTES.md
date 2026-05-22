# Honest Submission Bundle: `solid_v1`

This bundle is the recommended honest final submission package for the closed benchmark.

Chosen model:
- `solid_v1`

Why this bundle:
- batch-independent
- no retrieval
- no local Wikipedia / external DB / search-based evidence
- no Yura component
- aligned with the `500 ms per answer` framing from the task
- uses only one allowed GigaChat forward pass plus a lightweight post-forward detector stack

Reference public result for this honest line:
- `PR-AUC = 0.7203820165744488`
- file: `results/knowledge_bench_public_eval_solid_v1_metrics.json`

Important:
- stronger public numbers were obtained during research, but they were either public-tuned or offline-ensemble based
- this package is intended for the honest production-style line

## What Is Included

- full scripts used in the project
- dataset-building scripts
- bundled parquet/CSV data
- detector artifacts
- manifests
- logs
- environment snapshot

## What Is Not Included

- the base model weights for `ai-sage/GigaChat3-10B-A1.8B-bf16`

## Install The Model

Download:
- `ai-sage/GigaChat3-10B-A1.8B-bf16`

Place it here:
- `hf_cache/gigachat3_local/`

Expected files:
- `hf_cache/gigachat3_local/config.json`
- `hf_cache/gigachat3_local/tokenizer.json`

See also:
- `MODEL_INSTALL.txt`

## Run

From the bundle root:

```bash
./RUN_SOLID_V1.sh
```

This will:
1. run the honest batch-independent predictor
2. write predictions
3. evaluate the result

## Main Files

- `RUN_SOLID_V1.sh`
- `solidification_clean_manifest_bundle_v1.json`
- `scripts/predict_solid_hallucination.py`
- `results/knowledge_bench_public_eval_solid_v1_metrics.json`
- `PROJECT_STATUS_CURRENT.md`

## Notes

- `PROJECT_STATUS_CURRENT.md` contains the latest experiment history, including later public-tuned and honest follow-up work
- for the actual honest submission line, use `solid_v1`
