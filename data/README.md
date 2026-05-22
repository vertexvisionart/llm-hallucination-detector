# Data schema

Training data is **not** included in this repository. The original corpus derives from the sberchallenge knowledge benchmark, and its terms do not allow redistribution.

The pipeline can be re-trained on any extractive QA dataset that follows the schema below. Save records as a `.parquet` file (recommended) or `.csv`.

## Required columns

| column         | type            | description                                                                 |
|----------------|-----------------|-----------------------------------------------------------------------------|
| `id`           | str             | Stable identifier per QA pair.                                              |
| `question`     | str             | The natural-language question.                                              |
| `context`      | str             | Passage the answer is grounded in (extractive QA setting).                  |
| `gold_answers` | list[str] / str | One or more reference answer strings. Stringified Python lists are parsed.  |
| `model_answer` | str             | The candidate answer to score (output of the generator under test).         |

## Optional columns

| column         | type            | description                                                                 |
|----------------|-----------------|-----------------------------------------------------------------------------|
| `title`        | str             | Title or topic of the source passage. Used in prompt construction.          |
| `is_hallucination` | int (0/1)   | Ground-truth label, required for training; not required for inference.      |
| `split`        | str             | `train` / `validation` / `test`.                                            |

## Label semantics

`is_hallucination = 1` means the model answer does not match any gold answer beyond surface noise. `is_hallucination = 0` means the model answer is judged correct (high token-F1 with at least one gold, allowing inflectional and single-token paraphrase variation). The exact rule is implemented in `scripts/prepare_hallucination_labeling.py` and produces the labels automatically from gold answers, so you only need to supply `gold_answers` and `model_answer` for training-data construction.

## Example record

```json
{
  "id": "q-000123",
  "title": "Ленинградская область",
  "question": "В каком году была образована Ленинградская область?",
  "context": "Ленинградская область была образована 1 августа 1927 года ...",
  "gold_answers": ["1927", "в 1927 году"],
  "model_answer": "в 1927"
}
```

## Generating labels from raw QA outputs

```bash
python scripts/prepare_hallucination_labeling.py \
    --inputs-dir outputs \
    --output-dir artifacts
```

This emits a parquet file with `is_hallucination`, `token_f1_max`, `content_token_f1_max`, `char_similarity_max`, and the per-row reason for the label, ready to feed into the detector trainers.
