# Arabic Dialect Classification

End-to-end ML/MLOps project for Arabic dialect classification.

The project classifies short Arabic text snippets into one of five labels and serves the trained model through a local FastAPI application.

```text
raw dataset -> preprocessing -> baseline -> model training -> evaluation -> saved model -> FastAPI service -> feedback collection
```

## Problem statement

Arabic dialects share the Arabic script and much of their vocabulary, but they differ in spelling habits, informal writing conventions, particles, lexical markers, and regional expressions.

The goal is to classify an Arabic text snippet into one of five dialect classes:

| Label | Meaning |
|---|---|
| `EGY` | Egyptian Arabic |
| `GLF` | Gulf Arabic |
| `LAV` | Levantine Arabic |
| `NOR` | North African Arabic |
| `MSA` | Modern Standard Arabic |

A lightweight dialect classifier can be useful as a routing component before downstream NLP systems such as localization, search, customer support, moderation, analytics, or dialect-aware text processing.

This project prioritizes a reproducible end-to-end pipeline over model complexity.

## Dataset

Dataset: `drelhaj/Arabic-Dialects` from Hugging Face.

The project uses the `full_text` configuration:

```python
from datasets import load_dataset

ds = load_dataset("drelhaj/Arabic-Dialects", "full_text")
```

The original dataset columns are normalized into the project format:

| Original column | Project column | Meaning |
|---|---|---|
| `sentence` | `text` | Arabic text snippet |
| `dialect` | `label` | Dialect class |

The `freq_lists` configuration is not used as the main training dataset. It is useful for linguistic analysis, but the supervised classifier needs sentence-level text examples from `full_text`.

## Metric choice

The primary metric is **macro-F1**.

Macro-F1 is appropriate because this is a multi-class classification task and the classes are imbalanced. Accuracy alone can hide weak performance on smaller dialect classes. Macro-F1 gives equal weight to each dialect class.

Additional reported metrics:

- accuracy
- weighted F1
- per-class precision, recall, and F1
- confusion matrix

## Modeling approach

### Baseline

The baseline is a regex/rule-based classifier using manually selected dialect marker words. If no dialect marker is found, the baseline falls back to the majority class from the training data.

This baseline is intentionally simple. It gives a minimum comparison point and shows why a learned model is useful.

### Final model

The final served model is:

```text
TF-IDF character n-grams + Logistic Regression
```

It is implemented as a scikit-learn `Pipeline`:

```text
TfidfVectorizer(analyzer="char") -> LogisticRegression
```

Character n-grams are useful because dialectal Arabic is often informal, noisy, and inconsistently spelled. Character-level features can capture spelling patterns and short dialectal fragments without relying on perfect word tokenization.

### Hyperparameter tuning

Optuna is used to tune selected hyperparameters on the validation set:

- Logistic Regression regularization strength `C`
- character n-gram range
- `max_features`
- `min_df`
- `class_weight`

The optimization objective is validation macro-F1.

After tuning, the final model is trained on train plus validation data and evaluated once on the test set.

### Additional experiments

The notebooks include additional modeling experiments:

- regex/rule baseline
- Stanza / word-tokenization experiment
- hybrid ML + regex override layer
- threshold tuning for regex overrides
- combined character + word TF-IDF features
- optional LinearSVC comparison

The hybrid regex layer was tested but rejected because it did not improve macro-F1. This showed that manual dialect markers can help with interpretation, but they are too brittle for hard prediction overrides.

## Current results

The current final model achieved approximately:

| Metric | Value |
|---|---:|
| Accuracy | 0.76 |
| Macro-F1 | 0.71 |
| Weighted F1 | 0.76 |

The model performs especially well on Egyptian Arabic. The hardest classes to separate are Gulf, Levantine, and North African Arabic, which are often confused with each other in short text snippets.

## Repository structure

```text
arabic-dialect-classification/
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── schemas.py
│   └── static/
│       └── index.html
├── data/
│   ├── raw/
│   │   └── .gitkeep
│   ├── processed/
│   │   └── .gitkeep
│   └── feedback/
│       └── .gitkeep
├── models/
│   └── reports/
├── notebooks/
│   └── 01_eda_modeling.ipynb
├── src/
│   ├── __init__.py
│   ├── preprocess.py
│   ├── regex_baseline.py
│   ├── train.py
│   └── utils.py
├── .dvcignore
├── .gitignore
├── .python-version
├── config.yaml
├── dvc.lock
├── dvc.yaml
├── pyproject.toml
├── README.md
└── uv.lock
```

Generated artifacts are not committed directly to Git:

```text
data/raw/*.csv
data/processed/*.csv
data/feedback/*.csv
models/*.joblib
models/reports/
mlruns/
mlartifacts/
mlflow.db
```

These files can be regenerated locally. If DVC is configured, they can also be versioned through DVC.

## Setup from a fresh clone

Clone the repository:

```bash
git clone https://github.com/akdenizler/arabic-dialect-classifier.git
cd arabic-dialect-classifier
```

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Reproduce the pipeline locally

Download and preprocess the dataset:

```bash
python -m src.preprocess
```

Run the regex/rule baseline:

```bash
python -m src.regex_baseline
```

Train the final model:

```bash
python -m src.train
```

The trained model is saved to:

```text
models/arabic_dialect_model.joblib
```

Training reports are saved to:

```text
models/reports/
```

## MLflow experiment tracking

Training logs parameters, metrics, and artifacts to local MLflow.

This project uses SQLite as the local MLflow backend store:

```text
sqlite:///mlflow.db
```

Run training:

```bash
python -m src.train
```

Open the MLflow UI:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Then open:

```text
http://127.0.0.1:5000
```

Do not commit the following local MLflow files to Git:

```text
mlruns/
mlartifacts/
mlflow.db
```

## DVC artifact versioning

The course project expects generated artifacts to be versioned with DVC and stored in a Google Drive remote. This repository keeps generated artifacts out of Git.

Recommended DVC-tracked artifacts:

```text
data/raw/arabic_dialect_dataset.csv
data/processed/train.csv
data/processed/val.csv
data/processed/test.csv
models/arabic_dialect_model.joblib
models/reports/
```

If DVC has been configured, the pipeline can be reproduced with:

```bash
dvc repro
```

Artifacts can be pulled from the configured remote with:

```bash
dvc pull
```

If Google Drive OAuth blocks `dvc push`, the local DVC pipeline can still be used with `dvc repro`, and the generated artifacts can be shared manually through the course Google Drive folder.

## FastAPI serving

Start the API:

```bash
uvicorn app.main:app --reload
```

If port `8000` is already in use:

```bash
uvicorn app.main:app --reload --port 8001
```

Open the local web UI:

```text
http://127.0.0.1:8000/
```

Open Swagger UI:

```text
http://127.0.0.1:8000/docs
```

Swagger shows the Pydantic request and response schemas.

## Prediction endpoint

Endpoint:

```text
POST /predict
```

Example request:

```json
{
  "text": "انا عايز اعرف الاخبار النهارده"
}
```

Example response:

```json
{
  "predicted_dialect": "EGY",
  "probability": 0.82,
  "class_probabilities": {
    "EGY": 0.82,
    "GLF": 0.04,
    "LAV": 0.05,
    "MSA": 0.06,
    "NOR": 0.03
  }
}
```

Terminal test:

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "انا عايز اعرف الاخبار النهارده"}'
```

## Feedback endpoint

The application also includes a feedback collection endpoint.

Endpoint:

```text
POST /feedback
```

The model does not update itself live after every prediction. Instead, user corrections are saved as labeled feedback data and can be used in a later retraining cycle.

Example request:

```json
{
  "text": "شو بدك تعمل هلق",
  "predicted_dialect": "GLF",
  "correct_dialect": "LAV",
  "model_probability": 0.61,
  "class_probabilities": {
    "EGY": 0.05,
    "GLF": 0.61,
    "LAV": 0.24,
    "MSA": 0.04,
    "NOR": 0.06
  }
}
```

Corrections are saved to:

```text
data/feedback/labeled_feedback.csv
```

This is safer than live online learning because feedback can be reviewed, versioned, and evaluated before deployment.

## Notebook role

The notebooks are the exploratory and explanatory artifacts. They show the reasoning process behind the final model.

Expected notebook coverage:

1. Problem framing
2. Dataset loading
3. Class distribution
4. Text length analysis
5. Example texts per dialect
6. Preprocessing decisions
7. Regex/rule baseline
8. Stanza / word-tokenization experiment
9. TF-IDF character n-gram Logistic Regression
10. Optuna tuning summary
11. Final test metrics
12. Confusion matrix
13. Hybrid regex experiment
14. Combined character + word TF-IDF experiment
15. Feature importance from Logistic Regression coefficients
16. Error analysis
17. Production deployment and monitoring discussion

The notebooks are not the production pipeline. The reproducible code is in:

```text
src/preprocess.py
src/regex_baseline.py
src/train.py
app/main.py
```

## Feature importance

Feature importance is calculated from Logistic Regression coefficients.

For each class, the project extracts the character n-grams with the highest positive coefficients. These n-grams are the features that most strongly push the model toward that dialect.

This is not perfect linguistic proof, but it gives an interpretable view of what the model learned.

## Production use

In production, this model could run as a lightweight API service.

Possible flow:

```text
user text -> FastAPI service -> dialect prediction -> dialect-specific downstream pipeline
```

The predicted dialect and class probabilities can be used to route text to dialect-specific moderation, localization, support, search, or analytics systems.

Low-confidence predictions can be sent to fallback handling or human review.

## Monitoring and drift

Online metrics to monitor:

- request count
- API latency
- API error rate
- prediction distribution by dialect
- average model confidence
- percentage of low-confidence predictions
- human-reviewed accuracy or macro-F1 if labels become available later

Data drift signals:

- sudden change in dialect distribution
- input texts much longer or shorter than training examples
- new slang, spelling patterns, or code-switching
- rising low-confidence predictions
- lower human-reviewed accuracy

## Known limitations

This is a classical ML model, not a transformer model. It is fast, interpretable, and easy to deploy, but it may not capture deeper sentence-level semantics.

The model may struggle with:

- very short text
- mixed dialects
- Arabizi
- heavy spelling errors
- dialects not represented in the training labels
- ambiguous non-Egyptian examples, especially `GLF`, `LAV`, and `NOR`

## Main commands

```bash
pip install -r requirements.txt
python -m src.preprocess
python -m src.regex_baseline
python -m src.train
uvicorn app.main:app --reload
```

## Presentation summary

The project starts with a weak regex/rule baseline, then moves to a character n-gram TF-IDF Logistic Regression model. Optuna is used to tune the final model. Several improvement ideas were tested, including regex overrides and combined character plus word TF-IDF features. The final model was selected because it gave the best balance of macro-F1, simplicity, interpretability, and ease of deployment.

The API demo shows the model running locally through FastAPI with typed Pydantic schemas and a simple local frontend.
