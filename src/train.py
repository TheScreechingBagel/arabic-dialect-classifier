import argparse
import json
from pathlib import Path

import joblib
import mlflow
import numpy as np
import optuna
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.pipeline import Pipeline

from src.config import Config, load_config


def build_pipeline(params: dict) -> Pipeline:
    vectorizer = TfidfVectorizer(
        analyzer=params["analyzer"],
        ngram_range=(params["ngram_min"], params["ngram_max"]),
        max_features=params["max_features"],
        min_df=params["min_df"],
    )

    model = LogisticRegression(
        C=params["C"],
        class_weight=params["class_weight"],
        max_iter=params["max_iter"],
        solver="lbfgs",
        random_state=params["random_state"],
    )

    return Pipeline(
        [
            ("tfidf", vectorizer),
            ("clf", model),
        ]
    )


def evaluate(model: Pipeline, X, y) -> dict:
    preds = model.predict(X)
    return {
        "accuracy": accuracy_score(y, preds),
        "f1_macro": f1_score(y, preds, average="macro"),
        "f1_weighted": f1_score(y, preds, average="weighted"),
    }


def get_top_features(model: Pipeline, class_names, top_n: int = 20) -> dict:
    vectorizer = model.named_steps["tfidf"]
    clf = model.named_steps["clf"]

    feature_names = np.array(vectorizer.get_feature_names_out())
    result = {}

    for class_index, class_name in enumerate(class_names):
        coefs = clf.coef_[class_index]
        top_indices = np.argsort(coefs)[-top_n:][::-1]
        result[class_name] = [
            {
                "feature": str(feature_names[i]),
                "coefficient": float(coefs[i]),
            }
            for i in top_indices
        ]

    return result


def objective(trial, cfg: Config, train_df, val_df):
    params = {
        "analyzer": "char",
        "ngram_min": trial.suggest_int("ngram_min", 2, 3),
        "ngram_max": trial.suggest_int("ngram_max", 4, 6),
        "max_features": trial.suggest_categorical(
            "max_features", [20000, 50000, 80000]
        ),
        "min_df": trial.suggest_int("min_df", 1, 3),
        "C": trial.suggest_float("C", 0.1, 10.0, log=True),
        "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"]),
        "max_iter": cfg.model.max_iter,
        "random_state": cfg.project.random_state,
    }

    if params["ngram_max"] < params["ngram_min"]:
        raise optuna.TrialPruned()

    model = build_pipeline(params)
    model.fit(train_df["text"], train_df["label"])

    metrics = evaluate(model, val_df["text"], val_df["label"])
    return metrics["f1_macro"]


def main(config_path: str):
    cfg = load_config(config_path)

    train_df = pd.read_csv(cfg.data.train_path)
    val_df = pd.read_csv(cfg.data.val_path)
    test_df = pd.read_csv(cfg.data.test_path)

    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)

    base_params = {
        "analyzer": cfg.model.analyzer,
        "ngram_min": cfg.model.ngram_min,
        "ngram_max": cfg.model.ngram_max,
        "max_features": cfg.model.max_features,
        "min_df": cfg.model.min_df,
        "C": cfg.model.C,
        "class_weight": cfg.model.class_weight,
        "max_iter": cfg.model.max_iter,
        "random_state": cfg.project.random_state,
    }

    with mlflow.start_run(run_name="baseline_logreg_char_tfidf"):
        baseline_model = build_pipeline(base_params)
        baseline_model.fit(train_df["text"], train_df["label"])

        val_metrics = evaluate(baseline_model, val_df["text"], val_df["label"])

        mlflow.log_params(base_params)
        mlflow.log_metrics({f"val_{k}": v for k, v in val_metrics.items()})

        print("Baseline validation metrics:")
        print(val_metrics)

    best_params = base_params.copy()

    if cfg.training.run_optuna:
        study = optuna.create_study(direction="maximize")
        study.optimize(
            lambda trial: objective(trial, cfg, train_df, val_df),
            n_trials=cfg.training.optuna_trials,
        )

        best_params.update(study.best_params)

        print("\nBest Optuna params:")
        print(best_params)
        print(f"Best validation F1 macro: {study.best_value:.4f}")

    # Final training: train on train + validation, evaluate once on test.
    final_train_df = pd.concat([train_df, val_df], ignore_index=True)

    with mlflow.start_run(run_name="final_logreg_char_tfidf"):
        final_model = build_pipeline(best_params)
        final_model.fit(final_train_df["text"], final_train_df["label"])

        test_metrics = evaluate(final_model, test_df["text"], test_df["label"])
        test_preds = final_model.predict(test_df["text"])

        output_path = cfg.model.output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(final_model, output_path)

        reports_dir = Path("models/reports")
        reports_dir.mkdir(parents=True, exist_ok=True)

        class_names = list(final_model.named_steps["clf"].classes_)

        report = classification_report(
            test_df["label"],
            test_preds,
            output_dict=True,
            zero_division=0,
        )

        cm = confusion_matrix(test_df["label"], test_preds, labels=class_names)

        top_features = get_top_features(final_model, class_names, top_n=20)

        with open(
            reports_dir / "classification_report.json", "w", encoding="utf-8"
        ) as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        with open(reports_dir / "confusion_matrix.json", "w", encoding="utf-8") as f:
            json.dump(
                {"labels": class_names, "matrix": cm.tolist()},
                f,
                ensure_ascii=False,
                indent=2,
            )

        with open(reports_dir / "top_features.json", "w", encoding="utf-8") as f:
            json.dump(top_features, f, ensure_ascii=False, indent=2)

        mlflow.log_params(best_params)
        mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})
        mlflow.log_artifact(str(output_path))
        mlflow.log_artifacts(str(reports_dir))

        print("\nFinal test metrics:")
        print(test_metrics)

        print("\nClassification report:")
        print(classification_report(test_df["label"], test_preds, zero_division=0))

        print(f"\nSaved model to: {output_path}")
        print(f"Saved reports to: {reports_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    main(args.config)
