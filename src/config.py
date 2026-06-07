from pathlib import Path
from typing import ClassVar

import yaml
from pydantic import BaseModel, ConfigDict


class _Section(BaseModel):
    # extra="forbid" → unknown/typo'd keys fail at load; protected_namespaces=() → allow `model`
    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="forbid", protected_namespaces=()
    )


class DataConfig(_Section):
    hf_dataset_name: str
    hf_config: str
    text_column: str
    label_column: str
    raw_path: Path
    train_path: Path
    val_path: Path
    test_path: Path


class ModelConfig(_Section):
    output_path: Path
    analyzer: str
    ngram_min: int
    ngram_max: int
    max_features: int
    min_df: int
    C: float
    class_weight: str | None
    max_iter: int


class TrainingConfig(_Section):
    metric: str
    optuna_trials: int
    run_optuna: bool


class MLflowConfig(_Section):
    experiment_name: str
    tracking_uri: str


class ProjectConfig(_Section):
    name: str
    random_state: int


class Config(_Section):
    project: ProjectConfig
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    mlflow: MLflowConfig


def load_config(path: str | Path = "config.yaml") -> Config:
    with open(path, encoding="utf-8") as f:
        return Config.model_validate(yaml.safe_load(f))
