import argparse
import re
from pathlib import Path

import pandas as pd
import yaml
from datasets import load_dataset

ARABIC_DIACRITICS = re.compile(r"[\u0617-\u061A\u064B-\u0652]")


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def normalize_arabic_text(text: str) -> str:
    """
    Light normalization only.
    We do NOT over-clean dialect text because spelling variation is useful for classification.
    """
    if not isinstance(text, str):
        return ""

    text = text.strip()
    text = ARABIC_DIACRITICS.sub("", text)

    # Normalize common Arabic letter variants.
    text = re.sub("[إأآا]", "ا", text)
    text = re.sub("ى", "ي", text)
    text = re.sub("ؤ", "و", text)
    text = re.sub("ئ", "ي", text)
    text = re.sub("ة", "ه", text)

    # Remove URLs and usernames, keep hashtags text.
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = text.replace("#", "")

    # Normalize whitespace.
    text = re.sub(r"\s+", " ", text).strip()
    return text


def dataset_split_to_df(split, text_col: str, label_col: str) -> pd.DataFrame:
    df = pd.DataFrame(split)
    df = df[[text_col, label_col]].copy()
    df.columns = ["text", "label"]

    df["text"] = df["text"].astype(str).map(normalize_arabic_text)
    df["label"] = df["label"].astype(str)

    df = df.dropna()
    df = df[df["text"].str.len() > 0]
    df = df.drop_duplicates(subset=["text", "label"]).reset_index(drop=True)
    return df


def main(config_path: str):
    cfg = load_config(config_path)

    dataset_name = cfg["data"]["hf_dataset_name"]
    dataset_config = cfg["data"]["hf_config"]
    text_col = cfg["data"]["text_column"]
    label_col = cfg["data"]["label_column"]

    print(f"Loading dataset: {dataset_name}, config={dataset_config}")
    ds = load_dataset(dataset_name, dataset_config)

    train_df = dataset_split_to_df(ds["train"], text_col, label_col)

    # The dataset may call validation either "validation" or "dev".
    if "validation" in ds:
        val_split_name = "validation"
    elif "dev" in ds:
        val_split_name = "dev"
    else:
        raise ValueError(
            f"No validation/dev split found. Available splits: {list(ds.keys())}"
        )

    val_df = dataset_split_to_df(ds[val_split_name], text_col, label_col)
    test_df = dataset_split_to_df(ds["test"], text_col, label_col)

    raw_path = Path(cfg["data"]["raw_path"])
    train_path = Path(cfg["data"]["train_path"])
    val_path = Path(cfg["data"]["val_path"])
    test_path = Path(cfg["data"]["test_path"])

    raw_path.parent.mkdir(parents=True, exist_ok=True)
    train_path.parent.mkdir(parents=True, exist_ok=True)

    full_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    full_df.to_csv(raw_path, index=False, encoding="utf-8")
    train_df.to_csv(train_path, index=False, encoding="utf-8")
    val_df.to_csv(val_path, index=False, encoding="utf-8")
    test_df.to_csv(test_path, index=False, encoding="utf-8")

    print("Saved:")
    print(f"  raw:   {raw_path}   {full_df.shape}")
    print(f"  train: {train_path} {train_df.shape}")
    print(f"  val:   {val_path}   {val_df.shape}")
    print(f"  test:  {test_path}  {test_df.shape}")

    print("\nLabel distribution:")
    print(full_df["label"].value_counts())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    main(args.config)
