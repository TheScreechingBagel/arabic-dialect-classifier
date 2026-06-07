import argparse
from collections import Counter

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score

from src.config import load_config

EGY_MARKERS = ["مش", "اوي", "عايز", "كده", "ده", "دي", "احنا", "مفيش"]
GLF_MARKERS = ["شلون", "وايد", "ابي", "ابغى", "وش", "ليش", "هلا", "ترى"]
LAV_MARKERS = ["شو", "كتير", "هيك", "هلق", "بدي", "ليش", "نحنا"]
NOR_MARKERS = ["برشا", "بزاف", "شنو", "واش", "ديال", "راهو", "نحب"]


def predict_rule(text: str, fallback_label: str) -> str:
    text = str(text)

    scores = {
        "EGY": sum(m in text for m in EGY_MARKERS),
        "GLF": sum(m in text for m in GLF_MARKERS),
        "LAV": sum(m in text for m in LAV_MARKERS),
        "NOR": sum(m in text for m in NOR_MARKERS),
        "MSA": 0,
    }

    best_label, best_score = max(scores.items(), key=lambda x: x[1])
    if best_score == 0:
        return fallback_label
    return best_label


def main(config_path: str):
    cfg = load_config(config_path)
    train_df = pd.read_csv(cfg.data.train_path)
    test_df = pd.read_csv(cfg.data.test_path)

    majority_label = Counter(train_df["label"]).most_common(1)[0][0]

    y_true = test_df["label"]
    y_pred = [predict_rule(text, majority_label) for text in test_df["text"]]

    print("Regex/rule baseline")
    print(f"Fallback majority label: {majority_label}")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(f"F1 macro: {f1_score(y_true, y_pred, average='macro'):.4f}")
    print(classification_report(y_true, y_pred))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    main(args.config)
