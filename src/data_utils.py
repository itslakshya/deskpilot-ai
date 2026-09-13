"""
Shared data utilities: loading, cleaning, label encoding, and splitting.

Split strategy: stratified 70/15/15 train/val/test on `category` (the rarer,
harder-to-balance target), with a fixed random_state for reproducibility.
We split on already-deduplicated text (see generate_dataset.py) specifically
to avoid the same query_text appearing in both train and test, which would
silently inflate reported accuracy (train/test leakage).
"""
import json
import re
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "v1", "tickets.csv")
CATEGORY_CLASSES = None  # populated at runtime from schema for a stable label order
PRIORITY_CLASSES = ["P1-Critical", "P2-High", "P3-Medium", "P4-Low"]
SCHEMA_PATH = os.path.join(PROJECT_ROOT, "data", "schema", "ticket_schema.json")


def load_schema():
    with open(SCHEMA_PATH) as f:
        return json.load(f)


def get_category_classes():
    schema = load_schema()
    return schema["properties"]["category"]["enum"]


def clean_text(text: str) -> str:
    """Light, reversible cleaning - keep tech tokens (error codes, version numbers) intact."""
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9./\-\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    df["clean_text"] = df["query_text"].apply(clean_text)
    return df


def stratified_split(df, target_col="category", test_size=0.15, val_size=0.15, seed=42):
    """Returns (train_df, val_df, test_df), stratified on target_col."""
    train_val, test = train_test_split(
        df, test_size=test_size, stratify=df[target_col], random_state=seed
    )
    val_ratio = val_size / (1 - test_size)
    train, val = train_test_split(
        train_val, test_size=val_ratio, stratify=train_val[target_col], random_state=seed
    )
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


class LabelEncoderFixed:
    """Simple label<->int encoder with a fixed, explicit class order (unlike sklearn's
    LabelEncoder which sorts alphabetically and silently reorders if new classes appear -
    fixed order matters here because model output layer size/order must stay stable
    across retrains, see scripts/append_and_retrain.py)."""

    def __init__(self, classes):
        self.classes_ = list(classes)
        self._to_idx = {c: i for i, c in enumerate(self.classes_)}

    def transform(self, labels):
        return np.array([self._to_idx[l] for l in labels], dtype=np.int64)

    def inverse_transform(self, indices):
        return [self.classes_[i] for i in indices]

    @property
    def n_classes(self):
        return len(self.classes_)


if __name__ == "__main__":
    df = load_data()
    cat_classes = get_category_classes()
    print(f"Loaded {len(df)} rows, {len(cat_classes)} category classes")
    train, val, test = stratified_split(df)
    print(f"train={len(train)}  val={len(val)}  test={len(test)}")
    train.to_csv(os.path.join(PROJECT_ROOT, "data", "v1", "train.csv"), index=False)
    val.to_csv(os.path.join(PROJECT_ROOT, "data", "v1", "val.csv"), index=False)
    test.to_csv(os.path.join(PROJECT_ROOT, "data", "v1", "test.csv"), index=False)
    print("Saved train/val/test splits to data/v1/")
