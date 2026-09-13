"""
Head-to-head comparison of the three approaches, plus an analysis of what the
confidence threshold actually buys us (accuracy-by-confidence-bucket, and top-3
recall for the low-confidence subset) - this is the evidence behind the fallback
UX design, not just an assumption.
"""
import json

import joblib
import numpy as np
import pandas as pd

from data_utils import clean_text, get_category_classes, load_data, stratified_split

import os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")


def comparison_table():
    with open(f"{MODELS_DIR}/classical_results.json") as f:
        classical = json.load(f)
    with open(f"{MODELS_DIR}/dl_results.json") as f:
        dl = json.load(f)
    with open(f"{MODELS_DIR}/retrieval_results.json") as f:
        retrieval = json.load(f)

    rows = [
        {"approach": "Retrieval (TF-IDF centroid cosine-sim)", "task": "category",
         "test_accuracy": retrieval["test_accuracy"], "test_macro_f1": retrieval["test_macro_f1"],
         "params_or_notes": "0 trainable params - just corpus statistics"},
        {"approach": "Classical ML (tuned Logistic Regression)", "task": "category",
         "test_accuracy": classical["category_model"]["test_accuracy"],
         "test_macro_f1": classical["category_model"]["test_macro_f1"],
         "params_or_notes": "~54K coefficients, grid-searched C"},
        {"approach": "Deep Learning (multi-task NumPy net)", "task": "category",
         "test_accuracy": dl["test_category"]["accuracy"], "test_macro_f1": dl["test_category"]["macro_f1"],
         "params_or_notes": f"63,639 params, {dl['total_epochs_run']} epochs (best={dl['best_epoch']})"},
        {"approach": "Classical ML (tuned Logistic Regression, cascade)", "task": "priority",
         "test_accuracy": classical["priority_model"]["test_accuracy"],
         "test_macro_f1": classical["priority_model"]["test_macro_f1"],
         "params_or_notes": "text + department + role + predicted category"},
        {"approach": "Deep Learning (multi-task NumPy net, joint)", "task": "priority",
         "test_accuracy": dl["test_priority"]["accuracy"], "test_macro_f1": dl["test_priority"]["macro_f1"],
         "params_or_notes": "shared representation, joint loss"},
    ]
    df = pd.DataFrame(rows)
    df.to_csv(f"{MODELS_DIR}/comparison_table.csv", index=False)
    print(df.to_string(index=False))
    return df


def confidence_threshold_analysis(threshold=0.45):
    df = load_data()
    _, _, test = stratified_split(df)
    cat_pipe = joblib.load(f"{MODELS_DIR}/classical_category_model.joblib")

    probs = cat_pipe.predict_proba(test["clean_text"])
    classes = cat_pipe.classes_
    top1_idx = probs.argmax(axis=1)
    top1_conf = probs.max(axis=1)
    top1_pred = classes[top1_idx]
    correct = (top1_pred == test["category"].values)

    buckets = [(0.0, 0.3), (0.3, 0.45), (0.45, 0.6), (0.6, 0.8), (0.8, 1.01)]
    print("\n=== Accuracy by confidence bucket (test set) ===")
    bucket_rows = []
    for lo, hi in buckets:
        mask = (top1_conf >= lo) & (top1_conf < hi)
        n = mask.sum()
        acc = correct[mask].mean() if n > 0 else float("nan")
        bucket_rows.append({"confidence_range": f"[{lo:.2f}, {hi:.2f})", "n": int(n),
                             "pct_of_test_set": round(n / len(test) * 100, 1),
                             "top1_accuracy": round(float(acc), 3) if n > 0 else None})
        print(f"  conf in [{lo:.2f},{hi:.2f}): n={n:4d} ({n/len(test)*100:4.1f}%)  top1_acc={acc:.3f}" if n > 0
              else f"  conf in [{lo:.2f},{hi:.2f}): n=0")

    low_mask = top1_conf < threshold
    top3_idx = np.argsort(-probs, axis=1)[:, :3]
    top3_hit = np.array([test["category"].values[i] in classes[top3_idx[i]] for i in range(len(test))])
    print(f"\nAt threshold={threshold}: {low_mask.sum()} / {len(test)} test queries "
          f"({low_mask.mean()*100:.1f}%) route to the fallback UI.")
    print(f"  Top-1 accuracy on that low-confidence subset: {correct[low_mask].mean():.3f}")
    print(f"  Top-3 recall on that same subset (true label in top-3 shown to user): "
          f"{top3_hit[low_mask].mean():.3f}")
    print(f"  Top-1 accuracy on the high-confidence (auto-shown) subset: {correct[~low_mask].mean():.3f}")

    with open(f"{MODELS_DIR}/confidence_analysis.json", "w") as f:
        json.dump({
            "threshold": threshold,
            "buckets": bucket_rows,
            "low_confidence_pct": round(float(low_mask.mean()), 4),
            "low_confidence_top1_acc": round(float(correct[low_mask].mean()), 4),
            "low_confidence_top3_recall": round(float(top3_hit[low_mask].mean()), 4),
            "high_confidence_top1_acc": round(float(correct[~low_mask].mean()), 4),
        }, f, indent=2)


if __name__ == "__main__":
    print("=== Approach comparison (test set) ===")
    comparison_table()
    confidence_threshold_analysis()
