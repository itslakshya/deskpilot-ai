"""
Classical ML baseline for DeskPilot.

Two SEPARATE models in a cascade (this is a deliberate architecture choice worth
explaining in interviews - contrast with the joint multi-task deep learning model
in train_dl.py):
  1) Category model:  TF-IDF(text) -> tuned classifier -> catalog category (19-way)
  2) Priority model:  TF-IDF(text) + one-hot(department, role, PREDICTED category)
                       -> tuned classifier -> priority (4-way)
     The priority model consumes category as a feature (not the ground truth at
     inference time) because in production we only have the category model's
     prediction available at that point - this is a "cascade" / pipeline design,
     and it also means category-model errors can propagate into priority
     predictions, a real production risk we discuss in docs/INTERVIEW_PREP.md.

Model selection: for EACH head we try Logistic Regression, Linear SVM, and Random
Forest, tune each with GridSearchCV (5-fold stratified CV, macro-F1 scoring - macro,
not accuracy, because classes are imbalanced ~10x and we care about minority-class
performance like Password Reset and Email/Outlook Issue), then pick the winner by
validation-set macro-F1. Test set is touched exactly once, at the end.
"""
import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import LinearSVC

from data_utils import get_category_classes, load_data, stratified_split, PRIORITY_CLASSES

import os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")


def build_category_pipeline(clf):
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.9, sublinear_tf=True)),
        ("clf", clf),
    ])


def build_priority_pipeline(clf):
    # a FRESH ColumnTransformer every call (not a shared module-level instance) - Pipeline.fit()
    # mutates its steps in place rather than cloning them, so reusing one instance across
    # multiple Pipeline objects would let an unrelated .fit() call silently change what an
    # already-fitted pipeline predicts. GridSearchCV clones internally and would have masked
    # this, but the direct refit path in select_serving_pipeline does not.
    pre = ColumnTransformer([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.9, sublinear_tf=True), "clean_text"),
        ("cat", OneHotEncoder(handle_unknown="ignore"), ["department", "requester_role", "pred_category"]),
    ])
    return Pipeline([("pre", pre), ("clf", clf)])


CANDIDATES = {
    "logreg": (LogisticRegression(max_iter=2000, class_weight="balanced"),
               {"clf__C": [0.1, 1.0, 3.0, 10.0]}),
    "linear_svc": (LinearSVC(class_weight="balanced", max_iter=5000),
                   {"clf__C": [0.1, 1.0, 3.0]}),
    "random_forest": (RandomForestClassifier(n_estimators=300, class_weight="balanced_subsample",
                                              random_state=42, n_jobs=-1),
                       {"clf__max_depth": [None, 30]}),
}


def tune_and_select(X_train, y_train, X_val, y_val, tag):
    """Grid-search each candidate, pick the one with best VALIDATION macro-F1."""
    results = {}
    best_name, best_pipe, best_f1 = None, None, -1
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for name, (clf, grid) in CANDIDATES.items():
        t0 = time.time()
        pipe = build_category_pipeline(clf) if tag == "category" else build_priority_pipeline(clf)
        gs = GridSearchCV(pipe, param_grid=grid, scoring="f1_macro", cv=cv, n_jobs=-1)
        gs.fit(X_train, y_train)
        val_pred = gs.predict(X_val)
        val_f1 = f1_score(y_val, val_pred, average="macro")
        results[name] = {
            "best_params": gs.best_params_, "cv_f1_macro": round(gs.best_score_, 4),
            "val_f1_macro": round(val_f1, 4), "train_seconds": round(time.time() - t0, 1),
        }
        print(f"  [{tag}] {name}: cv_f1={gs.best_score_:.4f} val_f1={val_f1:.4f} "
              f"params={gs.best_params_} ({time.time()-t0:.1f}s)")
        if val_f1 > best_f1:
            best_f1, best_name, best_pipe = val_f1, name, gs.best_estimator_
    return best_name, best_pipe, results


def select_serving_pipeline(tag, results, best_name, best_pipe, best_f1, X_train, y_train,
                             tolerance=0.015):
    """Shared safeguard for BOTH heads (this used to only guard the category head, which
    was itself a bug: whichever classifier wins validation F1 is not automatically safe to
    serve if it lacks predict_proba - the confidence-thresholded fallback UX in
    src/inference.py calls predict_proba unconditionally, and LinearSVC does not implement
    it at all (only decision_function margins). If the winner lacks predict_proba, we
    deploy the best probability-capable candidate instead, as long as it's within
    `tolerance` macro-F1 of the winner (a difference that small is noise on a ~1000-row
    validation set anyway)."""
    if hasattr(best_pipe.named_steps["clf"], "predict_proba"):
        return best_name, best_pipe
    for name, res in sorted(results.items(), key=lambda kv: -kv[1]["val_f1_macro"]):
        if name == best_name:
            continue
        if res["val_f1_macro"] < best_f1 - tolerance:
            break  # sorted descending - nothing further will be within tolerance either
        clf, _ = CANDIDATES[name]
        cand_pipe = build_category_pipeline(clf) if tag == "category" else build_priority_pipeline(clf)
        cand_pipe.set_params(**res["best_params"])
        cand_pipe.fit(X_train, y_train)
        if hasattr(cand_pipe.named_steps["clf"], "predict_proba"):
            print(f">> [{tag}] Deploying '{name}' instead of '{best_name}' (val_f1={res['val_f1_macro']:.4f}, "
                  f"within {best_f1-res['val_f1_macro']:.4f} of best) - '{best_name}' lacks predict_proba, "
                  f"needed for confidence-based fallback.\n")
            return name, cand_pipe
    # nothing probability-capable was close enough - fail loudly rather than silently
    # shipping a model that will crash the first time /predict is called
    raise RuntimeError(
        f"[{tag}] Best model '{best_name}' has no predict_proba, and no probability-capable "
        f"candidate scored within {tolerance} macro-F1 of it ({best_f1:.4f}). Refusing to "
        f"save a model that would crash src/inference.py at serving time. Either widen "
        f"`tolerance`, or drop 'linear_svc' from CANDIDATES for this head.")


def main():
    df = load_data()
    cat_classes = get_category_classes()
    train, val, test = stratified_split(df)

    print(f"train={len(train)} val={len(val)} test={len(test)}  |  {len(cat_classes)} categories\n")

    # ---------------- Head 1: Category classifier (text-only) ----------------
    print("=== Tuning CATEGORY models ===")
    best_name, best_pipe, cat_results = tune_and_select(
        train["clean_text"], train["category"], val["clean_text"], val["category"], "category"
    )
    best_f1 = cat_results[best_name]["val_f1_macro"]
    print(f"\n>> Top validation performer: {best_name} (val_f1={best_f1:.4f})\n")

    # --- Production/serving selection is NOT purely "highest validation F1" ---
    # The confidence-based fallback UX (docs/INTERVIEW_PREP.md) needs *calibrated-ish*
    # class probabilities, so both heads run through the same predict_proba safeguard -
    # see select_serving_pipeline() above.
    best_name, best_pipe = select_serving_pipeline(
        "category", cat_results, best_name, best_pipe, best_f1,
        train["clean_text"], train["category"]
    )

    test_pred_cat = best_pipe.predict(test["clean_text"])
    cat_report = classification_report(test["category"], test_pred_cat, output_dict=True, zero_division=0)
    print("TEST SET category report (macro avg):", {k: round(v, 3) for k, v in cat_report["macro avg"].items()})

    joblib.dump(best_pipe, f"{MODELS_DIR}/classical_category_model.joblib")

    # ---------------- Head 2: Priority classifier (text + tabular + predicted category) ----------------
    print("\n=== Tuning PRIORITY models (cascade: uses category model's predictions) ===")

    def add_predicted_category(split_df, pipe):
        split_df = split_df.copy()
        split_df["pred_category"] = pipe.predict(split_df["clean_text"])
        return split_df

    train_p = add_predicted_category(train, best_pipe)
    val_p = add_predicted_category(val, best_pipe)
    test_p = add_predicted_category(test, best_pipe)

    best_p_name, best_p_pipe, priority_results = tune_and_select(
        train_p, train_p["priority"], val_p, val_p["priority"], "priority"
    )
    best_p_f1 = priority_results[best_p_name]["val_f1_macro"]
    print(f"\n>> Top validation performer: {best_p_name} (val_f1={best_p_f1:.4f})\n")

    best_p_name, best_p_pipe = select_serving_pipeline(
        "priority", priority_results, best_p_name, best_p_pipe, best_p_f1,
        train_p, train_p["priority"]
    )

    test_pred_pri = best_p_pipe.predict(test_p)
    pri_report = classification_report(test_p["priority"], test_pred_pri, output_dict=True, zero_division=0)
    print("TEST SET priority report (macro avg):", {k: round(v, 3) for k, v in pri_report["macro avg"].items()})

    joblib.dump(best_p_pipe, f"{MODELS_DIR}/classical_priority_model.joblib")

    # ---------------- Save everything needed for reporting / comparison ----------------
    summary = {
        "category_model": {"selected": best_name, "candidates": cat_results,
                            "test_macro_f1": round(cat_report["macro avg"]["f1-score"], 4),
                            "test_weighted_f1": round(cat_report["weighted avg"]["f1-score"], 4),
                            "test_accuracy": round(cat_report["accuracy"], 4)},
        "priority_model": {"selected": best_p_name, "candidates": priority_results,
                            "test_macro_f1": round(pri_report["macro avg"]["f1-score"], 4),
                            "test_weighted_f1": round(pri_report["weighted avg"]["f1-score"], 4),
                            "test_accuracy": round(pri_report["accuracy"], 4)},
        "n_train": len(train), "n_val": len(val), "n_test": len(test),
    }
    with open(f"{MODELS_DIR}/classical_results.json", "w") as f:
        json.dump(summary, f, indent=2)

    # confusion matrix for category (for the report / docs)
    cm = confusion_matrix(test["category"], test_pred_cat, labels=cat_classes)
    pd.DataFrame(cm, index=cat_classes, columns=cat_classes).to_csv(f"{MODELS_DIR}/classical_category_confusion.csv")

    print("\nFull per-class category report:")
    print(classification_report(test["category"], test_pred_cat, zero_division=0))

    print("\nSaved: classical_category_model.joblib, classical_priority_model.joblib, "
          "classical_results.json, classical_category_confusion.csv")


if __name__ == "__main__":
    main()
