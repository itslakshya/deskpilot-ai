"""
Export the trained classical pipelines (TF-IDF + Logistic Regression) to a compact
JSON format that a browser can run WITHOUT a Python backend - reimplemented in plain
JS in frontend/index.html (TF-IDF transform + linear layer + softmax). This lets the
demo UI be a real, live ML inference, not a mockup, while the FastAPI backend
(backend/app.py) remains the actual production serving path for the full system.
"""
import json
import os

import joblib
import numpy as np

from data_utils import get_category_classes, PRIORITY_CLASSES

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
OUT_PATH = os.path.join(PROJECT_ROOT, "frontend", "model_weights.json")


def r(x, nd=5):
    return float(np.round(x, nd))


def export_tfidf_logreg(vec, clf, classes):
    vocab = {k: int(v) for k, v in vec.vocabulary_.items()}
    idf = [r(x) for x in vec.idf_]
    coef = [[r(x) for x in row] for row in clf.coef_]
    intercept = [r(x) for x in clf.intercept_]
    return dict(vocabulary=vocab, idf=idf, coef=coef, intercept=intercept, classes=list(classes),
                ngram_range=list(vec.ngram_range))


def main():
    cat_pipe = joblib.load(f"{MODELS_DIR}/classical_category_model.joblib")
    cat_clf = cat_pipe.named_steps["clf"]
    # IMPORTANT: use the classifier's OWN fitted class order (clf.classes_, alphabetically
    # sorted by sklearn), not the schema's enum order - coef_ row i corresponds to
    # classes_[i], and exporting the wrong order silently mismatches probabilities to labels.
    cat_export = export_tfidf_logreg(cat_pipe.named_steps["tfidf"], cat_clf, cat_clf.classes_)

    pri_pipe = joblib.load(f"{MODELS_DIR}/classical_priority_model.joblib")
    pre = pri_pipe.named_steps["pre"]
    pclf = pri_pipe.named_steps["clf"]
    tfidf_p = pre.named_transformers_["tfidf"]
    ohe = pre.named_transformers_["cat"]  # fitted on [department, requester_role, pred_category]
    dept_cats, role_cats, predcat_cats = [list(c) for c in ohe.categories_]

    n_tfidf_features = len(tfidf_p.vocabulary_)
    priority_export = dict(
        vocabulary={k: int(v) for k, v in tfidf_p.vocabulary_.items()},
        idf=[r(x) for x in tfidf_p.idf_],
        ngram_range=list(tfidf_p.ngram_range),
        n_tfidf_features=n_tfidf_features,
        dept_categories=dept_cats, role_categories=role_cats, predcat_categories=predcat_cats,
        coef=[[r(x) for x in row] for row in pclf.coef_],
        intercept=[r(x) for x in pclf.intercept_],
        classes=list(pclf.classes_),
    )

    payload = dict(category_model=cat_export, priority_model=priority_export,
                    catalog_categories=list(cat_clf.classes_))
    with open(OUT_PATH, "w") as f:
        json.dump(payload, f, separators=(",", ":"))

    import os
    size_kb = os.path.getsize(OUT_PATH) / 1024
    print(f"Exported web model weights -> {OUT_PATH} ({size_kb:.0f} KB)")


if __name__ == "__main__":
    main()
