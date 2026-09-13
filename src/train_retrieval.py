"""
Retrieval-based (Rocchio/centroid) classifier - a third, classic approach included
deliberately for breadth: it needs no gradient-based training at all, just TF-IDF
vectors and cosine similarity, and is the natural stepping stone to explain in an
interview toward a semantic-search/RAG-style catalog lookup (nearest catalog item by
embedding similarity) as a "future work" extension documented in docs/INTERVIEW_PREP.md.

Method: represent each catalog category by the CENTROID (mean) of its training
examples' TF-IDF vectors. At inference, embed the query and return the category whose
centroid has the highest cosine similarity - along with the full ranked similarity
list, which is what powers the "top-3 alternatives" fallback UI.
"""
import json

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, f1_score
from sklearn.metrics.pairwise import cosine_similarity

from data_utils import get_category_classes, load_data, stratified_split

import os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")


def main():
    df = load_data()
    cat_classes = get_category_classes()
    train, val, test = stratified_split(df)

    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.9, sublinear_tf=True)
    Xtr = vec.fit_transform(train["clean_text"])

    centroids = np.zeros((len(cat_classes), Xtr.shape[1]))
    for i, c in enumerate(cat_classes):
        mask = (train["category"] == c).values
        if mask.sum() > 0:
            centroids[i] = Xtr[mask].mean(axis=0)

    def predict(texts):
        Xq = vec.transform(texts)
        sims = cosine_similarity(Xq, centroids)  # (n, n_classes)
        pred_idx = sims.argmax(axis=1)
        return np.array([cat_classes[i] for i in pred_idx]), sims

    test_pred, test_sims = predict(test["clean_text"])
    report = classification_report(test["category"], test_pred, output_dict=True, zero_division=0)
    macro_f1 = f1_score(test["category"], test_pred, average="macro")

    print("=== Retrieval (centroid cosine-similarity) classifier ===")
    print(f"test accuracy = {report['accuracy']:.4f}   macro F1 = {macro_f1:.4f}")

    with open(f"{MODELS_DIR}/retrieval_results.json", "w") as f:
        json.dump({"test_accuracy": round(report["accuracy"], 4),
                   "test_macro_f1": round(macro_f1, 4),
                   "test_weighted_f1": round(report["weighted avg"]["f1-score"], 4)}, f, indent=2)

    import joblib
    joblib.dump({"vectorizer": vec, "centroids": centroids, "classes": cat_classes},
                f"{MODELS_DIR}/retrieval_model.joblib")
    print("Saved: retrieval_model.joblib, retrieval_results.json")


if __name__ == "__main__":
    main()
