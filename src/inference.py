"""
DeskPilot inference service.

Core design for handling uncertain/out-of-distribution queries (explicitly requested:
"handle scenarios"): a single top-1 prediction is fine when the model is confident,
but wrong or misleading when it isn't. So every prediction returns a FULL ranked
distribution, and the caller (API / UI) applies a confidence threshold:

  - top-1 probability >= CONFIDENCE_THRESHOLD  -> show the recommendation directly
  - top-1 probability <  CONFIDENCE_THRESHOLD  -> show top-3 alternatives as a
    disambiguation menu instead of guessing, plus a "None of these - browse full
    catalog" escape hatch. This never lets the system silently guess wrong on an
    unfamiliar phrasing; it degrades gracefully to "let the human pick", which is
    both better UX and safer than a forced single answer.
"""
import joblib

from data_utils import clean_text, get_category_classes, PRIORITY_CLASSES

import os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
CONFIDENCE_THRESHOLD = 0.45
TOP_K = 3


class DeskPilotService:
    def __init__(self):
        self.cat_pipe = joblib.load(f"{MODELS_DIR}/classical_category_model.joblib")
        self.pri_pipe = joblib.load(f"{MODELS_DIR}/classical_priority_model.joblib")
        self.cat_classes = get_category_classes()

    def recommend(self, query_text: str, department: str = "Engineering",
                   requester_role: str = "Associate", top_k: int = TOP_K):
        text = clean_text(query_text)
        cat_probs = self.cat_pipe.predict_proba([text])[0]
        order = cat_probs.argsort()[::-1][:top_k]
        alternatives = [{"category": self.cat_pipe.classes_[i], "confidence": round(float(cat_probs[i]), 4)}
                        for i in order]
        top = alternatives[0]
        low_confidence = top["confidence"] < CONFIDENCE_THRESHOLD

        import pandas as pd
        pred_df = pd.DataFrame([{
            "clean_text": text, "department": department, "requester_role": requester_role,
            "pred_category": top["category"],
        }])
        pri_probs = self.pri_pipe.predict_proba(pred_df)[0]
        p_order = pri_probs.argsort()[::-1]
        priority_ranked = [{"priority": self.pri_pipe.classes_[i], "confidence": round(float(pri_probs[i]), 4)}
                            for i in p_order]

        return {
            "query": query_text,
            "top_recommendation": top["category"],
            "confidence": top["confidence"],
            "low_confidence": low_confidence,
            "alternatives": alternatives,
            "fallback_message": (
                "Not fully sure - here are the closest matches. Pick one, or browse the full catalog."
                if low_confidence else None
            ),
            "suggested_priority": priority_ranked[0]["priority"],
            "priority_confidence": priority_ranked[0]["confidence"],
            "priority_distribution": priority_ranked,
        }


if __name__ == "__main__":
    svc = DeskPilotService()
    samples = [
        ("need photoshop installed on my laptop for the design project", "Design", "Associate"),
        ("prod database is down and clients are affected right now", "Engineering", "Director"),
        ("hey can someone help me with my thing not working", "Finance", "Manager"),  # deliberately vague
        ("requesting vpn access for our new remote contractor", "Sales", "Manager"),
    ]
    for text, dept, role in samples:
        r = svc.recommend(text, dept, role)
        print(f"\nQuery: {text!r}  [{dept}/{role}]")
        print(f"  -> {r['top_recommendation']}  (confidence={r['confidence']:.2f}, "
              f"low_confidence={r['low_confidence']})")
        print(f"  priority: {r['suggested_priority']} ({r['priority_confidence']:.2f})")
        if r["low_confidence"]:
            print(f"  alternatives: {r['alternatives']}")
