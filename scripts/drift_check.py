"""
scripts/drift_check.py

A minimal but real drift check: compares the CATEGORY distribution of recent
predictions (as logged by the backend, or a batch of recent tickets) against the
distribution the model was trained on, using a chi-square goodness-of-fit test.
A significant deviation means the mix of incoming requests has shifted - e.g. a new
department onboarded, a new tool rolled out, a seasonal spike in onboarding tickets -
which is a signal to review whether the model still needs retraining, NOT proof the
model is wrong (distribution shift and model error are different things; this check
only catches the former).

Usage:
    python3 drift_check.py --recent path/to/recent_predictions.csv

recent_predictions.csv just needs a 'category' column (one row per prediction).
In production this would read from wherever backend/app.py logs predictions, on a
schedule (e.g. a nightly cron / Azure Function).
"""
import argparse
import json
import os
import sys

import pandas as pd
from scipy.stats import chisquare

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from data_utils import get_category_classes, load_data  # noqa: E402

ALPHA = 0.01  # significance level - conservative, to avoid alerting on normal sampling noise


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recent", required=True, help="CSV with a 'category' column of recent predictions")
    args = ap.parse_args()

    train_df = load_data()
    recent_df = pd.read_csv(args.recent)
    classes = get_category_classes()

    train_counts = train_df["category"].value_counts().reindex(classes, fill_value=0)
    train_probs = train_counts / train_counts.sum()

    recent_counts = recent_df["category"].value_counts().reindex(classes, fill_value=0)
    n_recent = recent_counts.sum()
    if n_recent < 30:
        print(f"Only {n_recent} recent predictions - too few for a reliable chi-square test "
              f"(need >=30). Skipping statistical test, showing raw comparison only.")
        expected = None
    else:
        expected = train_probs * n_recent
        # chi-square requires expected counts to not be too small; classes with <5 expected
        # are merged into an 'other' bucket to keep the test valid
        small = expected < 5
        if small.sum() > 0:
            obs_main, exp_main = recent_counts[~small], expected[~small]
            obs_other, exp_other = recent_counts[small].sum(), expected[small].sum()
            obs = list(obs_main) + [obs_other]
            exp = list(exp_main) + [exp_other]
        else:
            obs, exp = list(recent_counts), list(expected)
        stat, pval = chisquare(f_obs=obs, f_exp=exp)
        print(f"Chi-square drift test: statistic={stat:.2f}, p-value={pval:.4f}")
        if pval < ALPHA:
            print(f"DRIFT DETECTED (p < {ALPHA}): recent category distribution differs "
                  f"significantly from training distribution. Recommend reviewing recent "
                  f"tickets and considering a retrain via scripts/append_and_retrain.py.")
        else:
            print(f"No significant drift detected (p >= {ALPHA}).")

    comparison = pd.DataFrame({
        "train_pct": (train_probs * 100).round(2),
        "recent_pct": (recent_counts / max(n_recent, 1) * 100).round(2),
    })
    comparison["delta_pct_points"] = (comparison["recent_pct"] - comparison["train_pct"]).round(2)
    print("\nPer-category share (train vs recent):")
    print(comparison.sort_values("delta_pct_points", key=abs, ascending=False).to_string())


if __name__ == "__main__":
    main()
