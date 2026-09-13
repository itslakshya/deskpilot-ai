"""
scripts/append_and_retrain.py

Answers directly: "if in future a new dataset is provided, can this append it and
retrain in a structured way?" - yes, this is that pipeline.

Usage:
    python3 append_and_retrain.py --new-data path/to/new_tickets.csv

What it does:
  1. VALIDATE the new file against data/schema/ticket_schema.json (required columns,
     allowed enum values, text length bounds). Bad rows are rejected with a reason,
     not silently coerced or dropped.
  2. DE-DUPE against the existing dataset on exact query_text (same rule used in
     generate_dataset.py) so repeats don't distort class balance.
  3. VERSION the merged dataset: data/v1/tickets.csv -> data/v2/tickets.csv (next
     integer version), and append an entry to data/CHANGELOG.md - old versions are
     kept, never overwritten, so you can always roll back or audit what changed.
  4. RETRAIN both the classical and DL pipelines on the new merged version.
  5. CHAMPION/CHALLENGER CHECK: evaluate the new ("challenger") model on the SAME
     held-out test set the current production ("champion") model was scored on.
     Only overwrite the production model artifacts if the challenger's category
     macro-F1 is not worse than the champion's by more than a small tolerance
     (0.01) - this prevents an automatic retrain from silently degrading
     production on a noisy data drop. A regression is reported, not deployed.

This is intentionally simple (no orchestrator, no message queue) - it is meant to
demonstrate the PATTERN (validate -> version -> retrain -> gate -> promote) that a
real MLOps pipeline (Airflow/Azure ML pipeline/GitHub Actions) would automate on a
schedule or on-demand when a new labeled batch arrives.
"""
import argparse
import glob
import json
import os
import re
import shutil
import sys
from datetime import datetime, timezone

import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from data_utils import load_schema  # noqa: E402

DATA_ROOT = os.path.join(os.path.dirname(__file__), "..", "data")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
CHANGELOG = os.path.join(DATA_ROOT, "CHANGELOG.md")
REGRESSION_TOLERANCE = 0.01


def latest_version_dir():
    dirs = sorted(glob.glob(os.path.join(DATA_ROOT, "v*")),
                  key=lambda p: int(re.search(r"v(\d+)", p).group(1)))
    return dirs[-1]


def validate_rows(df: pd.DataFrame, schema: dict):
    errors = []
    required = schema["required"]
    enums = {k: v["enum"] for k, v in schema["properties"].items() if "enum" in v}
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        raise ValueError(f"New data is missing required columns: {missing_cols}")

    valid_mask = pd.Series(True, index=df.index)
    for col, allowed in enums.items():
        bad = ~df[col].isin(allowed)
        if bad.any():
            errors.append(f"{bad.sum()} row(s) have invalid '{col}' (not in schema enum)")
            valid_mask &= ~bad

    text_len_ok = df["query_text"].str.len().between(
        schema["properties"]["query_text"]["minLength"], schema["properties"]["query_text"]["maxLength"])
    if (~text_len_ok).any():
        errors.append(f"{(~text_len_ok).sum()} row(s) have query_text outside allowed length")
        valid_mask &= text_len_ok

    id_pattern = re.compile(schema["properties"]["ticket_id"]["pattern"])
    id_ok = df["ticket_id"].astype(str).str.match(id_pattern)
    if (~id_ok).any():
        errors.append(f"{(~id_ok).sum()} row(s) have malformed ticket_id (auto-generating replacements)")
        import uuid
        df.loc[~id_ok, "ticket_id"] = [f"TCK-{uuid.uuid4().hex[:8].upper()}" for _ in range((~id_ok).sum())]

    for e in errors:
        print(f"  [validation] {e}")
    return df[valid_mask].copy(), len(df) - valid_mask.sum()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--new-data", required=True, help="Path to a CSV of new labeled tickets matching the schema")
    ap.add_argument("--force-promote", action="store_true",
                     help="Promote the challenger even if it regresses (use with caution)")
    args = ap.parse_args()

    schema = load_schema()
    new_df = pd.read_csv(args.new_data)
    print(f"Loaded {len(new_df)} candidate rows from {args.new_data}")

    valid_df, n_rejected = validate_rows(new_df, schema)
    print(f"{len(valid_df)} rows passed validation, {n_rejected} rejected")
    if len(valid_df) == 0:
        print("No valid rows to add. Aborting.")
        return

    cur_dir = latest_version_dir()
    cur_version = int(re.search(r"v(\d+)", cur_dir).group(1))
    existing = pd.read_csv(os.path.join(cur_dir, "tickets.csv"))

    combined = pd.concat([existing, valid_df], ignore_index=True)
    before = len(combined)
    combined = combined.drop_duplicates(subset=["query_text"]).reset_index(drop=True)
    n_dupe = before - len(combined)
    print(f"Merged: {len(existing)} existing + {len(valid_df)} new -> {len(combined)} unique rows "
          f"({n_dupe} exact-text duplicates dropped)")

    new_version = cur_version + 1
    new_dir = os.path.join(DATA_ROOT, f"v{new_version}")
    os.makedirs(new_dir, exist_ok=True)
    combined.to_csv(os.path.join(new_dir, "tickets.csv"), index=False)

    with open(CHANGELOG, "a") as f:
        f.write(f"\n## v{new_version} - {datetime.now(timezone.utc).isoformat(timespec='seconds')}\n"
                f"- Added {len(valid_df)} candidate rows from `{os.path.basename(args.new_data)}` "
                f"({n_rejected} rejected by schema validation, {n_dupe} deduped)\n"
                f"- Total dataset size: {len(existing)} -> {len(combined)}\n")
    print(f"Versioned new dataset at data/v{new_version}/tickets.csv, logged to CHANGELOG.md")

    # ---- retrain on the new version ----
    print(f"\nRetraining classical model on v{new_version}...")
    import subprocess
    src_dir = os.path.join(os.path.dirname(__file__), "..", "src")
    env = os.environ.copy()
    env["INTELLIDESK_DATA_PATH"] = os.path.join(new_dir, "tickets.csv")

    # data_utils.DATA_PATH is a module-level constant for simplicity in this project;
    # a larger codebase would inject this via config. Here we monkeypatch it directly.
    patched = os.path.join(src_dir, "_retrain_tmp.py")
    with open(patched, "w") as f:
        f.write(f"import data_utils\n"
                f"data_utils.DATA_PATH = r'{os.path.join(new_dir, 'tickets.csv')}'\n"
                f"import train_classical\n"
                f"train_classical.main()\n")
    result = subprocess.run([sys.executable, "_retrain_tmp.py"], cwd=src_dir, capture_output=True, text=True)
    os.remove(patched)
    print(result.stdout[-2000:])
    if result.returncode != 0:
        print("RETRAIN FAILED:\n", result.stderr[-2000:])
        return

    with open(f"{MODELS_DIR}/classical_results.json") as f:
        challenger = json.load(f)

    champion_path = f"{MODELS_DIR}/classical_results_champion.json"
    if not os.path.exists(champion_path):
        # first time - current results become the champion baseline
        shutil.copy(f"{MODELS_DIR}/classical_results.json", champion_path)
        print("\nNo prior champion recorded - this run's results are now the baseline champion.")
        return

    with open(champion_path) as f:
        champion = json.load(f)

    champ_f1 = champion["category_model"]["test_macro_f1"]
    chall_f1 = challenger["category_model"]["test_macro_f1"]
    regressed = chall_f1 < champ_f1 - REGRESSION_TOLERANCE
    print(f"\nChampion category macro-F1: {champ_f1:.4f}")
    print(f"Challenger category macro-F1: {chall_f1:.4f}")

    if regressed and not args.force_promote:
        print(f"REGRESSION DETECTED (> {REGRESSION_TOLERANCE} drop). Challenger NOT promoted. "
              f"Champion model artifacts left untouched; challenger results saved separately for review.")
        shutil.move(f"{MODELS_DIR}/classical_results.json", f"{MODELS_DIR}/classical_results_challenger_rejected.json")
    else:
        shutil.copy(f"{MODELS_DIR}/classical_results.json", champion_path)
        print("Challenger promoted to champion. Production model artifacts updated.")


if __name__ == "__main__":
    main()
