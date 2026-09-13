# DeskPilot AI

**Intent classification + priority recommendation for enterprise IT service desk tickets.**

At a global helpdesk (the scenario used here mirrors a TCS-style enterprise service desk), employees raise Incidents, Service Requests, Change Requests (RFCs), and Admin requests by browsing a service catalog with 100+ items organized in a category tree. IntelliDesk replaces the browsing step: an employee types their request in plain English, and a trained model recommends the correct catalog item and priority level directly - with graceful fallback to a short pick-list when the request is genuinely ambiguous, rather than guessing.

```
"need Adobe Photoshop installed on my laptop"
        │
        ▼
  ┌─────────────────────────────┐
  │  Software Installation Req. │   confidence: 63%
  │  Priority: P4 - Low         │
  └─────────────────────────────┘
```

**Live demo:** `frontend/index.html` - runs a real, trained model entirely in your browser (no server needed to try it; see [Demo](#demo)).

---

## Why this project exists

This was built as an end-to-end portfolio piece to demonstrate applied ML/DL judgment, not just library calls: generating and validating a dataset, choosing between three genuinely different modeling approaches on evidence rather than defaults, handling class imbalance and label noise honestly, building a confidence-aware serving layer, and wiring up the pieces (schema validation, versioned retraining, drift detection) that separate a notebook from a system.

## Architecture

```
                     ┌───────────────────────────┐
                     │   data/v{N}/tickets.csv    │  versioned, schema-validated
                     └─────────────┬─────────────┘
                                   │ stratified 70/15/15 split (no leakage)
                 ┌─────────────────┼─────────────────┐
                 ▼                 ▼                 ▼
      ┌───────────────────┐ ┌───────────────┐ ┌─────────────────────┐
      │ Retrieval baseline │ │ Classical ML  │ │ Deep learning        │
      │ TF-IDF centroid    │ │ TF-IDF + tuned│ │ from-scratch NumPy   │
      │ cosine similarity  │ │ LogReg/SVM/RF │ │ multi-task net       │
      └───────────────────┘ └───────┬───────┘ └──────────────────────┘
                                     │  (best test macro-F1 → served)
                                     ▼
                       ┌─────────────────────────────┐
                       │   src/inference.py            │
                       │   confidence-thresholded,     │
                       │   top-k fallback logic         │
                       └───────┬─────────────┬────────┘
                               ▼             ▼
                    ┌───────────────┐ ┌─────────────────────────┐
                    │ backend/app.py │ │ frontend/index.html      │
                    │ FastAPI serving│ │ exported weights, live   │
                    │ + feedback log │ │ in-browser inference     │
                    └───────────────┘ └─────────────────────────┘
```

## Results (held-out test set, 956 tickets, never touched during training or model selection)

| Approach | Task | Accuracy | Macro-F1 | Notes |
|---|---|---:|---:|---|
| Retrieval (TF-IDF centroid, cosine similarity) | category (19-way) | 94.9% | 0.940 | zero trainable parameters |
| **Classical ML - tuned Logistic Regression** *(served)* | category (19-way) | **97.1%** | **0.962** | ~54K coefficients, grid-searched |
| Deep learning - from-scratch multi-task NumPy net | category (19-way) | 97.1% | 0.961 | 63,639 params, 18 epochs (best=12) |
| Classical ML - Logistic Regression (cascade) | priority (4-way) | 53.7% | 0.503 | text + department + role + predicted category |
| Deep learning - multi-task net (joint) | priority (4-way) | 54.6% | 0.502 | shared representation, joint loss |

**The honest, non-obvious finding:** the deep learning model does not beat the tuned classical model on category classification, and only marginally edges it on priority. On ~4,500 training examples with a fairly clean, keyword-rich vocabulary, a linear model with good features is competitive with a neural network - this project's dataset didn't need deep learning to solve the primary task well. That's a real, defensible finding, documented rather than hidden (see `docs/INTERVIEW_PREP.md` for how to talk about this and when the answer would flip).

**Why priority tops out around 0.50 macro-F1, not higher:** priority labels are *intentionally* not a deterministic function of the input in this dataset - they depend on category base-rate plus requester seniority plus department plus genuine random business variation (a Director's routine software request can outrank an Associate's minor incident, but not always). This mirrors reality: two humans given the same ticket wouldn't always agree on priority either. The ceiling here is a property of the label-generating process, not a modeling failure - see the confidence distribution below for how the system compensates.

### Confidence calibration

| Confidence bucket | % of test set | Top-1 accuracy |
|---|---:|---:|
| 0.00 - 0.30 | 0.1% | low (n=1) |
| 0.30 - 0.45 | 0.5% | 100% |
| 0.45 - 0.60 | 2.9% | 100% |
| 0.60 - 0.80 | 34.5% | 97.0% |
| 0.80 - 1.00 | 61.9% | 97.1% |

At the production threshold (0.45), only 0.6% of the clean test set routes to the fallback UI - but genuinely novel phrasing (not covered by any training template) triggers it reliably; e.g. *"prod database is down and clients are affected right now"* and *"my thing isn't working"* both correctly drop to low-confidence mode in the live demo, with 100% top-3 recall on the low-confidence subset of the test set. See `src/evaluate_comparison.py` for the full analysis.

## Demo

Open `frontend/index.html` in any browser. It embeds the actual exported TF-IDF + Logistic Regression weights and reimplements TF-IDF vectorization + the linear layer + softmax in vanilla JS - verified to match `sklearn`'s `predict_proba` output exactly (see `src/export_web_model.py` and the verification harness referenced in `docs/INTERVIEW_PREP.md`). It is a real inference engine, not a UI mockup.

## Running the full system

```bash
pip install -r requirements.txt

# 1. Generate the dataset (or use the one already in data/v1/)
python3 src/generate_dataset.py
python3 src/data_utils.py          # creates the stratified train/val/test split

# 2. Train all three approaches
python3 src/train_classical.py
python3 src/train_dl.py
python3 src/train_retrieval.py

# 3. Compare + export
python3 src/evaluate_comparison.py
python3 src/export_web_model.py    # refreshes frontend/model_weights.json

# 4. Serve
cd backend
uvicorn app:app --reload --port 8000
# POST /predict  {"query_text": "...", "department": "...", "requester_role": "..."}

# or, containerized:
docker build -t intellidesk .
docker run -p 8000:8000 intellidesk
```

> **Windows users:** the commands above are written one-per-line specifically so they work
> in every shell (bash, cmd, and Windows PowerShell) without changes. If you see
> `The token '&&' is not a valid statement separator in this version` while following
> other guides online, it's because `&&` chaining only works in PowerShell 7+, not the
> default "Windows PowerShell" (5.1) that ships with Windows. Either run commands on
> separate lines (as above), use `;` instead of `&&` in PowerShell, or install
> [PowerShell 7](https://aka.ms/powershell) if you want `&&` to work as in bash.

## Repo structure

```
data/
  schema/ticket_schema.json   # required columns, enums, validation rules
  v1/tickets.csv               # 6,371 synthetic tickets (current production dataset)
  CHANGELOG.md                 # dataset version history
src/
  generate_dataset.py          # synthetic data generator (templates + noise injection)
  data_utils.py                # cleaning, stratified split, label encoding
  train_classical.py           # TF-IDF + LogReg/SVM/RF, tuned, model-selected
  nn_from_scratch.py           # multi-task NN: embedding, backprop, Adam - all NumPy
  train_dl.py                  # training loop: epochs, early stopping, class weights
  train_retrieval.py           # TF-IDF centroid cosine-similarity baseline
  inference.py                 # confidence-thresholded prediction service
  export_web_model.py          # exports weights for the browser demo
  evaluate_comparison.py       # head-to-head comparison + confidence analysis
backend/app.py                 # FastAPI: /predict, /feedback, /health, /model-info
frontend/
  template.html + model_weights.json → index.html   # live browser demo
scripts/
  append_and_retrain.py        # validate → version → retrain → champion/challenger gate
  drift_check.py                # chi-square test: recent predictions vs training distribution
models/                        # trained artifacts + all results JSON
docs/
  training_curves.png, confusion_matrix.png
  pytorch_reference_architecture.py   # production-scale architecture (see below)
  INTERVIEW_PREP.md
```

## Appending new data / retraining

This was a specific design requirement: the system needs to accept a new batch of labeled tickets in the future and incorporate it without manual surgery.

```bash
python3 scripts/append_and_retrain.py --new-data path/to/new_tickets.csv
```

This: (1) validates every row against `data/schema/ticket_schema.json`, rejecting rows with invalid categories/priorities/malformed IDs rather than silently coercing them; (2) merges with the existing dataset, de-duplicating on exact query text; (3) writes a new versioned dataset (`data/v2/`, `data/v3/`, ...) and appends to `data/CHANGELOG.md` - old versions are never overwritten; (4) retrains; (5) only promotes the retrained model to production if it doesn't regress category macro-F1 by more than 0.01 against the current champion (a champion/challenger pattern) - a bad data drop can't silently degrade production.

`scripts/drift_check.py` runs a chi-square test comparing a recent batch of predictions' category distribution against the training distribution, as an early signal that a retrain may be worth considering.

## A note on exact reproducibility across machines

Running `generate_dataset.py` on a different machine can produce a slightly different row count (e.g. 6,383 rows instead of 6,371) and metrics that differ by a fraction of a percentage point, even though every seed in this codebase is fixed. This is expected, not a bug: `Faker` and `numpy` don't guarantee byte-identical output across their own versions for the same seed, `requirements.txt` uses `>=` version floors rather than exact pins, and small text differences change which rows collide during de-duplication. True bit-for-bit reproducibility across machines would require pinning the entire dependency tree (including transitive dependencies and BLAS backends) - a real engineering cost that's usually only worth paying in regulated environments. What *is* guaranteed, and was verified, is that the qualitative findings (classical ML ties/beats the from-scratch DL model, priority tops out around 0.50 macro-F1, the confidence threshold correctly isolates ambiguous queries) hold regardless of these small variations - see the "Design decisions" section below for a case where exactly this kind of cross-environment variation surfaced a real bug.

## Design decisions worth highlighting

- **Cascade (classical) vs. joint multi-task (DL) architectures, built side by side.** The classical pipeline predicts category, then feeds that prediction into a second model for priority. The neural net predicts both from one shared representation, trained jointly with a weighted loss. This is a real architectural fork worth being able to explain, not just a modeling afterthought.
- **Production model selection wasn't purely "highest validation score."** Linear SVC edged out Logistic Regression on category by 0.3 macro-F1 points (statistically noise on a 956-row validation set) but doesn't expose calibrated probabilities. Logistic Regression was deployed instead, deliberately, because the confidence-threshold fallback UX depends on real probabilities - see `src/train_classical.py`.
- **Label noise and class imbalance are deliberately part of the dataset**, not incidental. A first version of the generator produced a trivially separable, perfectly-labeled dataset that hit 100% test accuracy - a red flag, not a win - so realistic ~4.5% mislabeling between confusable category pairs (e.g. Software Installation vs. Software License) and ~10x class imbalance were added back in. See `docs/INTERVIEW_PREP.md` for the full story.
- **Reproducibility bug, found and fixed during development:** the first training runs of the neural net gave different results each run despite a fixed seed - traced to an unseeded `np.random.permutation` call for epoch shuffling, sitting alongside a properly-seeded `Generator` used for dropout. Fixed by seeding both sources of randomness explicitly.
- **A cross-environment bug, found because someone else ran the code on different data.** The category model had a safeguard - if the top validation performer lacks `predict_proba` (as `LinearSVC` does), fall back to the best candidate that has it, since the confidence-threshold UX needs real probabilities. That safeguard was only applied to the category head, not the priority head - an oversight invisible on my machine because logistic regression happened to win priority validation there. On a different machine, with a different dependency version and a very slightly different generated dataset, `LinearSVC` won priority validation instead and got saved as the production model - which would have crashed the very first `/predict` call with `AttributeError: 'LinearSVC' object has no attribute 'predict_proba'`. Fixed by refactoring the safeguard into one shared function (`select_serving_pipeline`) applied identically to both heads, with a loud `RuntimeError` (not a silent skip) if no probability-capable candidate is close enough to deploy. This is a good "tell me about a bug you shipped" story precisely because it's about a missing safeguard on one of two *symmetric* code paths, not a one-off typo - the kind of bug that survives code review because each path looks locally correct.

## Limitations & future work

- **Bag-of-embeddings, not sequence-aware.** Both the classical (TF-IDF) and from-scratch DL model are order-insensitive at the n-gram level; "vpn not working" and "not working, vpn" look identical to the mean-pooled embedding. `docs/pytorch_reference_architecture.py` documents a BiLSTM encoder (autograd, GPU-ready) that would fix this at production scale.
- **Semantic search extension.** The retrieval baseline (TF-IDF centroid) is a natural stepping stone to a proper embedding-based nearest-neighbor lookup against catalog item descriptions (a lightweight RAG-style pattern) - useful when new catalog items are added faster than labeled training examples accumulate for them.
- **Explicit uncertainty weighting for the multi-task loss** (e.g. Kendall et al.'s homoscedastic uncertainty weighting, or GradNorm) instead of the fixed 0.6/0.4 split used here.
- **A/B testing infrastructure** for champion vs. challenger models in the live serving path, not just offline evaluation.

## Resume bullet points (pick what fits the format)

- Designed and shipped an end-to-end ML system that classifies free-text IT service requests into 19 catalog categories (97.1% accuracy, 0.96 macro-F1) and recommends ticket priority, replacing manual service-catalog browsing; built three benchmarked approaches (retrieval, classical ML, a from-scratch multi-task neural network) and shipped the evidence-based winner.
- Implemented a multi-task neural network from scratch in NumPy (embedding layer, shared encoder, dual softmax heads, manual backpropagation, Adam optimizer) trained with class-weighted loss, early stopping, and learning-rate decay; benchmarked against a hyperparameter-tuned classical ML pipeline to make and justify a data-driven model selection.
- Built a confidence-thresholded serving layer with graceful fallback to top-k alternatives for out-of-distribution requests, verified via a held-out calibration analysis (100% top-3 recall on low-confidence predictions); deployed via FastAPI with an active-learning feedback loop and a schema-validated, versioned data-append/retrain pipeline with automatic champion/challenger regression gating.
