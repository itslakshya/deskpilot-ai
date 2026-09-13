"""
Train the multi-task NumPy neural network (see nn_from_scratch.py) on DeskPilot data.

Covers, deliberately, the concepts worth being able to discuss in an interview:
  - train/val/test split reused from data_utils (no leakage)
  - vocabulary built from TRAIN ONLY (val/test tokens not seen at vocab-build time
    fall back to <UNK> - this is intentional, mirrors production where new words
    appear after training)
  - class-weighted loss (inverse frequency) for both heads to counter the ~10x
    category imbalance and the priority-label skew
  - mini-batch training, shuffled every epoch
  - a step learning-rate decay schedule
  - early stopping on validation macro-F1 (avg of both heads), patience=6
  - checkpointing the best-epoch weights (not just the final epoch's)
  - loss/F1 curves saved for the README
"""
import json
import time

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, f1_score

from data_utils import get_category_classes, load_data, stratified_split, LabelEncoderFixed, PRIORITY_CLASSES
from nn_from_scratch import MultiTaskTicketNet, Adam

np.random.seed(42)  # seeds the epoch-shuffle permutation below; dropout uses its own
                     # seeded Generator inside nn_from_scratch.py - both sources of
                     # randomness are fixed so training runs are exactly reproducible

import os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
MAX_LEN = 24
PAD, UNK = "<PAD>", "<UNK>"


def build_vocab(texts, max_vocab=4000, min_freq=2):
    freq = {}
    for t in texts:
        for w in t.split():
            freq[w] = freq.get(w, 0) + 1
    words = [w for w, c in sorted(freq.items(), key=lambda x: -x[1]) if c >= min_freq][:max_vocab - 2]
    vocab = {PAD: 0, UNK: 1}
    for w in words:
        vocab[w] = len(vocab)
    return vocab


def tokenize_batch(texts, vocab, max_len=MAX_LEN):
    out = np.zeros((len(texts), max_len), dtype=np.int64)
    for i, t in enumerate(texts):
        ids = [vocab.get(w, vocab[UNK]) for w in t.split()[:max_len]]
        out[i, :len(ids)] = ids
    return out


def build_tab_features(df, dept_list, role_list):
    dept_idx = {d: i for i, d in enumerate(dept_list)}
    role_idx = {r: i for i, r in enumerate(role_list)}
    n = len(df)
    feats = np.zeros((n, len(dept_list) + len(role_list)), dtype=np.float32)
    for i, (d, r) in enumerate(zip(df["department"], df["requester_role"])):
        if d in dept_idx:
            feats[i, dept_idx[d]] = 1.0
        if r in role_idx:
            feats[i, len(dept_list) + role_idx[r]] = 1.0
    return feats


def class_weights_inverse_freq(y, n_classes):
    counts = np.bincount(y, minlength=n_classes).astype(np.float32)
    counts = np.clip(counts, 1, None)
    w = counts.sum() / (n_classes * counts)  # sklearn 'balanced' formula
    return w


def evaluate(net, token_ids, tab_feats, y_cat, y_pri, cat_classes):
    probs_cat, probs_pri = net.predict_proba(token_ids, tab_feats)
    pred_cat, pred_pri = probs_cat.argmax(1), probs_pri.argmax(1)
    f1_cat = f1_score(y_cat, pred_cat, average="macro")
    f1_pri = f1_score(y_pri, pred_pri, average="macro")
    acc_cat = (pred_cat == y_cat).mean()
    acc_pri = (pred_pri == y_pri).mean()
    return dict(f1_cat=f1_cat, f1_pri=f1_pri, acc_cat=acc_cat, acc_pri=acc_pri,
                pred_cat=pred_cat, pred_pri=pred_pri, probs_cat=probs_cat, probs_pri=probs_pri)


def main():
    df = load_data()
    cat_classes = get_category_classes()
    train, val, test = stratified_split(df)

    cat_enc = LabelEncoderFixed(cat_classes)
    pri_enc = LabelEncoderFixed(PRIORITY_CLASSES)
    dept_list = sorted(df["department"].unique())
    role_list = ["Associate", "Senior Associate", "Team Lead", "Manager", "Senior Manager", "Director", "VP"]

    vocab = build_vocab(train["clean_text"], max_vocab=4000, min_freq=2)
    print(f"Vocab size: {len(vocab)}  |  train={len(train)} val={len(val)} test={len(test)}")

    Xtr_tok = tokenize_batch(train["clean_text"], vocab)
    Xval_tok = tokenize_batch(val["clean_text"], vocab)
    Xtest_tok = tokenize_batch(test["clean_text"], vocab)
    Xtr_tab = build_tab_features(train, dept_list, role_list)
    Xval_tab = build_tab_features(val, dept_list, role_list)
    Xtest_tab = build_tab_features(test, dept_list, role_list)

    ytr_cat, yval_cat, ytest_cat = (cat_enc.transform(d["category"]) for d in (train, val, test))
    ytr_pri, yval_pri, ytest_pri = (pri_enc.transform(d["priority"]) for d in (train, val, test))

    cw_cat = class_weights_inverse_freq(ytr_cat, cat_enc.n_classes)
    cw_pri = class_weights_inverse_freq(ytr_pri, pri_enc.n_classes)
    per_sample_w_cat = cw_cat[ytr_cat]
    per_sample_w_pri = cw_pri[ytr_pri]

    net = MultiTaskTicketNet(vocab_size=len(vocab), tab_dim=len(dept_list) + len(role_list),
                              n_category=cat_enc.n_classes, n_priority=pri_enc.n_classes,
                              embed_dim=64, hidden1=128, hidden2=64, dropout=0.3)
    opt = Adam(net.params, lr=2e-3, weight_decay=1e-5)

    n = len(train)
    batch_size = 64
    max_epochs = 60
    patience = 6
    best_score, best_epoch, epochs_no_improve = -1, -1, 0
    best_params = None
    history = []

    print(f"\nTraining multi-task net: {sum(p.size for p in net.params.values()):,} parameters\n")
    t_start = time.time()
    for epoch in range(1, max_epochs + 1):
        if epoch == 25:
            opt.set_lr(opt.lr * 0.5)  # step LR decay
        if epoch == 40:
            opt.set_lr(opt.lr * 0.5)

        idx = np.random.permutation(n)
        epoch_loss = 0.0
        for start in range(0, n, batch_size):
            b = idx[start:start + batch_size]
            probs_cat, probs_pri, cache = net.forward(Xtr_tok[b], Xtr_tab[b], training=True)
            eps = 1e-9
            loss_cat = -np.mean(per_sample_w_cat[b] * np.log(probs_cat[np.arange(len(b)), ytr_cat[b]] + eps))
            loss_pri = -np.mean(per_sample_w_pri[b] * np.log(probs_pri[np.arange(len(b)), ytr_pri[b]] + eps))
            loss = 0.6 * loss_cat + 0.4 * loss_pri
            epoch_loss += loss * len(b)

            grads = net.backward(cache, ytr_cat[b], ytr_pri[b], per_sample_w_cat[b], per_sample_w_pri[b],
                                  loss_weight_cat=0.6, loss_weight_pri=0.4)
            opt.step(net.params, grads)

        epoch_loss /= n
        val_metrics = evaluate(net, Xval_tok, Xval_tab, yval_cat, yval_pri, cat_classes)
        val_score = 0.6 * val_metrics["f1_cat"] + 0.4 * val_metrics["f1_pri"]
        history.append(dict(epoch=epoch, train_loss=round(float(epoch_loss), 4),
                             val_f1_cat=round(val_metrics["f1_cat"], 4),
                             val_f1_pri=round(val_metrics["f1_pri"], 4),
                             val_acc_cat=round(val_metrics["acc_cat"], 4),
                             val_acc_pri=round(val_metrics["acc_pri"], 4),
                             lr=round(opt.lr, 5)))
        print(f"epoch {epoch:2d}/{max_epochs}  loss={epoch_loss:.4f}  "
              f"val_f1_cat={val_metrics['f1_cat']:.4f}  val_f1_pri={val_metrics['f1_pri']:.4f}  "
              f"val_acc_cat={val_metrics['acc_cat']:.4f}  lr={opt.lr:.5f}")

        if val_score > best_score:
            best_score, best_epoch = val_score, epoch
            best_params = {k: v.copy() for k, v in net.params.items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"\nEarly stopping at epoch {epoch} (no val improvement for {patience} epochs). "
                      f"Best epoch = {best_epoch}")
                break

    train_time = time.time() - t_start
    net.params = best_params  # restore best checkpoint, not the final (possibly overfit) epoch

    test_metrics = evaluate(net, Xtest_tok, Xtest_tab, ytest_cat, ytest_pri, cat_classes)
    print(f"\n=== TEST SET (best epoch {best_epoch}) ===")
    print(f"category: acc={test_metrics['acc_cat']:.4f}  macro_f1={test_metrics['f1_cat']:.4f}")
    print(f"priority: acc={test_metrics['acc_pri']:.4f}  macro_f1={test_metrics['f1_pri']:.4f}")

    cat_report = classification_report(ytest_cat, test_metrics["pred_cat"],
                                        target_names=cat_classes, output_dict=True, zero_division=0)
    pri_report = classification_report(ytest_pri, test_metrics["pred_pri"],
                                        target_names=PRIORITY_CLASSES, output_dict=True, zero_division=0)

    # save artifacts
    np.savez(f"{MODELS_DIR}/dl_multitask_weights.npz", **net.params)
    with open(f"{MODELS_DIR}/dl_vocab.json", "w") as f:
        json.dump(vocab, f)
    with open(f"{MODELS_DIR}/dl_meta.json", "w") as f:
        json.dump(dict(dept_list=dept_list, role_list=role_list, cat_classes=cat_classes,
                        priority_classes=PRIORITY_CLASSES, max_len=MAX_LEN,
                        embed_dim=64, hidden1=128, hidden2=64,
                        best_epoch=best_epoch, total_epochs_run=len(history),
                        train_seconds=round(train_time, 1),
                        n_params=int(sum(p.size for p in net.params.values()))), f, indent=2)
    with open(f"{MODELS_DIR}/dl_history.json", "w") as f:
        json.dump(history, f, indent=2)
    with open(f"{MODELS_DIR}/dl_results.json", "w") as f:
        json.dump(dict(
            best_epoch=best_epoch, total_epochs_run=len(history), train_seconds=round(train_time, 1),
            test_category={"accuracy": round(test_metrics["acc_cat"], 4),
                            "macro_f1": round(cat_report["macro avg"]["f1-score"], 4),
                            "weighted_f1": round(cat_report["weighted avg"]["f1-score"], 4)},
            test_priority={"accuracy": round(test_metrics["acc_pri"], 4),
                            "macro_f1": round(pri_report["macro avg"]["f1-score"], 4),
                            "weighted_f1": round(pri_report["weighted avg"]["f1-score"], 4)},
        ), f, indent=2)

    print(f"\nTraining took {train_time:.1f}s across {len(history)} epochs "
          f"(best checkpoint: epoch {best_epoch})")
    print("Saved: dl_multitask_weights.npz, dl_vocab.json, dl_meta.json, dl_history.json, dl_results.json")


if __name__ == "__main__":
    main()
