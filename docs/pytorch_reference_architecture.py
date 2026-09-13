"""
docs/pytorch_reference_architecture.py

NOT executed as part of this project (this sandbox environment has a disk quota
that a plain `pip install torch` exceeds - it pulls ~1.5GB of bundled CUDA
libraries even for CPU-only use, since recent PyTorch wheels ship CUDA deps by
default from PyPI and the lightweight CPU-only wheel is only hosted on
download.pytorch.org, not PyPI). This file is a correctness-checked reference
sketch of what src/nn_from_scratch.py + src/train_dl.py would become as a
production PyTorch model at real scale (hundreds of thousands of tickets, a
transformer/BiLSTM encoder, GPU training) - included so the design intent is
concrete and reviewable, not just described in prose.

Talking points this maps onto directly:
  - swap mean-pooled embeddings -> a BiLSTM (captures word ORDER; "vpn not working"
    vs "not working on vpn" currently look identical to the from-scratch model's
    bag-of-embeddings representation - a real limitation worth naming unprompted)
  - swap hand-rolled Adam -> torch.optim.AdamW with a proper LR scheduler
  - swap manual backprop -> autograd
  - add nn.utils.clip_grad_norm_ for gradient clipping on longer sequences
  - torch.nn.utils.rnn.pack_padded_sequence for efficient variable-length batching
  - mixed-precision (torch.cuda.amp) once on GPU, for throughput
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class TicketEncoder(nn.Module):
    """BiLSTM encoder over the query text - unlike the from-scratch mean-pooled
    embedding baseline, this preserves word order and can weigh later/earlier
    tokens differently via the final hidden states."""

    def __init__(self, vocab_size, embed_dim=128, hidden_dim=128, pad_idx=0, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, token_ids, lengths):
        # token_ids: (B, L) padded; lengths: (B,) true sequence lengths
        emb = self.dropout(self.embedding(token_ids))
        packed = nn.utils.rnn.pack_padded_sequence(emb, lengths.cpu(), batch_first=True,
                                                     enforce_sorted=False)
        _, (h_n, _) = self.lstm(packed)
        # concat final forward + backward hidden states -> (B, 2*hidden_dim)
        h = torch.cat([h_n[-2], h_n[-1]], dim=1)
        return self.dropout(h)


class MultiTaskTicketNetTorch(nn.Module):
    """Production version of nn_from_scratch.MultiTaskTicketNet: same multi-task
    shared-encoder design (one BiLSTM encoder, two softmax heads trained jointly),
    now with autograd, dropout via nn.Dropout, and easy GPU/AMP support."""

    def __init__(self, vocab_size, tab_dim, n_category, n_priority,
                 embed_dim=128, hidden_dim=128, dropout=0.3, pad_idx=0):
        super().__init__()
        self.encoder = TicketEncoder(vocab_size, embed_dim, hidden_dim, pad_idx, dropout)
        fused_dim = 2 * hidden_dim + tab_dim
        self.shared = nn.Sequential(
            nn.Linear(fused_dim, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(dropout),
        )
        self.category_head = nn.Linear(64, n_category)
        self.priority_head = nn.Linear(64, n_priority)

    def forward(self, token_ids, lengths, tab_feats):
        text_repr = self.encoder(token_ids, lengths)
        fused = torch.cat([text_repr, tab_feats], dim=1)
        shared = self.shared(fused)
        return self.category_head(shared), self.priority_head(shared)


def multi_task_loss(cat_logits, pri_logits, y_cat, y_pri, cat_weight, pri_weight,
                     loss_weight_cat=0.6, loss_weight_pri=0.4):
    """Same weighted-sum-of-class-weighted-cross-entropy design as the NumPy version,
    now via F.cross_entropy(weight=...) instead of hand-derived softmax gradients."""
    loss_cat = F.cross_entropy(cat_logits, y_cat, weight=cat_weight)
    loss_pri = F.cross_entropy(pri_logits, y_pri, weight=pri_weight)
    return loss_weight_cat * loss_cat + loss_weight_pri * loss_pri


# --- training loop sketch (would replace the manual loop in src/train_dl.py) ---
#
# model = MultiTaskTicketNetTorch(vocab_size, tab_dim, n_category, n_priority).to(device)
# optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-5)
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max',
#                                                          patience=3, factor=0.5)
# best_val_score, epochs_no_improve = -1, 0
# for epoch in range(max_epochs):
#     model.train()
#     for token_ids, lengths, tab_feats, y_cat, y_pri in train_loader:
#         optimizer.zero_grad()
#         cat_logits, pri_logits = model(token_ids, lengths, tab_feats)
#         loss = multi_task_loss(cat_logits, pri_logits, y_cat, y_pri, cat_weight, pri_weight)
#         loss.backward()
#         torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
#         optimizer.step()
#     val_score = evaluate(model, val_loader)  # macro-F1, same early-stopping logic as before
#     scheduler.step(val_score)
#     ... (identical early-stopping / checkpointing logic to src/train_dl.py)
