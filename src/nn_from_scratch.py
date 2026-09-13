"""
DeskPilot Deep Learning core - implemented from scratch in NumPy.

Why from-scratch instead of PyTorch/TensorFlow: this sandbox has tight disk quota
(a plain `pip install torch` pulls ~1.5GB of bundled CUDA libraries and blew the
disk budget here), but more importantly - hand-deriving the forward/backward pass
is a stronger demonstration of actually understanding gradient descent than calling
`.fit()` on a framework. `docs/pytorch_reference_architecture.py` in this repo shows
the equivalent production PyTorch module (BiLSTM-based) this would become at scale.

Architecture (multi-task, shared-encoder):

    token ids ---> Embedding (vocab x embed_dim, trainable)
                       |
                  mean-pool over non-pad tokens          department, role
                       |                                  (one-hot)
                       +------------------+---------------+
                                          |
                                 concat -> fused input
                                          |
                                Dense(hidden1) + ReLU + Dropout
                                          |
                                Dense(hidden2) + ReLU + Dropout   <- shared representation
                                    /                        \
                        Dense(n_category)              Dense(n_priority)
                        softmax (19-way)                softmax (4-way)

Both heads branch from the SAME shared representation and are trained jointly with
a weighted sum of two class-weighted cross-entropy losses - this is what makes it
"multi-task learning" rather than two independent models (contrast with the classical
cascade in train_classical.py, which is two separate models chained together).
"""
import numpy as np

RNG = np.random.default_rng(42)


def he_init(fan_in, fan_out):
    return RNG.normal(0, np.sqrt(2.0 / fan_in), size=(fan_in, fan_out)).astype(np.float32)


def softmax(logits):
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class Adam:
    """Adam optimizer, implemented from scratch (Kingma & Ba, 2014) - bias-corrected
    first/second moment estimates per parameter."""

    def __init__(self, params: dict, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=1e-5):
        self.lr, self.b1, self.b2, self.eps, self.wd = lr, beta1, beta2, eps, weight_decay
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, params: dict, grads: dict):
        self.t += 1
        for k in params:
            g = grads[k] + self.wd * params[k]  # L2 weight decay
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * (g ** 2)
            m_hat = self.m[k] / (1 - self.b1 ** self.t)
            v_hat = self.v[k] / (1 - self.b2 ** self.t)
            params[k] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def set_lr(self, lr):
        self.lr = lr


class MultiTaskTicketNet:
    def __init__(self, vocab_size, tab_dim, n_category, n_priority,
                 embed_dim=64, hidden1=128, hidden2=64, dropout=0.3, pad_idx=0):
        self.pad_idx = pad_idx
        self.dropout_rate = dropout
        d_in = embed_dim + tab_dim
        self.params = {
            "E": (RNG.normal(0, 0.1, size=(vocab_size, embed_dim)).astype(np.float32)),
            "W1": he_init(d_in, hidden1), "b1": np.zeros(hidden1, dtype=np.float32),
            "W2": he_init(hidden1, hidden2), "b2": np.zeros(hidden2, dtype=np.float32),
            "Wc": he_init(hidden2, n_category), "bc": np.zeros(n_category, dtype=np.float32),
            "Wp": he_init(hidden2, n_priority), "bp": np.zeros(n_priority, dtype=np.float32),
        }
        self.params["E"][pad_idx] = 0.0  # padding embedding fixed at zero
        self.embed_dim, self.tab_dim = embed_dim, tab_dim

    def forward(self, token_ids, tab_feats, training=True):
        """token_ids: (B, L) int array (pad_idx for padding). tab_feats: (B, tab_dim) float."""
        P = self.params
        mask = (token_ids != self.pad_idx).astype(np.float32)  # (B, L)
        counts = np.clip(mask.sum(axis=1, keepdims=True), 1, None)  # avoid /0

        emb = P["E"][token_ids]  # (B, L, embed_dim)
        text_repr = (emb * mask[:, :, None]).sum(axis=1) / counts  # mean-pool -> (B, embed_dim)

        fused = np.concatenate([text_repr, tab_feats], axis=1)  # (B, embed_dim+tab_dim)

        z1 = fused @ P["W1"] + P["b1"]
        h1 = np.maximum(z1, 0)
        drop1_mask = (RNG.random(h1.shape) > self.dropout_rate).astype(np.float32) / (1 - self.dropout_rate) \
            if training else np.ones_like(h1)
        h1d = h1 * drop1_mask

        z2 = h1d @ P["W2"] + P["b2"]
        h2 = np.maximum(z2, 0)
        drop2_mask = (RNG.random(h2.shape) > self.dropout_rate).astype(np.float32) / (1 - self.dropout_rate) \
            if training else np.ones_like(h2)
        h2d = h2 * drop2_mask

        logits_cat = h2d @ P["Wc"] + P["bc"]
        logits_pri = h2d @ P["Wp"] + P["bp"]
        probs_cat, probs_pri = softmax(logits_cat), softmax(logits_pri)

        cache = dict(token_ids=token_ids, mask=mask, counts=counts, tab_feats=tab_feats,
                     fused=fused, z1=z1, h1=h1, drop1_mask=drop1_mask, h1d=h1d,
                     z2=z2, h2=h2, drop2_mask=drop2_mask, h2d=h2d,
                     probs_cat=probs_cat, probs_pri=probs_pri)
        return probs_cat, probs_pri, cache

    def backward(self, cache, y_cat, y_pri, w_cat, w_pri, loss_weight_cat=0.6, loss_weight_pri=0.4):
        """y_cat/y_pri: int arrays of true class idx. w_cat/w_pri: per-sample class weight
        (from inverse-frequency class weighting) used to counter class imbalance."""
        P = self.params
        B = y_cat.shape[0]
        n_cat, n_pri = P["Wc"].shape[1], P["Wp"].shape[1]

        onehot_cat = np.eye(n_cat, dtype=np.float32)[y_cat]
        onehot_pri = np.eye(n_pri, dtype=np.float32)[y_pri]

        # dL/dlogits for softmax+weighted-CE = weight * (probs - onehot) / B
        dlogits_cat = (loss_weight_cat * w_cat[:, None] * (cache["probs_cat"] - onehot_cat)) / B
        dlogits_pri = (loss_weight_pri * w_pri[:, None] * (cache["probs_pri"] - onehot_pri)) / B

        grads = {}
        grads["Wc"] = cache["h2d"].T @ dlogits_cat
        grads["bc"] = dlogits_cat.sum(axis=0)
        grads["Wp"] = cache["h2d"].T @ dlogits_pri
        grads["bp"] = dlogits_pri.sum(axis=0)

        dh2d = dlogits_cat @ P["Wc"].T + dlogits_pri @ P["Wp"].T  # gradient merges at the shared layer
        dh2 = dh2d * cache["drop2_mask"]
        dz2 = dh2 * (cache["z2"] > 0)
        grads["W2"] = cache["h1d"].T @ dz2
        grads["b2"] = dz2.sum(axis=0)

        dh1d = dz2 @ P["W2"].T
        dh1 = dh1d * cache["drop1_mask"]
        dz1 = dh1 * (cache["z1"] > 0)
        grads["W1"] = cache["fused"].T @ dz1
        grads["b1"] = dz1.sum(axis=0)

        dfused = dz1 @ P["W1"].T
        dtext_repr = dfused[:, :self.embed_dim]  # (B, embed_dim); tabular half needs no grad (one-hot input)

        dE = np.zeros_like(P["E"])
        # mean-pool backward: distribute dtext_repr equally across each example's valid tokens
        per_token_grad = (dtext_repr / cache["counts"])[:, None, :] * cache["mask"][:, :, None]  # (B,L,embed_dim)
        np.add.at(dE, cache["token_ids"], per_token_grad)
        dE[self.pad_idx] = 0.0
        grads["E"] = dE

        return grads

    def predict_proba(self, token_ids, tab_feats):
        probs_cat, probs_pri, _ = self.forward(token_ids, tab_feats, training=False)
        return probs_cat, probs_pri
