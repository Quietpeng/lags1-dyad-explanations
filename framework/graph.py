"""Hand-written one-layer GraphSAGE-style encoder + dyad head.

No PyG, no DGL, no torch.  The encoder is

    H_m = ReLU( X_m W_self + (P_m X_m) W_neigh + b )

where ``P_m`` is the row-normalised undirected actor graph at month ``m``
(self-loops included), and the dyad head is

    logit_ij = c + w_d^T x_ij + w_s^T H_i + w_t^T H_j + w_x^T (H_i * H_j).

The analytic backward pass is implemented below and checked against finite
differences in ``tests/test_graph_grad.py``.  The model is transductive over the
196 actors; the point of P1 is a transparent, reproducible encoder, not a
state-of-the-art GNN.

Neighbour ablation and edge counterfactuals are implemented by rebuilding the
one-month propagation matrix and re-running the encoder, holding all parameters
fixed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .metrics import binary_metrics


def _sigmoid(z):
    z = np.clip(z, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-z))


@dataclass
class SageConfig:
    dim: int = 8
    epochs: int = 10
    lr: float = 0.01
    l2: float = 1e-4
    max_rows_per_month: int = 4000
    seed: int = 0
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    grad_clip: float = 5.0
    verbose: bool = True


def row_normalise(B: np.ndarray) -> np.ndarray:
    """B [A,A] binary -> row-normalised with self-loops."""
    A = B.shape[0]
    P = B.astype(np.float32) + np.eye(A, dtype=np.float32)
    deg = P.sum(axis=1, keepdims=True)
    return P / np.maximum(deg, 1e-12)


def propagate(P: np.ndarray, X: np.ndarray) -> np.ndarray:
    return (P @ X).astype(np.float32)


class SageDyadModel:
    def __init__(self, cfg: SageConfig, n_actor_features: int, n_dyad_features: int):
        self.cfg = cfg
        self.F = int(n_actor_features)
        self.D = int(n_dyad_features)
        self.d = int(cfg.dim)
        rng = np.random.default_rng(cfg.seed)
        scale = np.sqrt(2.0 / max(self.F, 1))
        self.W_self = (rng.normal(0, scale, (self.F, self.d))).astype(np.float32)
        self.W_neigh = (rng.normal(0, scale, (self.F, self.d))).astype(np.float32)
        self.b = np.zeros(self.d, dtype=np.float32)
        H = self.D + 3 * self.d
        self.w = (rng.normal(0, 1.0 / np.sqrt(max(H, 1)), H)).astype(np.float32)
        self.c = np.float32(-2.1)  # logit of the ~11% train active rate
        self.grad_clip = float(getattr(cfg, "grad_clip", 5.0))
        self._m = [np.zeros_like(p) for p in (self.W_self, self.W_neigh, self.b, self.w)]
        self._v = [np.zeros_like(p) for p in (self.W_self, self.W_neigh, self.b, self.w)]
        self._t = 0

    # ------------------------------------------------------------------
    # forward / backward primitives
    # ------------------------------------------------------------------
    def forward_month(self, X: np.ndarray, PX: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        pre = X @ self.W_self + PX @ self.W_neigh + self.b
        H = np.maximum(pre, 0.0).astype(np.float32)
        return H, pre

    def _head(self, Xd, Hs, Ht):
        Z = np.concatenate([Xd, Hs, Ht, Hs * Ht], axis=1)
        return Z, Z @ self.w + self.c

    def _step_month(self, X, PX, Xd, sid, tid, y, l2: float) -> float:
        n = len(y)
        H, pre = self.forward_month(X, PX)
        Hs, Ht = H[sid], H[tid]
        Z, logit = self._head(Xd, Hs, Ht)
        p = _sigmoid(logit.astype(np.float64))
        eps = 1e-9
        loss = float(-np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps)))
        g_logit = ((p - y) / n).astype(np.float32)

        g_w = Z.T @ g_logit + l2 * self.w
        g_c = np.float32(g_logit.sum())
        dZ = g_logit[:, None] * self.w[None, :]
        dHs = dZ[:, self.D:self.D + self.d] + dZ[:, self.D + 2 * self.d:] * Ht
        dHt = dZ[:, self.D + self.d:self.D + 2 * self.d] + dZ[:, self.D + 2 * self.d:] * Hs
        dH = np.zeros_like(H)
        np.add.at(dH, sid, dHs)
        np.add.at(dH, tid, dHt)
        dpre = dH * (pre > 0).astype(np.float32)
        g_Ws = X.T @ dpre + l2 * self.W_self
        g_Wn = PX.T @ dpre + l2 * self.W_neigh
        g_b = dpre.sum(axis=0)
        if self.grad_clip > 0:
            for g in (g_Ws, g_Wn, g_b, g_w):
                n = float(np.linalg.norm(g))
                if n > self.grad_clip:
                    g *= self.grad_clip / n

        # Adam on the four parameter blocks; c is a scalar with its own update.
        self._t += 1
        b1, b2 = self.cfg.beta1, self.cfg.beta2
        blocks = [(self.W_self, g_Ws), (self.W_neigh, g_Wn),
                  (self.b, g_b), (self.w, g_w)]
        for i, (par, g) in enumerate(blocks):
            self._m[i] = b1 * self._m[i] + (1 - b1) * g
            self._v[i] = b2 * self._v[i] + (1 - b2) * (g * g)
            mhat = self._m[i] / (1 - b1 ** self._t)
            vhat = self._v[i] / (1 - b2 ** self._t)
            par -= self.cfg.lr * mhat / (np.sqrt(vhat) + self.cfg.eps)
        self.c = np.float32(self.c - self.cfg.lr * g_c)
        return loss

    # ------------------------------------------------------------------
    # fit / predict
    # ------------------------------------------------------------------
    def fit(self, tensors: dict, train_mask: np.ndarray, select_mask: np.ndarray,
            dyad_X: np.ndarray, sid: np.ndarray, tid: np.ndarray,
            y: np.ndarray, t_idx: np.ndarray, select_y: np.ndarray) -> dict:
        cfg = self.cfg
        X, PX = tensors["X"], tensors["PX"]
        M = X.shape[0]
        rng = np.random.default_rng(cfg.seed + 17)
        rows_by_month = _rows_by_month(t_idx, M)
        tr_by_month = [r[np.asarray(train_mask)[r]] for r in rows_by_month]
        sel_by_month = [r[np.asarray(select_mask)[r]] for r in rows_by_month]
        best = None
        history = []
        for epoch in range(cfg.epochs):
            order = rng.permutation(M)
            losses = []
            for m in order:
                rows = tr_by_month[m]
                if len(rows) == 0:
                    continue
                if len(rows) > cfg.max_rows_per_month:
                    pos = rows[y[rows] == 1]
                    neg = rows[y[rows] == 0]
                    k = max(cfg.max_rows_per_month - len(pos), 1)
                    if len(neg) > k:
                        neg = rng.choice(neg, size=k, replace=False)
                    rows = np.concatenate([pos, neg])
                losses.append(self._step_month(
                    X[m], PX[m], dyad_X[rows], sid[rows], tid[rows],
                    y[rows].astype(np.float32), cfg.l2))
            p_sel = self.predict(dyad_X, sid, tid, t_idx, select_mask,
                                 tensors=tensors)
            m = binary_metrics(select_y, p_sel)
            history.append({"epoch": epoch + 1, "train_loss": round(float(np.mean(losses)), 6),
                            "select_log_loss": m["log_loss"], "select_auc": m["auc"]})
            if cfg.verbose:
                print(f"  sage epoch {epoch + 1}/{cfg.epochs} "
                      f"train={np.mean(losses):.4f} select_ll={m['log_loss']:.4f}")
            if best is None or m["log_loss"] < best[0]:
                best = (m["log_loss"], epoch + 1, self._snapshot())
        if best is not None:
            self._restore(best[2])
        return {"best_epoch": best[1] if best else None,
                "best_select_log_loss": best[0] if best else None,
                "history": history}

    def predict(self, dyad_X, sid, tid, t_idx, mask=None,
                tensors: dict | None = None) -> np.ndarray:
        X, PX = tensors["X"], tensors["PX"]
        n = len(t_idx)
        out = np.zeros(n, dtype=np.float64)
        m_arr = np.asarray(t_idx)
        for m in np.unique(m_arr):
            rows = np.where(m_arr == m)[0]
            if mask is not None:
                rows = rows[np.asarray(mask)[rows]]
            if len(rows) == 0:
                continue
            H, _ = self.forward_month(X[m], PX[m])
            _, logit = self._head(dyad_X[rows], H[sid[rows]], H[tid[rows]])
            out[rows] = _sigmoid(logit.astype(np.float64))
        if mask is not None:
            return out[np.asarray(mask)]
        return out

    def predict_all(self, tensors: dict, dyad_X, sid, tid, t_idx) -> np.ndarray:
        out = np.zeros(len(t_idx), dtype=np.float64)
        m_arr = np.asarray(t_idx)
        for m in range(tensors["X"].shape[0]):
            rows = np.where(m_arr == m)[0]
            if len(rows) == 0:
                continue
            H, _ = self.forward_month(tensors["X"][m], tensors["PX"][m])
            _, logit = self._head(dyad_X[rows], H[sid[rows]], H[tid[rows]])
            out[rows] = _sigmoid(logit.astype(np.float64))
        return out

    # ------------------------------------------------------------------
    # interpretability hooks
    # ------------------------------------------------------------------
    def head_decomposition(self, dyad_X, Hs, Ht, names: list[str]) -> dict:
        D, d = self.D, self.d
        zd = dyad_X @ self.w[:D]
        zs = Hs @ self.w[D:D + d]
        zt = Ht @ self.w[D + d:D + 2 * d]
        zx = (Hs * Ht) @ self.w[D + 2 * d:]
        return {
            "intercept": float(self.c),
            "dyad_features": float(zd), "self_embedding": float(zs),
            "target_embedding": float(zt), "interaction": float(zx),
            "dyad_terms": sorted(
                [{"feature": n, "contribution": round(float(v), 6)}
                 for n, v in zip(names, dyad_X * self.w[:D])],
                key=lambda r: -abs(r["contribution"]))[:8],
        }

    def _predict_rows(self, tensors: dict, dyad_X, sid, tid, t_idx,
                      rows: np.ndarray) -> np.ndarray:
        rows = np.asarray(rows)
        out = np.zeros(len(rows), dtype=np.float64)
        m_arr = np.asarray(t_idx)
        for m in np.unique(m_arr[rows]):
            sel = np.where(m_arr[rows] == m)[0]
            r = rows[sel]
            H, _ = self.forward_month(tensors["X"][m], tensors["PX"][m])
            _, logit = self._head(dyad_X[r], H[sid[r]], H[tid[r]])
            out[sel] = _sigmoid(logit.astype(np.float64))
        return out

    def neighbour_ablation(self, tensors: dict, month: int, focal: int,
                           dyad_X, sid, tid, t_idx, y) -> dict:
        rows = np.where((t_idx == month) & ((sid == focal) | (tid == focal)))[0]
        if len(rows) == 0:
            return {"rows": 0}
        p0 = self._predict_rows(tensors, dyad_X, sid, tid, t_idx, rows)
        B = tensors["B"][month].copy()
        B[focal, :] = 0
        B[:, focal] = 0
        P = row_normalise(B)
        H = self.forward_month(tensors["X"][month], propagate(P, tensors["X"][month]))[0]
        _, logit = self._head(dyad_X[rows], H[sid[rows]], H[tid[rows]])
        p1 = _sigmoid(logit.astype(np.float64))
        return {
            "focal_aid": int(focal), "month": int(month), "rows": int(len(rows)),
            "mean_p_before": round(float(p0.mean()), 6),
            "mean_p_after": round(float(p1.mean()), 6),
            "mean_delta_p": round(float((p1 - p0).mean()), 6),
            "mean_abs_delta_p": round(float(np.abs(p1 - p0).mean()), 6),
        }

    def edge_counterfactual(self, tensors: dict, month: int, i: int, j: int,
                            dyad_X, sid, tid, t_idx, mode: str = "flip") -> dict:
        rows = np.where((t_idx == month) & ((sid == i) | (tid == i)
                                            | (sid == j) | (tid == j)))[0]
        if len(rows) == 0:
            return {"rows": 0}
        p0 = self._predict_rows(tensors, dyad_X, sid, tid, t_idx, rows)
        B = tensors["B"][month].copy()
        if mode == "flip":
            B[i, j] = B[j, i] = 1 - B[i, j]
        elif mode == "remove":
            B[i, j] = B[j, i] = 0
        elif mode == "add":
            B[i, j] = B[j, i] = 1
        else:
            raise ValueError(mode)
        P = row_normalise(B)
        H = self.forward_month(tensors["X"][month], propagate(P, tensors["X"][month]))[0]
        _, logit = self._head(dyad_X[rows], H[sid[rows]], H[tid[rows]])
        p1 = _sigmoid(logit.astype(np.float64))
        return {
            "i": int(i), "j": int(j), "month": int(month), "mode": mode,
            "rows": int(len(rows)),
            "mean_p_before": round(float(p0.mean()), 6),
            "mean_p_after": round(float(p1.mean()), 6),
            "mean_delta_p": round(float((p1 - p0).mean()), 6),
        }

    def _snapshot(self):
        return {k: getattr(self, k).copy() for k in
                ("W_self", "W_neigh", "b", "w", "c")}

    def _restore(self, snap):
        for k, v in snap.items():
            setattr(self, k, v.copy())


def _rows_by_month(t_idx: np.ndarray, M: int) -> list[np.ndarray]:
    order = np.argsort(t_idx, kind="stable")
    ts = np.asarray(t_idx)[order]
    starts = np.searchsorted(ts, np.arange(M + 1))
    return [order[starts[m]:starts[m + 1]] for m in range(M)]


def build_graph_tensors(actor_d: dict, panel_d: dict, cfg: dict,
                        mask_actors: set[int] | None = None,
                        standardize: bool = True) -> dict:
    """Build monthly actor features X, propagation P and PX.

    ``actor_d`` is the actor split dict (``X`` shaped [rows, F], ``aid``,
    ``t_idx``) for **all** months; ``panel_d`` is the dyad split dict with
    ``sid, tid, t_idx, active``.  Only months present in the panel are used for
    the graph snapshot; actor features are indexed by month.
    """
    names = actor_d["feature_names"]
    A = int(max(actor_d["aid"].max(), panel_d["sid"].max(), panel_d["tid"].max())) + 1
    M = int(max(panel_d["t_idx"].max(), actor_d["t_idx"].max())) + 1
    Xa = np.zeros((M, A, len(names)), dtype=np.float32)
    present = np.zeros((M, A), dtype=bool)
    a_idx = actor_d["aid"].astype(int)
    t_idx = actor_d["t_idx"].astype(int)
    Xa[t_idx, a_idx] = actor_d["X"]
    present[t_idx, a_idx] = True
    # Last month has no label in actor_d, but features exist; fill from the
    # actor table if a caller supplies the full (unfiltered) frame later.
    mu = np.zeros(len(names), dtype=np.float32)
    sd = np.ones(len(names), dtype=np.float32)
    if standardize:
        train_months = np.asarray(actor_d["t_idx"]) < _month_idx(cfg["split"]["train_end"])
        mu = np.asarray(actor_d["X"][train_months].mean(0), dtype=np.float32)
        sd = np.asarray(actor_d["X"][train_months].std(0), dtype=np.float32)
        sd = np.where(sd < 1e-8, 1.0, sd)
        Xa = (Xa - mu) / sd
    Xa[~present] = 0.0

    B = np.zeros((M, A, A), dtype=np.uint8)
    B[panel_d["t_idx"], panel_d["sid"], panel_d["tid"]] = panel_d["active"].astype(np.uint8)
    B = np.maximum(B, B.transpose(0, 2, 1))
    if mask_actors:
        keep = np.ones(A, dtype=bool)
        keep[list(mask_actors)] = False
        B[:, ~keep, :] = 0
        B[:, :, ~keep] = 0
        Xa[:, ~keep, :] = 0.0
    P = np.empty_like(B, dtype=np.float32)
    PX = np.empty((M, A, len(names)), dtype=np.float32)
    for m in range(M):
        P[m] = row_normalise(B[m])
        PX[m] = propagate(P[m], Xa[m])
    return {"X": Xa, "B": B, "P": P, "PX": PX,
            "actor_mu": mu, "actor_sd": sd, "names": names, "A": A, "M": M}


def _month_idx(date_str: str) -> int:
    y, m, _ = date_str.split("-")
    return (int(y) - 1995) * 12 + (int(m) - 1)


# ---------------------------------------------------------------------------
# Stable variant: PCA-projected one-layer propagation + learned logistic head
# ---------------------------------------------------------------------------
class PCAEncoder:
    """Fixed (unsupervised) one-layer GraphSAGE-style propagation.

    ``H_m = ReLU( Xc_m W + alpha * (P_m Xc_m) W )`` where ``W`` is the top-d
    PCA projection of the training-month actor features.  The projection is
    fit on train months only; the downstream dyad head is a standard logistic
    regression.  This keeps the graph channel transparent and stable while the
    end-to-end :class:`SageDyadModel` remains available as a diagnostic.
    """

    def __init__(self, dim: int = 8, alpha: float = 0.5):
        self.dim = int(dim)
        self.alpha = float(alpha)
        self.mu = None
        self.W = None
        self.h_mu = None
        self.h_sd = None

    def fit(self, tensors: dict, months: np.ndarray) -> "PCAEncoder":
        X = tensors["X"]
        flat = np.concatenate([X[m] for m in months], axis=0)
        self.mu = flat.mean(axis=0).astype(np.float32)
        Xc = flat - self.mu
        # economy SVD; rows are actor-months, columns actor features.
        _, _, vt = np.linalg.svd(Xc, full_matrices=False)
        self.W = vt[:self.dim].T.astype(np.float32)
        H = np.concatenate([self._h(X[m], tensors["PX"][m]) for m in months], axis=0)
        self.h_mu = H.mean(axis=0).astype(np.float32)
        self.h_sd = np.where(H.std(axis=0) < 1e-8, 1.0, H.std(axis=0)).astype(np.float32)
        return self

    def _h(self, Xm, PXm):
        Xc = Xm - self.mu
        PXc = PXm - self.mu  # P is row-stochastic, so P(X - mu) = PX - mu
        return np.maximum(Xc @ self.W + self.alpha * (PXc @ self.W), 0.0).astype(np.float32)

    def transform(self, tensors: dict) -> np.ndarray:
        M = tensors["X"].shape[0]
        H = np.empty((M, tensors["X"].shape[1], self.dim), dtype=np.float32)
        for m in range(M):
            h = self._h(tensors["X"][m], tensors["PX"][m])
            H[m] = (h - self.h_mu) / self.h_sd
        return H


class GraphLogisticModel:
    """Logistic dyad head on ``[x_dyad, H_i, H_j, H_i*H_j]``."""

    def __init__(self, cfg: dict, encoder: PCAEncoder):
        self.cfg = cfg
        self.encoder = encoder
        self.head = None
        self.feature_names = None

    def _design(self, H, Xd, sid, tid, t_idx) -> np.ndarray:
        if H.ndim == 3:
            Hs, Ht = H[t_idx, sid], H[t_idx, tid]
        else:
            Hs, Ht = H[sid], H[tid]
        return np.concatenate([Xd, Hs, Ht, Hs * Ht], axis=1).astype(np.float32)

    def fit(self, tensors, H, Xd, sid, tid, t_idx, y, train_mask, select_mask):
        from sklearn.linear_model import LogisticRegression

        rows_tr = np.where(train_mask)[0]
        rows_sel = np.where(select_mask)[0]
        Ztr = self._design(H, Xd, sid, tid, t_idx)[rows_tr]
        Zsel = self._design(H, Xd, sid, tid, t_idx)[rows_sel]
        mu, sd = Ztr.mean(0), Ztr.std(0)
        sd = np.where(sd < 1e-8, 1.0, sd)
        Ztr = (Ztr - mu) / sd
        Zsel = (Zsel - mu) / sd
        grid = self.cfg["models"]["logistic"]["C_grid"]
        best = None
        curve = []
        for C in grid:
            clf = LogisticRegression(C=float(C), max_iter=300, solver="lbfgs",
                                     random_state=int(self.cfg["seed"]))
            clf.fit(Ztr, y[rows_tr])
            ll = binary_metrics(y[rows_sel],
                                clf.predict_proba(Zsel)[:, 1])["log_loss"]
            curve.append({"C": float(C), "select_log_loss": ll})
            if best is None or ll < best[0]:
                best = (ll, float(C), clf)
        self.head = best[2]
        self.C = best[1]
        self.select_log_loss = best[0]
        self.curve = curve
        self.z_mu, self.z_sd = mu, sd
        self.feature_names = (["dyad:" + str(i) for i in range(Xd.shape[1])]
                              + [f"H_s_{i}" for i in range(H.shape[2])]
                              + [f"H_t_{i}" for i in range(H.shape[2])]
                              + [f"H_int_{i}" for i in range(H.shape[2])])
        return self

    def predict(self, H, Xd, sid, tid, t_idx, rows=None) -> np.ndarray:
        if rows is None:
            rows = np.arange(len(Xd))
        Z = self._design(H, Xd, sid, tid, t_idx)[rows]
        Z = (Z - self.z_mu) / self.z_sd
        return self.head.predict_proba(Z)[:, 1]

    def neighbour_ablation(self, tensors, H, month, focal, Xd, sid, tid, t_idx):
        rows = np.where((t_idx == month) & ((sid == focal) | (tid == focal)))[0]
        if len(rows) == 0:
            return {"rows": 0}
        p0 = self.predict(H, Xd, sid, tid, t_idx, rows)
        B = tensors["B"][month].copy()
        B[focal, :] = 0
        B[:, focal] = 0
        P = row_normalise(B)
        h = self.encoder._h(tensors["X"][month], propagate(P, tensors["X"][month]))
        Hm = (h - self.encoder.h_mu) / self.encoder.h_sd
        H2 = H.copy()
        H2[month] = Hm
        p1 = self.predict(H2, Xd[rows], sid[rows], tid[rows], t_idx[rows])
        return {
            "focal_aid": int(focal), "month": int(month), "rows": int(len(rows)),
            "mean_p_before": round(float(p0.mean()), 6),
            "mean_p_after": round(float(p1.mean()), 6),
            "mean_delta_p": round(float((p1 - p0).mean()), 6),
            "mean_abs_delta_p": round(float(np.abs(p1 - p0).mean()), 6),
        }

    def edge_counterfactual(self, tensors, H, month, i, j, Xd, sid, tid, t_idx,
                            mode="flip"):
        rows = np.where((t_idx == month) & ((sid == i) | (tid == i)
                                            | (sid == j) | (tid == j)))[0]
        if len(rows) == 0:
            return {"rows": 0}
        p0 = self.predict(H, Xd, sid, tid, t_idx, rows)
        B = tensors["B"][month].copy()
        if mode == "flip":
            B[i, j] = B[j, i] = 1 - B[i, j]
        elif mode == "remove":
            B[i, j] = B[j, i] = 0
        elif mode == "add":
            B[i, j] = B[j, i] = 1
        P = row_normalise(B)
        h = self.encoder._h(tensors["X"][month], propagate(P, tensors["X"][month]))
        Hm = (h - self.encoder.h_mu) / self.encoder.h_sd
        H2 = H.copy()
        H2[month] = Hm
        p1 = self.predict(H2, Xd[rows], sid[rows], tid[rows], t_idx[rows])
        return {
            "i": int(i), "j": int(j), "month": int(month), "mode": mode,
            "rows": int(len(rows)),
            "mean_p_before": round(float(p0.mean()), 6),
            "mean_p_after": round(float(p1.mean()), 6),
            "mean_delta_p": round(float((p1 - p0).mean()), 6),
        }
