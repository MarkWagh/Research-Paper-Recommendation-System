"""Common recommender interface.

Life cycle used by every script:
    model = build_model(name, params)
    model.fit(train_graph, features, split)    # learn anything trainable (training graph only)
    model.prepare(eval_graph, "val")           # split-specific, time-valid graph statistics
    scores = model.score_batch(queryset, rows) # (B, N) dense scores; masking is done by the evaluator
    top_idx, top_scores = model.score(queryset, k=50, evaluator=ev)
``set_params`` changes query-time hyper-parameters (e.g. the own/history mixing weight)
without refitting, which is what validation tuning uses.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch

from ..data import QuerySet, Split
from ..graph import CitationGraph


class Recommender:
    name = "base"
    defaults: dict = {}

    def __init__(self, **params):
        unknown = set(params) - set(self.defaults) - {"seed", "device"}
        if unknown:
            raise ValueError(f"{self.name}: unknown params {sorted(unknown)}")
        self.params = {**self.defaults, **params}
        self.graph: CitationGraph | None = None

    def set_params(self, **kw) -> "Recommender":
        self.params.update(kw)
        return self

    def fit(self, train_graph: CitationGraph, features, split: Split) -> "Recommender":
        self.features = features
        return self

    def prepare(self, graph: CitationGraph, which: str) -> "Recommender":
        self.graph = graph
        return self

    def score_batch(self, qs: QuerySet, rows: np.ndarray) -> torch.Tensor:
        raise NotImplementedError

    def score(self, qs: QuerySet, k: int = 50, evaluator=None, rows: np.ndarray | None = None):
        """Top-k eligible candidates (indices, scores) per query, using the evaluator's mask."""
        rows = np.arange(len(qs)) if rows is None else rows
        idx, val = [], []
        for s in range(0, len(rows), evaluator.batch_size):
            r = rows[s:s + evaluator.batch_size]
            sc = self.score_batch(qs, r)
            i, v = evaluator.topk(sc, evaluator.candidate_mask(qs, r), k)
            idx.append(i)
            val.append(v)
        return np.concatenate(idx), np.concatenate(val)

    def __call__(self, qs: QuerySet):
        """Bind a query set so the model can be passed to Evaluator.evaluate as a score fn."""
        return lambda rows: self.score_batch(qs, rows)


# ------------------------------------------------------------- helpers shared by models
def history_matrix(qs: QuerySet, rows: np.ndarray, n: int, normalize: bool = True) -> sp.csr_matrix:
    """(B, N) sparse; row b has weight 1/|H_b| (or 1) on each history item."""
    H = qs.history_matrix(n, rows)
    if normalize:
        cnt = np.maximum(np.diff(H.indptr), 1)
        H = sp.diags(1.0 / cnt) @ H
    return H.tocsr().astype(np.float32)


def history_mean(E: torch.Tensor, qs: QuerySet, rows: np.ndarray) -> torch.Tensor:
    """(B, D) mean of the history rows of embedding matrix E, computed on E's device."""
    sizes = np.diff(qs.hist_ptr)[rows]
    idx = np.concatenate([qs.history(i) for i in rows])
    seg = torch.as_tensor(np.repeat(np.arange(len(rows)), sizes), device=E.device)
    out = torch.zeros(len(rows), E.shape[1], device=E.device, dtype=E.dtype)
    out.index_add_(0, seg, E[torch.as_tensor(idx, device=E.device)])
    return out / torch.as_tensor(np.maximum(sizes, 1), device=E.device, dtype=E.dtype).unsqueeze(1)


def minmax_rows(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    lo = x.min(1, keepdim=True).values
    hi = x.max(1, keepdim=True).values
    return (x - lo) / (hi - lo).clamp(min=eps)


def to_dense_tensor(m) -> torch.Tensor:
    if sp.issparse(m):
        m = m.toarray()
    return torch.as_tensor(np.asarray(m, dtype=np.float32))
