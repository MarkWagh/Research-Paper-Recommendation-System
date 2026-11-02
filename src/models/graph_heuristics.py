"""Graph-only heuristics computed from the history set on the time-valid evaluation graph.

A is the directed adjacency (A[i, j] = 1: i cites j), h the (B, N) binary history indicator.
  co-citation(c)  = sum_{x in H} #papers citing both x and c      = (h A^T A)[c]
  coupling(c)     = sum_{x in H} #references shared by x and c    = (h A A^T)[c]
  both            = co-citation + coupling
  PPR             = personalised PageRank restarting uniformly on H
All scores are back-filled with popularity: counts are integers, so adding
0.5 * pop/max(pop) (< 1) breaks ties by popularity without changing the count order, and
candidates with zero overlap fall back to the popularity ranking.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch

from .base import Recommender, history_matrix, to_dense_tensor


def _pop_norm(graph) -> np.ndarray:
    p = np.log1p(graph.in_degree)
    return (p / max(p.max(), 1e-12)).astype(np.float32)


class CoCitation(Recommender):
    name = "cocitation"
    defaults = {"mode": "both"}   # cocitation | coupling | both

    def prepare(self, graph, which):
        super().prepare(graph, which)
        self.A, self.AT = graph.A, graph.AT
        self.pop = torch.as_tensor(_pop_norm(graph))
        return self

    def score_batch(self, qs, rows):
        H = history_matrix(qs, rows, self.A.shape[0], normalize=False)
        mode = self.params["mode"]
        s = sp.csr_matrix((len(rows), self.A.shape[0]), dtype=np.float32)
        if mode in ("cocitation", "both"):
            s = s + (H @ self.AT) @ self.A
        if mode in ("coupling", "both"):
            s = s + (H @ self.A) @ self.AT
        return to_dense_tensor(s.toarray()) + 0.5 * self.pop


class PersonalizedPageRank(Recommender):
    name = "ppr"
    defaults = {"alpha": 0.15, "num_iter": 20, "direction": "both"}  # restart prob; out | in | both

    def prepare(self, graph, which):
        super().prepare(graph, which)
        A = graph.A
        d = self.params["direction"]
        S = A if d == "out" else A.T if d == "in" else ((A + A.T) > 0).astype(np.float32)
        deg = np.asarray(S.sum(1)).ravel()
        P = sp.diags(1.0 / np.maximum(deg, 1)) @ S          # row-stochastic (dangling rows = 0)
        PT = P.T.tocsr().astype(np.float32)                   # x_{t+1}^T = PT @ x_t^T
        self.PT = torch.sparse_csr_tensor(torch.as_tensor(PT.indptr), torch.as_tensor(PT.indices),
                                          torch.as_tensor(PT.data), PT.shape)
        self.pop = torch.as_tensor(_pop_norm(graph))
        self.n = A.shape[0]
        return self

    def score_batch(self, qs, rows):
        a, T = self.params["alpha"], self.params["num_iter"]
        h = to_dense_tensor(history_matrix(qs, rows, self.n).toarray()).T.contiguous()  # (N, B), cols sum to 1
        x = h.clone()
        for _ in range(T):
            x = (1 - a) * (self.PT @ x) + a * h
        x = x.T
        # unreached candidates (score 0) are back-filled with popularity, strictly below reached ones
        return torch.where(x > 0, x, self.pop - 1.0)
