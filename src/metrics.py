"""Ranking metrics computed from the 1-based ranks of each hidden target in the full ranking.

For a query with target set T and target ranks r_t (over the eligible candidate set):
  Recall@K = |{t : r_t <= K}| / |T|
  NDCG@K   = sum_{t : r_t <= K} 1/log2(r_t + 1)  /  sum_{i=1}^{min(|T|, K)} 1/log2(i + 1)
  MRR      = 1 / min_t r_t        (reciprocal rank of the first hit, over the full ranking)
Ranks may be ``np.inf`` (target absent from a truncated ranking): it then counts as a miss.
"""
from __future__ import annotations

import numpy as np


def recall_at_k(ranks: np.ndarray, k: int) -> float:
    ranks = np.asarray(ranks, dtype=np.float64)
    return float((ranks <= k).sum() / len(ranks)) if len(ranks) else 0.0


def ndcg_at_k(ranks: np.ndarray, k: int) -> float:
    ranks = np.asarray(ranks, dtype=np.float64)
    if len(ranks) == 0:
        return 0.0
    hit = ranks[ranks <= k]
    dcg = (1.0 / np.log2(hit + 1.0)).sum()
    idcg = (1.0 / np.log2(np.arange(1, min(len(ranks), k) + 1) + 1.0)).sum()
    return float(dcg / idcg)


def reciprocal_rank(ranks: np.ndarray) -> float:
    ranks = np.asarray(ranks, dtype=np.float64)
    return float(1.0 / ranks.min()) if len(ranks) and np.isfinite(ranks.min()) else 0.0


def per_query_metrics(rank_lists: list[np.ndarray], ks=(10, 20, 50)) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for k in ks:
        out[f"recall@{k}"] = np.array([recall_at_k(r, k) for r in rank_lists])
        out[f"ndcg@{k}"] = np.array([ndcg_at_k(r, k) for r in rank_lists])
    out["mrr"] = np.array([reciprocal_rank(r) for r in rank_lists])
    return out


def aggregate(per_query: dict[str, np.ndarray]) -> dict[str, float]:
    return {k: float(np.mean(v)) if len(v) else 0.0 for k, v in per_query.items()}
