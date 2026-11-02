"""Full-ranking evaluation harness (no sampled negatives).

For every query P the eligible candidate set is
    { c : year(c) < year(P) }  (if time_mask)   minus  {P}  minus  history(P).
Each hidden target's rank over that set is computed exactly:
    rank(t) = 1 + #{c eligible : s_c > s_t  or (s_c == s_t and prio_c > prio_t)}
where ``prio`` is a fixed seeded random permutation, so score ties are broken randomly but
reproducibly and identically for every model. Scores are processed in query batches on the
selected device (GPU/MPS if available, else CPU); no N x N matrix is ever materialised.
"""
from __future__ import annotations

from typing import Callable, Union

import numpy as np
import torch

from .data import QuerySet
from .metrics import aggregate, per_query_metrics
from .utils import get_logger

log = get_logger(__name__)

ScoreFn = Callable[[np.ndarray], Union[torch.Tensor, np.ndarray]]


class Evaluator:
    def __init__(self, year: np.ndarray, ks=(10, 20, 50), time_mask: bool = True, tie_seed: int = 12345,
                 batch_size: int = 256, device: str | torch.device = "cpu"):
        self.n = len(year)
        self.ks = tuple(ks)
        self.time_mask = time_mask
        self.batch_size = batch_size
        self.device = torch.device(device)
        self.year = torch.as_tensor(year, dtype=torch.int64, device=self.device)
        prio = np.random.default_rng(tie_seed).permutation(self.n).astype(np.float32)
        self.prio = torch.as_tensor(prio, device=self.device)

    @classmethod
    def from_config(cls, d_year: np.ndarray, cfg: dict, device=None) -> "Evaluator":
        from .utils import get_device
        e = cfg["eval"]
        dev = device or get_device(cfg.get("device", "auto"))
        return cls(d_year, e["ks"], e["time_mask"], e["tie_seed"], e["batch_size"], dev)

    # ------------------------------------------------------------------ masks
    def candidate_mask(self, qs: QuerySet, rows: np.ndarray) -> torch.Tensor:
        """(B, N) bool: True where the node is an eligible candidate for the query."""
        B = len(rows)
        if self.time_mask:
            qy = torch.as_tensor(qs.year[rows], device=self.device)
            mask = self.year.unsqueeze(0) < qy.unsqueeze(1)
        else:
            mask = torch.ones(B, self.n, dtype=torch.bool, device=self.device)
        r_idx, c_idx = [], []
        for b, i in enumerate(rows):
            excl = np.append(qs.history(i), qs.node[i])
            r_idx.append(np.full(len(excl), b))
            c_idx.append(excl)
        mask[torch.as_tensor(np.concatenate(r_idx), device=self.device),
             torch.as_tensor(np.concatenate(c_idx), device=self.device)] = False
        return mask

    # ------------------------------------------------------------------ ranking
    def target_ranks(self, scores: torch.Tensor, mask: torch.Tensor, qs: QuerySet, rows: np.ndarray,
                     chunk: int = 512) -> list[np.ndarray]:
        scores = torch.nan_to_num(scores.to(self.device, torch.float32), nan=-torch.inf)
        q_of, t_of = [], []
        for b, i in enumerate(rows):
            t = qs.targets(i)
            q_of.append(np.full(len(t), b))
            t_of.append(t)
        q_of = torch.as_tensor(np.concatenate(q_of), device=self.device)
        t_of = torch.as_tensor(np.concatenate(t_of), device=self.device)
        ranks = torch.empty(len(t_of), dtype=torch.float64)
        for s in range(0, len(t_of), chunk):
            qb, tb = q_of[s:s + chunk], t_of[s:s + chunk]
            S, M = scores[qb], mask[qb]
            st = S.gather(1, tb.unsqueeze(1))
            pt = self.prio[tb].unsqueeze(1)
            ahead = ((S > st) | ((S == st) & (self.prio.unsqueeze(0) > pt))) & M
            ranks[s:s + chunk] = (ahead.sum(1) + 1).double().cpu()
        ranks = ranks.numpy()
        sizes = [len(qs.targets(i)) for i in rows]
        return np.split(ranks, np.cumsum(sizes)[:-1])

    def topk(self, scores: torch.Tensor, mask: torch.Tensor, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Top-k eligible node ids per row with the same tie-breaking as ``target_ranks``."""
        S = torch.nan_to_num(scores.to(self.device, torch.float32), nan=-torch.inf).masked_fill(~mask, -torch.inf)
        # Pre-order columns by descending priority, then stable-sort by score: ties keep priority order.
        order = torch.argsort(self.prio, descending=True)
        vals, pos = torch.sort(S[:, order], dim=1, descending=True, stable=True)
        return order[pos[:, :k]].cpu().numpy(), vals[:, :k].cpu().numpy()

    # ------------------------------------------------------------------ main entry
    def evaluate(self, scores_or_rankings, qs: QuerySet, return_per_query: bool = False,
                 return_topk: int = 0) -> dict:
        """scores_or_rankings is one of:
          * callable f(query_rows: np.ndarray) -> (B, N) scores (streamed in batches),
          * dense (Q, N) array/tensor of scores,
          * integer (Q, K) array of ranked node ids (ineligible ids are dropped; MRR is then
            truncated at K, i.e. a query whose targets are all outside the list scores 0).
        """
        Q = len(qs)
        if callable(scores_or_rankings):
            fn = scores_or_rankings
        else:
            arr = scores_or_rankings
            if isinstance(arr, np.ndarray) and np.issubdtype(arr.dtype, np.integer) or \
               isinstance(arr, torch.Tensor) and not torch.is_floating_point(arr):
                return self._evaluate_rankings(np.asarray(arr), qs, return_per_query)
            arr = torch.as_tensor(arr)
            fn = lambda rows: arr[torch.as_tensor(rows)]

        rank_lists: list[np.ndarray] = []
        tops = []
        for s in range(0, Q, self.batch_size):
            rows = np.arange(s, min(s + self.batch_size, Q))
            scores = torch.as_tensor(fn(rows)).to(self.device, torch.float32)
            assert scores.shape == (len(rows), self.n), f"score fn returned {tuple(scores.shape)}"
            mask = self.candidate_mask(qs, rows)
            rank_lists.extend(self.target_ranks(scores, mask, qs, rows))
            if return_topk:
                tops.append(self.topk(scores, mask, return_topk)[0])
        return self._finish(rank_lists, return_per_query, np.concatenate(tops) if tops else None)

    def _evaluate_rankings(self, rankings: np.ndarray, qs: QuerySet, return_per_query: bool) -> dict:
        assert len(rankings) == len(qs)
        rank_lists = []
        for i in range(len(qs)):
            mask = self.candidate_mask(qs, np.array([i]))[0].cpu().numpy()
            lst = [c for c in rankings[i] if 0 <= c < self.n and mask[c]]
            pos = {c: r + 1 for r, c in enumerate(dict.fromkeys(lst))}
            rank_lists.append(np.array([pos.get(t, np.inf) for t in qs.targets(i)], dtype=np.float64))
        return self._finish(rank_lists, return_per_query, None)

    def _finish(self, rank_lists, return_per_query, tops) -> dict:
        pq = per_query_metrics(rank_lists, self.ks)
        out: dict = {"metrics": aggregate(pq), "num_queries": len(rank_lists)}
        if return_per_query:
            out["per_query"] = pq
            out["target_ranks"] = rank_lists
        if tops is not None:
            out["topk"] = tops
        return out


def evaluate(scores_or_rankings, split: QuerySet, year: np.ndarray, **kw) -> dict:
    """Convenience wrapper: ``evaluate(scores, queryset, node_years) -> {"recall@10": ...}``."""
    ev_kw = {k: kw.pop(k) for k in list(kw) if k in ("ks", "time_mask", "tie_seed", "batch_size", "device")}
    return Evaluator(year, **ev_kw).evaluate(scores_or_rankings, split, **kw)["metrics"]
