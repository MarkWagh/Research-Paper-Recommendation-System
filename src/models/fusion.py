"""Score fusion: weighted sum of per-query normalised component scores.

    s(c) = sum_k w_k * norm_k(s_k(c))
``norm`` is computed per query over its time-eligible candidates (year < year(P)):
  zscore  - (s - mean) / std
  minmax  - (s - min) / (max - min)
  logrank - -log(rank of the candidate within the query's ranking): scale-free, and robust
            to heavy-tailed scores and popularity back-fill values (e.g. PPR, where z-scores
            collapse every reached candidate onto almost the same value). Component scores of a batch
are memoised (float16) so that tuning the weights on validation does not recompute them;
the memo is cleared whenever the evaluation graph changes.
"""
from __future__ import annotations

import numpy as np
import torch

from . import build_model
from .base import Recommender


class ScoreFusion(Recommender):
    name = "fusion"
    # components: {key: {name: <registry name>, params: {...}}}; weights: {key: float}
    defaults = {"components": {}, "weights": {}, "norm": "zscore", "memo_gb": 2.5}

    def __init__(self, **params):
        super().__init__(**params)
        seed = self.params.get("seed", 0)
        self.models = {k: build_model(c["name"], **{**c.get("params", {}), "seed": seed})
                       for k, c in self.params["components"].items()}
        self._memo: dict = {}

    def set_params(self, **kw):
        # flat weight overrides like w_graph=0.5 (used by grid tuning)
        for k in list(kw):
            if k.startswith("w_"):
                self.params["weights"] = {**self.params["weights"], k[2:]: kw.pop(k)}
        return super().set_params(**kw)

    def fit(self, train_graph, features, split):
        super().fit(train_graph, features, split)
        self.year = torch.as_tensor(features.year)
        for m in self.models.values():
            m.fit(train_graph, features, split)
        return self

    def prepare(self, graph, which):
        super().prepare(graph, which)
        self._memo.clear()
        for m in self.models.values():
            m.prepare(graph, which)
        return self

    def _normalised(self, key, qs, rows) -> torch.Tensor:
        mk = (key, self.params["norm"], id(qs), int(rows[0]), len(rows))
        if mk in self._memo:
            return self._memo[mk].float()
        s = self.models[key].score_batch(qs, rows).float().cpu()
        elig = self.year.unsqueeze(0) < torch.as_tensor(qs.year[rows]).unsqueeze(1)
        s = torch.where(elig, s, torch.nan)
        if self.params["norm"] == "zscore":
            mu = torch.nanmean(s, 1, keepdim=True)
            sd = torch.sqrt(torch.nanmean((s - mu) ** 2, 1, keepdim=True)).clamp(min=1e-8)
            s = (s - mu) / sd
        elif self.params["norm"] == "minmax":
            lo = torch.where(elig, s, torch.inf).min(1, keepdim=True).values
            hi = torch.where(elig, s, -torch.inf).max(1, keepdim=True).values
            s = (s - lo) / (hi - lo).clamp(min=1e-12)
        elif self.params["norm"] == "logrank":
            order = torch.argsort(torch.nan_to_num(s, nan=-torch.inf), dim=1, descending=True, stable=True)
            rank = torch.empty_like(s)
            rank.scatter_(1, order, torch.arange(1, s.shape[1] + 1, dtype=s.dtype).expand_as(s).contiguous())
            s = torch.where(elig, -torch.log(rank), torch.nan)
        else:
            raise ValueError(self.params["norm"])
        s = torch.nan_to_num(s, nan=-1e4)  # ineligible: masked by the evaluator anyway
        if sum(v.numel() * 2 for v in self._memo.values()) + s.numel() * 2 < self.params["memo_gb"] * 1e9:
            self._memo[mk] = s.half()
        return s

    def score_batch(self, qs, rows):
        out = None
        for key, w in self.params["weights"].items():
            if w == 0:
                continue
            s = w * self._normalised(key, qs, rows)
            out = s if out is None else out + s
        return out

    def close(self):
        self._memo.clear()
