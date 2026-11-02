"""Popularity baseline: rank candidates by how often they are cited in the time-valid graph."""
from __future__ import annotations

import numpy as np
import torch

from .base import Recommender


class Popularity(Recommender):
    name = "popularity"
    defaults = {}

    def prepare(self, graph, which):
        super().prepare(graph, which)
        # In-degree within the evaluation graph: only citations made by papers older than the
        # query (see src/graph.py), so no future information leaks into the ranking.
        self.pop = torch.as_tensor(np.log1p(graph.in_degree), dtype=torch.float32)
        return self

    def score_batch(self, qs, rows):
        return self.pop.unsqueeze(0).expand(len(rows), -1)
