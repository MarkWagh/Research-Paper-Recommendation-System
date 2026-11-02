"""Metric and evaluator checks against small hand-computed examples."""
import math

import numpy as np
import pytest
import torch

from src.data import QuerySet
from src.evaluate import Evaluator, evaluate
from src.metrics import ndcg_at_k, per_query_metrics, recall_at_k, reciprocal_rank

L2, L3, L4 = math.log2(2), math.log2(3), math.log2(4)


# ------------------------------------------------------------- pure metrics
def test_recall():
    assert recall_at_k([1, 3], 1) == 0.5
    assert recall_at_k([1, 3], 2) == 0.5
    assert recall_at_k([1, 3], 3) == 1.0
    assert recall_at_k([np.inf], 50) == 0.0


def test_ndcg():
    # ranks {1, 3}, K=2: DCG = 1/log2(2); IDCG = 1/log2(2) + 1/log2(3)
    assert ndcg_at_k([1, 3], 2) == pytest.approx((1 / L2) / (1 / L2 + 1 / L3))
    # K=3: DCG = 1 + 1/log2(4); IDCG = 1 + 1/log2(3)
    assert ndcg_at_k([1, 3], 3) == pytest.approx((1 + 1 / L4) / (1 + 1 / L3))
    # single target at rank 2: 1/log2(3)
    assert ndcg_at_k([2], 10) == pytest.approx(1 / L3)
    # perfect ranking
    assert ndcg_at_k([1, 2, 3], 10) == pytest.approx(1.0)
    # IDCG is capped at K ideal positions: 5 targets, K=2, top-2 hits => 1.0
    assert ndcg_at_k([1, 2, 30, 40, 50], 2) == pytest.approx(1.0)
    assert ndcg_at_k([np.inf, np.inf], 10) == 0.0


def test_mrr():
    assert reciprocal_rank([4, 2, 9]) == 0.5
    assert reciprocal_rank([1]) == 1.0
    assert reciprocal_rank([np.inf]) == 0.0


def test_per_query_keys():
    pq = per_query_metrics([np.array([1.0]), np.array([np.inf])], ks=(10, 20, 50))
    assert set(pq) == {"recall@10", "ndcg@10", "recall@20", "ndcg@20", "recall@50", "ndcg@50", "mrr"}
    assert pq["mrr"].tolist() == [1.0, 0.0]


# ------------------------------------------------------------- evaluator on a toy corpus
YEARS = np.array([2000, 2000, 2001, 2001, 2002, 2003])


def toy_queries():
    # query 0: paper 5 (2003), history {0}, targets {2, 4}
    # query 1: paper 3 (2001), history {0}, targets {1}
    return QuerySet.from_lists([5, 3], [2003, 2001], [np.array([0]), np.array([0])],
                               [np.array([2, 4]), np.array([1])])


SCORES = np.array([
    [9.0, 1.0, 5.0, 7.0, 3.0, 100.0],   # eligible {1,2,3,4}: order 3,2,4,1 -> rank(2)=2, rank(4)=3
    [0.0, 1.0, 50.0, 2.0, 99.0, 99.0],  # eligible {1} only (year < 2001, minus history 0) -> rank 1
], dtype=np.float32)


def test_evaluator_hand_computed():
    out = Evaluator(YEARS, ks=(1, 2, 3)).evaluate(SCORES, toy_queries(), return_per_query=True)
    r = out["target_ranks"]
    assert r[0].tolist() == [2.0, 3.0]
    assert r[1].tolist() == [1.0]
    pq = out["per_query"]
    assert pq["recall@1"].tolist() == [0.0, 1.0]
    assert pq["recall@2"].tolist() == [0.5, 1.0]
    assert pq["recall@3"].tolist() == [1.0, 1.0]
    assert pq["mrr"].tolist() == [0.5, 1.0]
    assert pq["ndcg@3"][0] == pytest.approx((1 / L3 + 1 / L4) / (1 + 1 / L3))
    assert out["metrics"]["mrr"] == pytest.approx(0.75)


def test_history_and_self_are_excluded():
    # Self (score 100) and history item (score 9) would otherwise rank above every target.
    out = Evaluator(YEARS, ks=(1,)).evaluate(SCORES[:1], toy_queries().subset(np.array([0])), return_per_query=True)
    assert out["target_ranks"][0].tolist() == [2.0, 3.0]


def test_time_mask_effect():
    qs = toy_queries().subset(np.array([1]))
    on = Evaluator(YEARS, ks=(1,), time_mask=True).evaluate(SCORES[1:], qs, return_per_query=True)
    off = Evaluator(YEARS, ks=(1,), time_mask=False).evaluate(SCORES[1:], qs, return_per_query=True)
    assert on["target_ranks"][0].tolist() == [1.0]
    # without mask, eligible = {1,2,4,5} (minus self 3 and history 0); target 1 has the lowest score
    assert off["target_ranks"][0].tolist() == [4.0]


def test_callable_equals_dense_and_rankings():
    qs = toy_queries()
    dense = evaluate(SCORES, qs, YEARS, ks=(1, 2, 3))
    fn = evaluate(lambda rows: torch.as_tensor(SCORES[rows]), qs, YEARS, ks=(1, 2, 3), batch_size=1)
    assert dense == fn
    # explicit rankings containing ineligible ids (self, history, future) that must be dropped
    rankings = np.array([[5, 0, 3, 2, 4, 1], [4, 2, 0, 1, 5, 3]])
    rk = evaluate(rankings, qs, YEARS, ks=(1, 2, 3))
    assert rk == pytest.approx(dense)


def test_ties_broken_by_fixed_priority_and_consistent_with_topk():
    n = 50
    years = np.full(n, 2000)
    years[-1] = 2010
    qs = QuerySet.from_lists([n - 1], [2010], [np.array([0])], [np.array([7, 21])])
    ev = Evaluator(years, ks=(10,), tie_seed=3)
    scores = torch.zeros(1, n)
    ranks = ev.evaluate(scores, qs, return_per_query=True)["target_ranks"][0]
    mask = ev.candidate_mask(qs, np.array([0]))
    top, _ = ev.topk(scores, mask, n - 2)
    top = top[0].tolist()
    assert [top.index(t) + 1 for t in (7, 21)] == ranks.tolist()
    # same seed => identical ranks; different seed => (almost surely) different
    again = Evaluator(years, ks=(10,), tie_seed=3).evaluate(scores, qs, return_per_query=True)["target_ranks"][0]
    assert again.tolist() == ranks.tolist()


def test_ties_with_nonzero_scores():
    # Target tied with exactly one other candidate: rank is 1 or 2 depending on priority.
    years = np.array([2000, 2000, 2000, 2005])
    qs = QuerySet.from_lists([3], [2005], [np.array([0])], [np.array([1])])
    ev = Evaluator(years, ks=(1,))
    r = ev.evaluate(np.array([[0.0, 1.0, 1.0, 0.0]], dtype=np.float32), qs, return_per_query=True)["target_ranks"][0][0]
    expected = 1 + int(ev.prio[2] > ev.prio[1])
    assert r == expected
