"""Leakage checks.

Part 1 runs on a synthetic graph (always). Part 2 runs on the real saved split and training
graph (skipped if the dataset has not been downloaded yet).
"""
import numpy as np
import pytest

from src.data import ArxivData, QuerySet, build_split, eligible_references, load_or_build_split
from src.graph import CitationGraph, build_eval_graph, build_train_graph, query_edges
from src.utils import ROOT, load_config


def check_split_integrity(d: ArxivData, split, scfg):
    R = eligible_references(d)
    for name, qs in (("train", split.train), ("val", split.val), ("test", split.test)):
        if name == "train":
            assert (qs.year <= scfg["train_max_year"]).all()
        else:
            assert (qs.year == scfg[f"{name}_year"]).all()
            cap = scfg[f"max_{name}_queries"]
            assert cap is None or len(qs) <= cap
        assert len(np.unique(qs.node)) == len(qs.node)
        for i in range(len(qs)):
            p, h, t = qs.node[i], qs.history(i), qs.targets(i)
            assert len(h) >= 1 and len(t) >= 1
            assert len(h) + len(t) >= scfg["min_refs"]
            assert not set(h) & set(t), "history and targets overlap"
            assert p not in h and p not in t
            assert (d.year[h] < d.year[p]).all() and (d.year[t] < d.year[p]).all()
            refs = set(R.indices[R.indptr[p]:R.indptr[p + 1]].tolist())
            assert set(h) | set(t) == refs, "history+targets must equal P's eligible references"


def check_graph_leak_free(d: ArxivData, split, g: CitationGraph, evaluated_on: list[str]):
    """``g`` must be safe for training and for evaluating the splits in ``evaluated_on``."""
    E = g.edge_set()
    eval_nodes = np.concatenate([split.val.node, split.test.node])
    # (1) no outgoing edge of any validation/test query
    assert not np.isin(g.edge_index[0], eval_nodes).any()
    # (2) no hidden target edge (nor history edge) of any validation/test query
    for qs in (split.val, split.test):
        for which in ("targets", "history"):
            rows, cols = query_edges(qs, which)
            assert not any((r, c) in E for r, c in zip(rows.tolist(), cols.tolist()))
    # (3) time validity: every edge's citing paper is older than every query evaluated on g,
    #     so statistics such as popularity counts only use edges older than the query
    for name in evaluated_on:
        if len(split[name]):
            assert (d.year[g.edge_index[0]] < split[name].year.min()).all()
    # (4) the popularity statistic (in-degree) is exactly the count over those edges
    uniq = np.unique(g.edge_index.T, axis=0)
    ref = np.bincount(uniq[:, 1], minlength=d.num_nodes)
    assert (g.in_degree == ref).all()


def check_all_graphs(d, split):
    tg = build_train_graph(d, split)
    assert (d.year[tg.edge_index[0]] <= split.config["train_max_year"]).all()
    check_graph_leak_free(d, split, tg, ["val", "test"])
    check_graph_leak_free(d, split, build_eval_graph(d, split, "val"), ["val"])
    check_graph_leak_free(d, split, build_eval_graph(d, split, "test"), ["test"])
    return tg


# ============================================================ synthetic graph
def synthetic(seed=0, n=400):
    rng = np.random.default_rng(seed)
    year = rng.integers(2010, 2020, size=n)
    year[:40] = 2018
    year[40:80] = 2019
    src, dst = [], []
    for p in range(n):
        older = np.flatnonzero(year < year[p])
        k = min(len(older), rng.integers(0, 12))
        src += [p] * k
        dst += rng.choice(older, size=k, replace=False).tolist() if k else []
        # a few same-year and "future" citations, which must never become queries/targets
        if rng.random() < 0.2:
            src.append(p)
            dst.append(int(rng.integers(0, n)))
    ei = np.array([src, dst], dtype=np.int64)
    return ArxivData(n, ei, year.astype(np.int64), np.zeros((n, 4), np.float32), np.zeros(n, np.int64),
                     np.arange(n))


SCFG = dict(min_refs=5, history_frac=0.5, seed=0, train_max_year=2017, val_year=2018, test_year=2019,
            max_val_queries=25, max_test_queries=None)


def test_synthetic_split_and_graph():
    d = synthetic()
    s = build_split(d, SCFG)
    assert len(s.val) > 0 and len(s.test) > 0 and len(s.train) > 0
    check_split_integrity(d, s, SCFG)
    check_all_graphs(d, s)


def test_split_is_deterministic():
    d = synthetic()
    a, b = build_split(d, SCFG), build_split(d, SCFG)
    for name in ("train", "val", "test"):
        for k in ("node", "hist_idx", "tgt_idx", "hist_ptr"):
            assert np.array_equal(getattr(a[name], k), getattr(b[name], k))
    c = build_split(d, {**SCFG, "seed": 1})
    assert not np.array_equal(a.val.hist_idx, c.val.hist_idx)


def test_history_fraction_respected():
    d = synthetic()
    for frac in (0.25, 0.75):
        s = build_split(d, {**SCFG, "history_frac": frac})
        n = np.diff(s.test.hist_ptr) + np.diff(s.test.tgt_ptr)
        h = np.diff(s.test.hist_ptr)
        assert np.all(np.abs(h - np.clip(np.round(frac * n), 1, n - 1)) == 0)


# ============================================================ real data
DATA_READY = (ROOT / "data" / "ogbn_arxiv" / "raw" / "edge.csv.gz").exists()


@pytest.fixture(scope="module")
def real():
    from src.data import load_arxiv
    cfg = load_config()
    d = load_arxiv(cfg["paths"]["data_root"], with_text=False)
    split = load_or_build_split(d, cfg)
    return d, split, cfg


@pytest.mark.skipif(not DATA_READY, reason="ogbn-arxiv not downloaded")
def test_real_split_integrity(real):
    d, split, cfg = real
    check_split_integrity(d, split, cfg["split"])


@pytest.mark.skipif(not DATA_READY, reason="ogbn-arxiv not downloaded")
def test_real_train_graph_leak_free(real):
    d, split, cfg = real
    g = check_all_graphs(d, split)
    # sanity: training graph is a strict, non-trivial subset of the full graph
    assert 0.3 * d.edge_index.shape[1] < g.num_edges < d.edge_index.shape[1]


def test_popularity_uses_only_time_valid_edges():
    from src.models.popularity import Popularity
    d = synthetic()
    s = build_split(d, SCFG)
    for which in ("val", "test"):
        g = build_eval_graph(d, s, which)
        pop = Popularity().prepare(g, which).pop.numpy()
        qy = s[which].year.min()
        src, dst = d.edge_index
        keep = (d.year[src] < qy) & ~np.isin(src, np.concatenate([s.val.node, s.test.node]))
        uniq = np.unique(d.edge_index[:, keep].T, axis=0)
        assert np.allclose(pop, np.log1p(np.bincount(uniq[:, 1], minlength=d.num_nodes)))
