"""Leak-free graph construction.

Two graph views are used, both built from the directed citation edges (A[i, j] = 1: i cites j):

* **Training graph** - edges whose citing paper is from the training era
  (year <= train_max_year). Learned models (node2vec, GNN) are trained on it.
* **Evaluation graph for split s** - edges whose citing paper is strictly older than the
  split's query year, minus *every* outgoing edge of *every* validation/test query paper.
  For validation (2018) this equals the training graph. For test (2019) it additionally
  contains the citations made by non-query 2018 papers, which are legitimately observable
  before 2019 (without them 2018 papers - 25% of test targets - would be isolated).
  Graph statistics used at evaluation time (popularity, co-citation, PPR, node2vec refit,
  GNN neighbourhoods) are computed from this graph only.

Both views therefore (asserted in tests/test_leakage.py):
  * contain no outgoing edge of any validation/test query (hence no hidden target edge);
  * only contain edges whose citing paper is older than every query evaluated on them.

Training queries (year <= 2017) are training-era papers: their reference lists *are* the
training signal. Their history/target split is only used by models that train in the query
format, which must mask the targets themselves (see models/gnn.py).
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from .data import ArxivData, QuerySet, Split, edges_to_csr


class CitationGraph:
    """Directed graph with cached CSR views. A[i, j] = 1 iff i cites j."""

    def __init__(self, edge_index: np.ndarray, num_nodes: int, name: str = "", max_citing_year: int | None = None):
        self.name = name
        self.num_nodes = num_nodes
        self.max_citing_year = max_citing_year
        self.A = edges_to_csr(edge_index, num_nodes)                    # citing -> cited
        coo = self.A.tocoo()
        self.edge_index = np.vstack([coo.row, coo.col]).astype(np.int64)  # deduplicated
        self.AT = self.A.T.tocsr()                                      # cited -> citing
        self.in_degree = np.asarray(self.A.sum(0)).ravel()              # times cited
        self.out_degree = np.asarray(self.A.sum(1)).ravel()             # references

    @property
    def num_edges(self) -> int:
        return int(self.A.nnz)

    def edge_set(self) -> set[tuple[int, int]]:
        return set(zip(self.edge_index[0].tolist(), self.edge_index[1].tolist()))

    def __repr__(self) -> str:
        return f"CitationGraph({self.name}, N={self.num_nodes}, E={self.num_edges})"


def eval_query_nodes(split: Split) -> np.ndarray:
    return np.concatenate([split.val.node, split.test.node])


def build_train_graph(d: ArxivData, split: Split) -> CitationGraph:
    max_y = split.config["train_max_year"]
    src = d.edge_index[0]
    keep = (d.year[src] <= max_y) & ~np.isin(src, eval_query_nodes(split))
    return CitationGraph(d.edge_index[:, keep], d.num_nodes, "train", max_y)


def build_eval_graph(d: ArxivData, split: Split, which: str) -> CitationGraph:
    """Observed graph for evaluating split ``which`` ('val' or 'test')."""
    if which == "train":
        return build_train_graph(d, split)
    qy = int(split[which].year.min())
    assert (split[which].year == qy).all(), "evaluation queries of one split share a year"
    src = d.edge_index[0]
    keep = (d.year[src] < qy) & ~np.isin(src, eval_query_nodes(split))
    return CitationGraph(d.edge_index[:, keep], d.num_nodes, f"eval_{which}", qy - 1)


def remove_edges(A: sp.csr_matrix, rows: np.ndarray, cols: np.ndarray) -> sp.csr_matrix:
    """Return a copy of binary CSR A with entries (rows[k], cols[k]) removed."""
    if len(rows) == 0:
        return A.copy()
    D = sp.csr_matrix((np.ones(len(rows), np.float32), (rows, cols)), shape=A.shape)
    D.data[:] = 1.0
    out = (A - A.multiply(D)).tocsr()
    out.eliminate_zeros()
    return out


def query_edges(qs: QuerySet, which: str) -> tuple[np.ndarray, np.ndarray]:
    ptr, idx = (qs.hist_ptr, qs.hist_idx) if which == "history" else (qs.tgt_ptr, qs.tgt_idx)
    rows = np.repeat(qs.node, np.diff(ptr))
    return rows, idx
