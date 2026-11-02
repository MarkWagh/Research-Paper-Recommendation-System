"""Model-level unit tests on tiny inputs."""
import numpy as np
import pytest
import scipy.sparse as sp
import torch

from src.data import QuerySet
from src.graph import CitationGraph
from src.models.base import history_mean
from src.models.content import BM25Weights
from src.models.graph_heuristics import CoCitation, PersonalizedPageRank


def test_bm25_matches_rank_bm25():
    rank_bm25 = pytest.importorskip("rank_bm25")
    corpus = [["graph", "neural", "network"], ["graph", "graph", "citation"], ["text", "embedding", "model"],
              ["neural", "text", "model", "model"], ["citation", "recommendation"], ["graph", "recommendation"]]
    queries = [["graph", "recommendation"], ["neural", "neural", "model"], ["graph"]]
    vocab = sorted({w for doc in corpus + queries for w in doc})
    col = {w: i for i, w in enumerate(vocab)}

    def counts(docs):
        r, c = [], []
        for i, doc in enumerate(docs):
            for w in doc:
                r.append(i)
                c.append(col[w])
        return sp.csr_matrix((np.ones(len(r), np.float32), (r, c)), shape=(len(docs), len(vocab)))

    ref = rank_bm25.BM25Okapi(corpus)
    W = BM25Weights(counts(corpus)).W
    ours = (counts(queries) @ W.T).toarray()
    for qi, q in enumerate(queries):
        assert np.allclose(ours[qi], ref.get_scores(q), atol=1e-5)


def _toy_graph():
    # 0 cites 2,3 ; 1 cites 2,3,4 ; 5 cites 4 ; 6 cites 3
    ei = np.array([[0, 0, 1, 1, 1, 5, 6], [2, 3, 2, 3, 4, 4, 3]])
    return CitationGraph(ei, 8, "toy")


def test_cocitation_and_coupling_counts():
    g = _toy_graph()
    qs = QuerySet.from_lists([7], [2020], [np.array([2])], [np.array([3])])
    m = CoCitation(mode="cocitation").prepare(g, "val")
    s = m.score_batch(qs, np.array([0])).numpy()[0] - 0.5 * m.pop.numpy()
    # papers co-cited with 2: 3 (by 0 and 1) -> 2 ; 4 (by 1) -> 1
    assert np.allclose(s, [0, 0, 2, 2, 1, 0, 0, 0])  # node 2 (history, masked later) co-cited with itself twice
    qs2 = QuerySet.from_lists([7], [2020], [np.array([0])], [np.array([1])])
    m2 = CoCitation(mode="coupling").prepare(g, "val")
    s2 = m2.score_batch(qs2, np.array([0])).numpy()[0] - 0.5 * m2.pop.numpy()
    # references shared with 0 (= {2,3}): paper 1 shares 2, paper 6 shares 1, paper 0 itself 2
    assert np.allclose(s2[[1, 6, 5]], [2, 1, 0])


def test_ppr_scores_sum_and_locality():
    g = _toy_graph()
    qs = QuerySet.from_lists([7], [2020], [np.array([5])], [np.array([4])])
    m = PersonalizedPageRank(alpha=0.3, num_iter=50).prepare(g, "val")
    s = m.score_batch(qs, np.array([0])).numpy()[0]
    assert s[4] == max(s[i] for i in range(7) if i != 5)  # direct neighbour of the seed ranks first
    assert s[7] < 0  # isolated node is back-filled below every reached node


def test_history_mean():
    E = torch.arange(12, dtype=torch.float32).view(6, 2)
    qs = QuerySet.from_lists([0, 1], [2020, 2020], [np.array([2, 4]), np.array([5])], [np.array([3]), np.array([3])])
    out = history_mean(E, qs, np.array([0, 1]))
    assert torch.allclose(out, torch.stack([(E[2] + E[4]) / 2, E[5]]))


# ------------------------------------------------------------- GNN
def test_gnn_masked_graph_and_inductive_query():
    from src.models.gnn import Encoder, mask_pseudo_queries, scatter_agg, sparse_agg
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    n, din = 30, 8
    A = sp.random(n, n, density=0.15, format="csr", random_state=1, dtype=np.float32)
    A.data[:] = 1.0
    A.setdiag(0)
    A.eliminate_zeros()
    # (1) pseudo-queries lose every outgoing edge, other rows are untouched
    Q = np.array([2, 5, 11])
    M = mask_pseudo_queries(A, Q)
    assert M[Q].nnz == 0
    others = np.setdiff1d(np.arange(n), Q)
    assert (M[others] != A[others]).nnz == 0
    # (2) sampled-style scatter aggregation == full sparse aggregation on the same graph
    net = Encoder(din, 16, 16, 2, dropout=0.0).eval()
    x = torch.randn(n, din)
    coo = A.tocoo()
    ref = torch.as_tensor(np.vstack([coo.col, coo.row]), dtype=torch.int64)
    cit = torch.as_tensor(np.vstack([coo.row, coo.col]), dtype=torch.int64)
    with torch.no_grad():
        h1 = net.items(x, scatter_agg(ref, cit, n))
        h2 = net.items(x, sparse_agg(A))
    for a, b in zip(h1, h2):
        assert torch.allclose(a, b, atol=1e-5)
    # (3) cold-start query == item embedding of a new node P that aggregates its history as
    #     references, while the history items do not see P (P is absent from the graph at
    #     evaluation time, so it must not become their citer)
    P, H = n, np.array([3, 7, 19])
    ref2 = torch.cat([ref, torch.as_tensor(np.vstack([H, np.full(3, P)]), dtype=torch.int64)], 1)
    x_q = torch.randn(1, din)
    with torch.no_grad():
        hs_new = net.items(torch.cat([x, x_q]), scatter_agg(ref2, cit, n + 1))
        z_new = net.project(hs_new[-1][P])
        hs = net.items(x, sparse_agg(A))
        q = net.queries(x_q, [h[torch.as_tensor(H)] for h in hs[:-1]], torch.zeros(3, dtype=torch.int64))
    assert torch.allclose(q[0], z_new, atol=1e-5)


def test_fusion_logrank_normalisation():
    from src.models.fusion import ScoreFusion

    class Fixed:
        def __init__(self, s): self.s = torch.tensor(s)
        def score_batch(self, qs, rows): return self.s

    f = ScoreFusion(norm="logrank")
    f.year = torch.tensor([2000, 2000, 2000, 2010])
    f.models = {"a": Fixed([[5.0, 1.0, 3.0, 9.0]])}
    qs = QuerySet.from_lists([3], [2005], [np.array([0])], [np.array([1])])
    out = f._normalised("a", qs, np.array([0]))[0]
    # eligible = year < 2005 -> nodes 0,1,2 ranked 0 (1st), 2 (2nd), 1 (3rd)
    assert torch.allclose(out[:3], -torch.log(torch.tensor([1.0, 3.0, 2.0])))
    assert out[3] < out[:3].min()
