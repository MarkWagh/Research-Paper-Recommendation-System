"""Content-only recommenders: TF-IDF, BM25 and dense-embedding cosine.

Query representations (``query_mode``):
  own      - the query paper's own title+abstract
  history  - the mean of the history papers' representations
  combo    - alpha * own + (1 - alpha) * history
For TF-IDF and dense embeddings the combination happens in (L2-normalised) vector space,
which equals a weighted sum of cosine similarities. BM25 scores are unbounded, so its combo
mixes per-query max-normalised score vectors.

Text statistics (vocabulary, IDF, average document length) are computed over all papers'
titles/abstracts. This uses no citation information, and the text of every paper - including
the query itself - is part of the model's input under the task definition.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import normalize

from ..utils import Timer, get_device, get_logger
from .base import Recommender, history_matrix, history_mean, to_dense_tensor

log = get_logger(__name__)


def _texts(features) -> list[str]:
    d = features.d
    return [f"{d.title[i]}. {d.abstract[i]}" for i in range(d.num_nodes)]


class _QueryMixin:
    """Builds query vectors from a (N, D) representation matrix ``self.X`` (sparse or dense)."""

    def _own(self, qs, rows):
        return self.X[qs.node[rows]]

    def _hist(self, qs, rows):
        return history_matrix(qs, rows, self.X.shape[0]) @ self.X


# ------------------------------------------------------------------- TF-IDF
class TfidfModel(_QueryMixin, Recommender):
    name = "tfidf"
    defaults = {"query_mode": "combo", "alpha": 0.5, "min_df": 2, "max_df": 0.5, "max_features": 200000,
                "ngram_range": [1, 1], "sublinear_tf": True}

    def fit(self, train_graph, features, split):
        super().fit(train_graph, features, split)
        p = self.params
        with Timer(log, "TF-IDF fit"):
            vec = TfidfVectorizer(min_df=p["min_df"], max_df=p["max_df"], max_features=p["max_features"],
                                  ngram_range=tuple(p["ngram_range"]), sublinear_tf=p["sublinear_tf"],
                                  stop_words="english", dtype=np.float32)
            self.X = vec.fit_transform(_texts(features)).tocsr()   # rows L2-normalised
            self.XT = self.X.T.tocsr()
        log.info(f"TF-IDF matrix {self.X.shape}, nnz={self.X.nnz}")
        return self

    def score_batch(self, qs, rows):
        mode, a = self.params["query_mode"], self.params["alpha"]
        if mode == "own":
            Q = self._own(qs, rows)
        elif mode == "history":
            Q = normalize(self._hist(qs, rows))
        else:
            Q = a * normalize(self._own(qs, rows)) + (1 - a) * normalize(self._hist(qs, rows))
        return to_dense_tensor((Q @ self.XT).toarray())


# ------------------------------------------------------------------- BM25
class BM25Weights:
    """Okapi BM25 identical to rank_bm25.BM25Okapi, as a sparse (N, V) weight matrix W so
    that scores(query) = W @ query_term_counts."""

    def __init__(self, counts: sp.csr_matrix, k1: float = 1.5, b: float = 0.75, epsilon: float = 0.25):
        counts = counts.tocsr().astype(np.float32)
        n_docs = counts.shape[0]
        dl = np.asarray(counts.sum(1)).ravel()
        avgdl = dl.mean()
        df = np.bincount(counts.indices, minlength=counts.shape[1]).astype(np.float64)
        idf = np.log(n_docs - df + 0.5) - np.log(df + 0.5)
        # rank_bm25 averages idf over terms *present in the corpus* and floors negative idf
        present = df > 0
        idf[present & (idf < 0)] = epsilon * idf[present].mean()
        self.idf = idf
        rows = np.repeat(np.arange(n_docs), np.diff(counts.indptr))
        tf = counts.data
        denom = tf + k1 * (1 - b + b * dl[rows] / avgdl)
        w = idf[counts.indices] * tf * (k1 + 1) / denom
        self.W = sp.csr_matrix((w.astype(np.float32), counts.indices, counts.indptr), shape=counts.shape)


class BM25Model(_QueryMixin, Recommender):
    name = "bm25"
    defaults = {"query_mode": "combo", "alpha": 0.5, "k1": 1.5, "b": 0.75, "min_df": 2, "max_df": 0.5,
                "max_features": 200000}

    def fit(self, train_graph, features, split):
        super().fit(train_graph, features, split)
        p = self.params
        with Timer(log, "BM25 fit"):
            cv = CountVectorizer(min_df=p["min_df"], max_df=p["max_df"], max_features=p["max_features"],
                                 stop_words="english", dtype=np.float32)
            self.X = cv.fit_transform(_texts(features)).tocsr()   # term counts = query vectors
            self.WT = BM25Weights(self.X, p["k1"], p["b"]).W.T.tocsr()
        return self

    def _scores(self, Q):
        return to_dense_tensor((Q @ self.WT).toarray())

    def score_batch(self, qs, rows):
        mode, a = self.params["query_mode"], self.params["alpha"]
        if mode == "own":
            return self._scores(self._own(qs, rows))
        if mode == "history":
            return self._scores(self._hist(qs, rows))
        so, sh = self._scores(self._own(qs, rows)), self._scores(self._hist(qs, rows))
        mx = lambda s: s / s.max(1, keepdim=True).values.clamp(min=1e-12)
        return a * mx(so) + (1 - a) * mx(sh)


# ------------------------------------------------------------------- dense embeddings
class EmbeddingModel(Recommender):
    """Cosine similarity on a dense feature matrix: ogb (128-d), minilm, specter2."""
    name = "embedding"
    defaults = {"feature": "specter2", "query_mode": "combo", "alpha": 0.5}

    def fit(self, train_graph, features, split):
        super().fit(train_graph, features, split)
        self.device = get_device(self.params.get("device", "auto"))
        self.E = torch.as_tensor(features[self.params["feature"]], device=self.device)  # L2-normalised
        self.n = self.E.shape[0]
        return self

    def query_vectors(self, qs, rows) -> torch.Tensor:
        mode, a = self.params["query_mode"], self.params["alpha"]
        nrm = lambda x: torch.nn.functional.normalize(x, dim=1)
        own = self.E[torch.as_tensor(qs.node[rows], device=self.device)]
        if mode == "own":
            return own
        hist = nrm(history_mean(self.E, qs, rows))
        return hist if mode == "history" else a * own + (1 - a) * hist

    def score_batch(self, qs, rows):
        return self.query_vectors(qs, rows) @ self.E.T
