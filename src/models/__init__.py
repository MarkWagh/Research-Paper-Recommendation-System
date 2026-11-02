"""Model registry: ``build_model(name, **params)``."""
from importlib import import_module

_REGISTRY = {
    "popularity": ("popularity", "Popularity"),
    "tfidf": ("content", "TfidfModel"),
    "bm25": ("content", "BM25Model"),
    "embedding": ("content", "EmbeddingModel"),
    "cocitation": ("graph_heuristics", "CoCitation"),
    "ppr": ("graph_heuristics", "PersonalizedPageRank"),
    "node2vec": ("node2vec", "Node2VecModel"),
    "fusion": ("fusion", "ScoreFusion"),
    "gnn": ("gnn", "GNNHybrid"),
}


def build_model(name: str, **params):
    if name not in _REGISTRY:
        raise KeyError(f"unknown model {name!r}; choose from {sorted(_REGISTRY)}")
    mod, cls = _REGISTRY[name]
    return getattr(import_module(f".{mod}", __name__), cls)(**params)
