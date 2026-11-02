"""Dataset loading, raw-text join, and query split construction.

Conventions used everywhere in this repo:
  * ``edge_index[0] -> edge_index[1]`` means "paper src cites paper dst" (directed).
  * A *query* is a paper P with >= ``min_refs`` eligible references, where a reference is
    eligible iff it is an in-dataset paper published strictly before P (year(ref) < year(P)).
    Only eligible references can ever be ranked, since candidates are restricted to
    year < year(P).
  * The eligible references are split (fixed seed) into ``history`` (revealed) and
    ``targets`` (hidden).
"""
from __future__ import annotations

import csv
import io
import tarfile
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

from .utils import Timer, get_logger, resolve

log = get_logger(__name__)

TITLEABS_URL = "https://snap.stanford.edu/ogb/data/misc/ogbn_arxiv/titleabs.tsv.gz"


# =============================================================== raw dataset
@dataclass
class ArxivData:
    num_nodes: int
    edge_index: np.ndarray        # (2, E) int64, src cites dst
    year: np.ndarray              # (N,) int64
    x: np.ndarray                 # (N, 128) float32 OGB skip-gram features
    label: np.ndarray             # (N,) subject area, unused by recommenders
    paper_id: np.ndarray          # (N,) MAG paper id
    title: list[str] = field(default_factory=list)
    abstract: list[str] = field(default_factory=list)
    has_text: np.ndarray | None = None

    def text(self, i: int) -> str:
        return f"{self.title[i]}. {self.abstract[i]}".strip(" .")

    def texts(self) -> list[str]:
        return [self.text(i) for i in range(self.num_nodes)]

    def adjacency(self) -> sp.csr_matrix:
        """A[i, j] = 1 iff i cites j (full graph; use src/graph.py for leak-free views)."""
        return edges_to_csr(self.edge_index, self.num_nodes)


def edges_to_csr(edge_index: np.ndarray, n: int) -> sp.csr_matrix:
    ones = np.ones(edge_index.shape[1], dtype=np.float32)
    A = sp.csr_matrix((ones, (edge_index[0], edge_index[1])), shape=(n, n))
    A.sum_duplicates()
    A.data[:] = 1.0
    return A


def _read_titleabs(path: Path) -> pd.DataFrame:
    """titleabs.tsv.gz is a gzipped tar holding titleabs.tsv (paper id, title, abstract)."""
    def parse(fh) -> pd.DataFrame:
        return pd.read_csv(fh, sep="\t", header=None, names=["paper_id", "title", "abstract"],
                           quoting=csv.QUOTE_NONE, dtype={"paper_id": str}, keep_default_na=False,
                           on_bad_lines="warn", encoding="utf-8", encoding_errors="replace")
    try:
        with tarfile.open(path, "r:gz") as tf:
            member = next(m for m in tf.getmembers() if m.isfile())
            df = parse(io.TextIOWrapper(tf.extractfile(member), encoding="utf-8", errors="replace"))
    except tarfile.ReadError:  # plain gzipped TSV
        df = parse(path)
    df["paper_id"] = df["paper_id"].str.strip().str.strip("\x00")
    df = df[df["paper_id"].str.fullmatch(r"\d+")]
    df["paper_id"] = df["paper_id"].astype(np.int64)
    return df.drop_duplicates("paper_id")


def load_arxiv(data_root: str | Path = "data", with_text: bool = True) -> ArxivData:
    root = resolve(data_root)
    cache = root / "arxiv_cache.npz"
    text_cache = root / "arxiv_text.parquet"
    if cache.exists():
        z = np.load(cache)
        d = ArxivData(int(z["num_nodes"]), z["edge_index"], z["year"], z["x"], z["label"], z["paper_id"])
    else:
        from ogb.nodeproppred import PygNodePropPredDataset
        _orig = torch.load
        torch.load = lambda *a, **k: _orig(*a, **{**k, "weights_only": False})  # OGB pickles PyG Data
        try:
            g = PygNodePropPredDataset(name="ogbn-arxiv", root=str(root))[0]
        finally:
            torch.load = _orig
        mapping = pd.read_csv(root / "ogbn_arxiv" / "mapping" / "nodeidx2paperid.csv.gz")
        assert (mapping["node idx"].values == np.arange(g.num_nodes)).all()
        d = ArxivData(
            num_nodes=int(g.num_nodes),
            edge_index=g.edge_index.numpy().astype(np.int64),
            year=g.node_year.view(-1).numpy().astype(np.int64),
            x=g.x.numpy().astype(np.float32),
            label=g.y.view(-1).numpy().astype(np.int64),
            paper_id=mapping["paper id"].values.astype(np.int64),
        )
        np.savez(cache, num_nodes=d.num_nodes, edge_index=d.edge_index, year=d.year, x=d.x,
                 label=d.label, paper_id=d.paper_id)
    if with_text:
        if not text_cache.exists():
            tpath = root / "titleabs.tsv.gz"
            if not tpath.exists():
                log.info(f"Downloading {TITLEABS_URL}")
                urllib.request.urlretrieve(TITLEABS_URL, tpath)
            with Timer(log, "Joining titles/abstracts"):
                df = _read_titleabs(tpath)
                nodes = pd.DataFrame({"paper_id": d.paper_id})
                j = nodes.merge(df, on="paper_id", how="left")
                assert len(j) == d.num_nodes
                j["title"] = j["title"].fillna("")
                j["abstract"] = j["abstract"].fillna("")
                j[["paper_id", "title", "abstract"]].to_parquet(text_cache)
        j = pd.read_parquet(text_cache)
        d.title = j["title"].tolist()
        d.abstract = j["abstract"].tolist()
        d.has_text = ((j["title"].str.len() > 0) | (j["abstract"].str.len() > 0)).values
    return d


# =============================================================== query split
@dataclass
class QuerySet:
    """Queries in CSR form. history/targets are node ids; ``*_ptr`` index into them."""
    node: np.ndarray
    year: np.ndarray
    hist_ptr: np.ndarray
    hist_idx: np.ndarray
    tgt_ptr: np.ndarray
    tgt_idx: np.ndarray

    def __len__(self) -> int:
        return len(self.node)

    def history(self, i: int) -> np.ndarray:
        return self.hist_idx[self.hist_ptr[i]:self.hist_ptr[i + 1]]

    def targets(self, i: int) -> np.ndarray:
        return self.tgt_idx[self.tgt_ptr[i]:self.tgt_ptr[i + 1]]

    def subset(self, idx: np.ndarray) -> "QuerySet":
        idx = np.asarray(idx)
        hs = [self.history(i) for i in idx]
        ts = [self.targets(i) for i in idx]
        return QuerySet.from_lists(self.node[idx], self.year[idx], hs, ts)

    def history_matrix(self, n: int, idx: np.ndarray | None = None) -> sp.csr_matrix:
        """(Q, N) binary matrix of history items (rows optionally restricted to idx)."""
        qs = self if idx is None else self.subset(idx)
        data = np.ones(len(qs.hist_idx), dtype=np.float32)
        return sp.csr_matrix((data, qs.hist_idx, qs.hist_ptr), shape=(len(qs), n))

    @staticmethod
    def from_lists(node, year, hists, tgts) -> "QuerySet":
        hp = np.zeros(len(hists) + 1, dtype=np.int64)
        tp = np.zeros(len(tgts) + 1, dtype=np.int64)
        hp[1:] = np.cumsum([len(h) for h in hists])
        tp[1:] = np.cumsum([len(t) for t in tgts])
        cat = lambda xs: np.concatenate(xs).astype(np.int64) if len(xs) else np.zeros(0, np.int64)
        return QuerySet(np.asarray(node, np.int64), np.asarray(year, np.int64), hp, cat(hists), tp, cat(tgts))

    def to_dict(self, prefix: str) -> dict:
        return {f"{prefix}_{k}": getattr(self, k) for k in ["node", "year", "hist_ptr", "hist_idx", "tgt_ptr", "tgt_idx"]}

    @staticmethod
    def from_dict(z, prefix: str) -> "QuerySet":
        return QuerySet(*[np.asarray(z[f"{prefix}_{k}"]) for k in ["node", "year", "hist_ptr", "hist_idx", "tgt_ptr", "tgt_idx"]])


@dataclass
class Split:
    train: QuerySet
    val: QuerySet
    test: QuerySet
    config: dict

    def __getitem__(self, name: str) -> QuerySet:
        return getattr(self, name)


def split_filename(cfg: dict) -> str:
    s = cfg["split"]
    cap = lambda v: "all" if v is None else str(v)
    return (f"split_minrefs{s['min_refs']}_hist{s['history_frac']}_seed{s['seed']}"
            f"_val{cap(s['max_val_queries'])}_test{cap(s['max_test_queries'])}.npz")


def eligible_references(d: ArxivData) -> sp.csr_matrix:
    """CSR (N, N): row P holds P's references published strictly before P."""
    src, dst = d.edge_index
    keep = (d.year[dst] < d.year[src]) & (src != dst)
    return edges_to_csr(d.edge_index[:, keep], d.num_nodes)


def build_split(d: ArxivData, scfg: dict) -> Split:
    rng = np.random.default_rng(scfg["seed"])
    R = eligible_references(d)
    nref = np.diff(R.indptr)
    is_q = nref >= scfg["min_refs"]

    def make(mask: np.ndarray, cap: int | None) -> QuerySet:
        nodes = np.flatnonzero(mask & is_q)
        if cap is not None and len(nodes) > cap:
            nodes = np.sort(rng.choice(nodes, size=cap, replace=False))
        hists, tgts = [], []
        for p in nodes:
            refs = R.indices[R.indptr[p]:R.indptr[p + 1]]
            perm = rng.permutation(refs)
            h = int(round(scfg["history_frac"] * len(perm)))
            h = min(max(h, 1), len(perm) - 1)  # >= 1 history item and >= 1 target
            hists.append(np.sort(perm[:h]))
            tgts.append(np.sort(perm[h:]))
        return QuerySet.from_lists(nodes, d.year[nodes], hists, tgts)

    y = d.year
    train = make(y <= scfg["train_max_year"], None)
    val = make(y == scfg["val_year"], scfg["max_val_queries"])
    test = make(y == scfg["test_year"], scfg["max_test_queries"])
    return Split(train, val, test, dict(scfg))


def load_or_build_split(d: ArxivData, cfg: dict) -> Split:
    path = resolve(cfg["paths"]["cache"]) / "splits" / split_filename(cfg)
    if path.exists():
        z = np.load(path, allow_pickle=False)
        log.info(f"Loaded split {path.name}")
        return Split(*(QuerySet.from_dict(z, k) for k in ["train", "val", "test"]), dict(cfg["split"]))
    with Timer(log, f"Building split {path.name}"):
        s = build_split(d, cfg["split"])
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **s.train.to_dict("train"), **s.val.to_dict("val"), **s.test.to_dict("test"))
    log.info(f"Saved split to {path}")
    return s
