"""node2vec (PyG) embeddings; a query is the mean of its history embeddings (cold start).

node2vec is transductive, so it is (re)trained on the observed graph of each evaluation
split: the training graph for validation and the pre-2019 evaluation graph for test (i.e. a
refit on all data available at test time; neither contains any val/test query edge). Walks
use both edge directions. Candidates with no edges in the graph only have an untrained
random embedding, so they are back-filled with popularity below all connected candidates.
Embeddings are cached in cache/node2vec/.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np
import torch
from torch_geometric.nn import Node2Vec

from ..utils import Timer, get_logger, resolve
from .base import Recommender, history_mean

log = get_logger(__name__)


class Node2VecModel(Recommender):
    name = "node2vec"
    defaults = {"dim": 128, "walk_length": 20, "context_size": 10, "walks_per_node": 10, "p": 1.0, "q": 1.0,
                "num_negative_samples": 1, "epochs": 3, "batch_size": 256, "lr": 0.01, "num_workers": 0,
                "cache_dir": "cache/node2vec"}

    def train_embeddings(self, graph) -> np.ndarray:
        p = self.params
        seed = int(p.get("seed", 0))
        key = {k: p[k] for k in ("dim", "walk_length", "context_size", "walks_per_node", "p", "q",
                                 "num_negative_samples", "epochs", "batch_size", "lr")}
        h = hashlib.md5(json.dumps(key, sort_keys=True).encode()).hexdigest()[:8]
        path = resolve(p["cache_dir"]) / f"{graph.name}_E{graph.num_edges}_seed{seed}_{h}.npy"
        if path.exists():
            log.info(f"Loaded cached node2vec {path.name}")
            return np.load(path)
        torch.manual_seed(seed)
        np.random.seed(seed)
        ei = torch.as_tensor(graph.edge_index)
        ei = torch.cat([ei, ei.flip(0)], 1)  # walk along both directions
        model = Node2Vec(ei, embedding_dim=p["dim"], walk_length=p["walk_length"], context_size=p["context_size"],
                         walks_per_node=p["walks_per_node"], p=p["p"], q=p["q"],
                         num_negative_samples=p["num_negative_samples"], num_nodes=graph.num_nodes, sparse=True)
        loader = model.loader(batch_size=p["batch_size"], shuffle=True, num_workers=p["num_workers"])
        opt = torch.optim.SparseAdam(list(model.parameters()), lr=p["lr"])
        with Timer(log, f"node2vec on {graph} (seed {seed})"):
            for ep in range(p["epochs"]):
                model.train()
                tot, nb = 0.0, 0
                for pos_rw, neg_rw in loader:
                    opt.zero_grad()
                    loss = model.loss(pos_rw, neg_rw)
                    loss.backward()
                    opt.step()
                    tot += loss.item()
                    nb += 1
                log.info(f"node2vec epoch {ep + 1}/{p['epochs']} loss={tot / nb:.4f}")
        emb = model.embedding.weight.detach().numpy().astype(np.float32)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, emb)
        return emb

    def prepare(self, graph, which):
        super().prepare(graph, which)
        E = torch.as_tensor(self.train_embeddings(graph))
        self.E = torch.nn.functional.normalize(E, dim=1)
        connected = torch.as_tensor((graph.in_degree + graph.out_degree) > 0)
        pop = np.log1p(graph.in_degree)
        self.backfill = torch.as_tensor(pop / max(pop.max(), 1e-12), dtype=torch.float32) - 2.0
        self.connected = connected
        return self

    def score_batch(self, qs, rows):
        q = torch.nn.functional.normalize(history_mean(self.E, qs, rows), dim=1)
        s = q @ self.E.T                                   # cosine in [-1, 1]
        return torch.where(self.connected, s, self.backfill)  # isolated nodes: popularity, below all
