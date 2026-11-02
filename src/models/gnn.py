"""Inductive directional GraphSAGE hybrid (text features + citation graph), trained for
link prediction with neighbour sampling.

Item tower. Every paper v gets embeddings h^0..h^L over the citation graph:
    h^0_v = W_in x_v                                    (x = SPECTER2 text embedding)
    h^l_v = h^{l-1}_v + act(W_self h^{l-1}_v + W_ref mean_{u in refs(v)} h^{l-1}_u
                                             + W_cit mean_{u in citers(v)} h^{l-1}_u)
    z_v   = normalise(W_out h^L_v)
"refs" and "citers" have separate weights, so edge direction is preserved.

Query tower (cold start). A query paper P has no edges. It is treated as a temporary node
whose only neighbours are its history items (as references) and which has no citers:
    g^0 = W_in x_P,   g^l = g^{l-1} + act(W_self g^{l-1} + W_ref mean_{h in H} h^{l-1}_h)
    q   = normalise(W_out g^L);   score(P, c) = <q, z_c> / tau
Target edges are never used.

Training mimics evaluation exactly. In each epoch a random fraction of training-era papers
becomes pseudo-queries. ALL their outgoing edges are removed from the message-passing graph,
and their eligible references are split into history and targets. A pseudo-query is
therefore never in the graph, and its targets never appear in its own input. Mini-batches
sample item neighbourhoods with PyG's NeighborLoader on that masked graph. The loss is BPR
(or BCE, or sampled softmax) against shared random time-valid negatives plus optional hard negatives: text-similar,
older papers that the query does not cite. Early stopping uses validation NDCG@20.
"""
from __future__ import annotations

import copy
import hashlib
import json
import time

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from torch_geometric.utils import scatter

from ..evaluate import Evaluator
from ..graph import CitationGraph
from ..utils import get_device, get_logger, resolve
from .base import Recommender

log = get_logger(__name__)


# =============================================================== network
class DirSAGELayer(nn.Module):
    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.self_lin = nn.Linear(dim, dim)
        self.ref_lin = nn.Linear(dim, dim, bias=False)
        self.cit_lin = nn.Linear(dim, dim, bias=False)
        self.norm = nn.LayerNorm(dim)
        self.dropout = dropout

    def combine(self, h, agg_ref, agg_cit):
        m = self.self_lin(h) + self.ref_lin(agg_ref)
        if agg_cit is not None:
            m = m + self.cit_lin(agg_cit)
        return h + F.dropout(F.gelu(self.norm(m)), self.dropout, self.training)


class Encoder(nn.Module):
    def __init__(self, din: int, hidden: int, out: int, num_layers: int, dropout: float,
                 num_nodes: int = 0, text_skip: bool = False):
        super().__init__()
        self.inp = nn.Linear(din, hidden)
        self.layers = nn.ModuleList([DirSAGELayer(hidden, dropout) for _ in range(num_layers)])
        self.out = nn.Linear(hidden, out)
        # optional learnable node-ID embeddings (zero-init: unseen nodes stay text-only)
        self.ids = nn.Embedding(num_nodes, hidden) if num_nodes else None
        if self.ids is not None:
            nn.init.zeros_(self.ids.weight)
        # optional learnable weight on the raw text cosine in the final score
        self.text_w = nn.Parameter(torch.tensor(1.0)) if text_skip else None

    def h0(self, x, n_id=None):
        h = self.inp(x)
        if self.ids is not None and n_id is not None:
            h = h + self.ids(n_id)
        return h

    def score(self, q, z, xq=None, xz=None):
        """(Bq, d) x (Bz, d) -> (Bq, Bz) logits before temperature."""
        s = q @ z.T
        if self.text_w is not None:
            s = s + self.text_w * (xq @ xz.T)
        return s

    # -- items on a (sub)graph. ``agg(l, h) -> (mean over refs, mean over citers, n_out)``:
    #    layer l only needs outputs for the first n_out nodes (hop-ordered sampled subgraphs)
    def items(self, x, agg, n_id=None) -> list[torch.Tensor]:
        hs = [self.h0(x, n_id)]
        for l, layer in enumerate(self.layers):
            h = hs[-1]
            a_ref, a_cit, n_out = agg(l, h)
            hs.append(layer.combine(h[:n_out], a_ref, a_cit))
        return hs

    def project(self, h):
        return F.normalize(self.out(h), dim=-1)

    # -- cold-start queries: x_q (B, din); hist_h[l] (M, hidden) layer-l embeddings of the
    #    history items; seg (M,) query index of each history item
    def queries(self, x_q, hist_hs: list[torch.Tensor], seg: torch.Tensor):
        B = x_q.shape[0]
        g = self.inp(x_q)
        for l, layer in enumerate(self.layers):
            agg = scatter(hist_hs[l], seg, 0, dim_size=B, reduce="mean")
            g = layer.combine(g, agg, None)
        return self.project(g)


def scatter_agg(ei_ref, ei_cit, n):
    """Mean aggregation over message edges, all nodes at every layer."""
    return lambda l, h: (scatter(h[ei_ref[0]], ei_ref[1], 0, dim_size=n, reduce="mean"),
                         scatter(h[ei_cit[0]], ei_cit[1], 0, dim_size=n, reduce="mean"), n)


def trimmed_agg(ei, et, nodes_per_hop, edges_per_hop, num_layers):
    """Mean aggregation on a NeighborLoader subgraph (nodes/edges ordered by hop). Layer l
    only computes nodes within L-1-l hops of the seeds, using edges into those nodes."""
    def agg(l, h):
        keep_hops = num_layers - l               # hops 0..keep_hops-1 still needed downstream
        n_out = int(sum(nodes_per_hop[:keep_hops]))
        n_e = int(sum(edges_per_hop[:keep_hops]))
        e, t = ei[:, :n_e], et[:n_e]
        r, c = e[:, t == 0], e[:, t == 1]
        return (scatter(h[r[0]], r[1], 0, dim_size=n_out, reduce="mean"),
                scatter(h[c[0]], c[1], 0, dim_size=n_out, reduce="mean"), n_out)
    return agg


def sparse_agg(A: sp.csr_matrix):
    """Mean aggregation over full neighbourhoods via row-normalised sparse matrices (inference)."""
    def mean_op(M):
        deg = np.asarray(M.sum(1)).ravel()
        M = (sp.diags(1.0 / np.maximum(deg, 1)) @ M).tocsr().astype(np.float32)
        return torch.sparse_csr_tensor(torch.as_tensor(M.indptr, dtype=torch.int64),
                                       torch.as_tensor(M.indices, dtype=torch.int64),
                                       torch.as_tensor(M.data), M.shape)
    R, C = mean_op(A), mean_op(A.T.tocsr())   # R: my references, C: my citers
    return lambda l, h: (R @ h, C @ h, h.shape[0])


def mask_pseudo_queries(A: sp.csr_matrix, Q: np.ndarray) -> sp.csr_matrix:
    """Message graph for one epoch: drop ALL outgoing edges of the pseudo-queries Q, so a
    pseudo-query is absent from the graph exactly like a real validation/test query."""
    keep = np.ones(A.shape[0], dtype=np.float32)
    keep[Q] = 0.0
    M = (sp.diags(keep) @ A).tocsr()
    M.eliminate_zeros()
    return M


# =============================================================== recommender
class GNNHybrid(Recommender):
    name = "gnn"
    defaults = {"feature": "specter2", "hidden": 256, "out_dim": 256, "num_layers": 2, "fanout": [10, 5],
                "dropout": 0.1, "lr": 1e-3, "weight_decay": 0.0, "epochs": 30, "patience": 4,
                "query_frac": 0.5, "history_frac": 0.5, "min_train_refs": 2, "batch_queries": 256,
                "num_shared_neg": 512, "hard_neg_per_query": 0, "hard_k": 50, "loss": "bpr", "tau": 0.1,
                "eval_queries": 1000, "max_steps_per_epoch": None, "cache_dir": "cache/gnn",
                "train_device": "auto", "save_history": True, "id_emb": False, "text_skip": False}

    # ------------------------------------------------------------ setup helpers
    def _rel_edges(self, A: sp.csr_matrix) -> tuple[np.ndarray, np.ndarray]:
        """Message edges for NeighborLoader: type 0 = ref -> citing paper (aggregate my refs),
        type 1 = citer -> cited paper (aggregate my citers)."""
        coo = A.tocoo()
        ref = np.vstack([coo.col, coo.row])
        cit = np.vstack([coo.row, coo.col])
        ei = np.hstack([ref, cit]).astype(np.int64)
        et = np.concatenate([np.zeros(ref.shape[1]), np.ones(cit.shape[1])]).astype(np.int64)
        return ei, et

    def _hard_negatives(self, X: torch.Tensor, nodes: np.ndarray, year: np.ndarray, A: sp.csr_matrix) -> np.ndarray:
        """Top-k most text-similar strictly older papers that the node does not cite."""
        k = self.params["hard_k"]
        yr = torch.as_tensor(year)
        out = np.full((len(nodes), k), -1, dtype=np.int64)
        for s in range(0, len(nodes), 2048):
            b = nodes[s:s + 2048]
            sim = X[torch.as_tensor(b)] @ X.T
            sim[yr.unsqueeze(0) >= yr[torch.as_tensor(b)].unsqueeze(1)] = -2
            R = A[b].tocoo()
            sim[torch.as_tensor(R.row), torch.as_tensor(R.col)] = -2
            v, i = sim.topk(k, dim=1)
            i[v <= -2] = -1
            out[s:s + 2048] = i.numpy()
        return out

    def _cache_key(self) -> str:
        keep = {k: v for k, v in self.params.items() if k not in ("device", "train_device", "save_history",
                                                                   "cache_dir", "eval_queries")}
        return hashlib.md5(json.dumps(keep, sort_keys=True, default=str).encode()).hexdigest()[:10]

    # ------------------------------------------------------------ training
    def fit(self, train_graph: CitationGraph, features, split):
        super().fit(train_graph, features, split)
        p = self.params
        seed = int(p.get("seed", 0))
        self.year = features.year
        self.X = torch.as_tensor(features[p["feature"]], dtype=torch.float32)
        self.dev = get_device(p["train_device"])
        self.net = Encoder(self.X.shape[1], p["hidden"], p["out_dim"], p["num_layers"], p["dropout"],
                           num_nodes=self.X.shape[0] if p["id_emb"] else 0, text_skip=p["text_skip"])
        ckpt = resolve(p["cache_dir"]) / f"{self._cache_key()}_seed{seed}.pt"
        if ckpt.exists():
            st = torch.load(ckpt, map_location="cpu", weights_only=False)
            self.net.load_state_dict(st["state"])
            self.history = st.get("history", [])
            log.info(f"Loaded cached GNN {ckpt.name} (best epoch {st.get('best_epoch')})")
            return self
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        self.net.to(self.dev)
        opt = torch.optim.Adam(self.net.parameters(), lr=p["lr"], weight_decay=p["weight_decay"])

        A = train_graph.A
        n = train_graph.num_nodes
        # eligible references: strictly older papers (mirrors the evaluation task)
        coo = A.tocoo()
        older = self.year[coo.col] < self.year[coo.row]
        E = sp.csr_matrix((np.ones(older.sum(), np.float32), (coo.row[older], coo.col[older])), shape=A.shape)
        nref = np.diff(E.indptr)
        cand_q = np.flatnonzero(nref >= p["min_train_refs"])
        train_items = np.flatnonzero(self.year <= train_graph.max_citing_year)
        log.info(f"GNN: {len(cand_q)} candidate pseudo-queries, device={self.dev}")
        hard = None
        if p["hard_neg_per_query"] > 0:
            hard = self._hard_negatives(self.X, cand_q, self.year, A)
            hard_row = np.full(n, -1)
            hard_row[cand_q] = np.arange(len(cand_q))

        # validation for early stopping
        val = split.val
        if p["eval_queries"] and p["eval_queries"] < len(val):
            val = val.subset(np.sort(np.random.default_rng(0).choice(len(val), p["eval_queries"], replace=False)))
        ev = Evaluator(self.year, ks=(20,), device="cpu")

        best, best_state, bad, self.history = -1.0, None, 0, []
        for epoch in range(1, p["epochs"] + 1):
            t0 = time.time()
            self.net.train()
            # ---- per-epoch pseudo-queries and masked message graph
            Q = rng.choice(cand_q, size=int(p["query_frac"] * len(cand_q)), replace=False)
            M = mask_pseudo_queries(A, Q)
            ei, et = self._rel_edges(M)
            data = Data(edge_index=torch.as_tensor(ei), edge_type=torch.as_tensor(et), num_nodes=n)
            loader = NeighborLoader(data, num_neighbors=p["fanout"], batch_size=1, input_nodes=None)
            hists, tgts = [], []
            for u in Q:
                refs = rng.permutation(E.indices[E.indptr[u]:E.indptr[u + 1]])
                h = min(max(int(round(p["history_frac"] * len(refs))), 1), len(refs) - 1)
                hists.append(refs[:h])
                tgts.append(refs[h:])
            order = rng.permutation(len(Q))
            tot, steps = 0.0, 0
            for s in range(0, len(Q), p["batch_queries"]):
                bi = order[s:s + p["batch_queries"]]
                loss = self._step(bi, Q, hists, tgts, loader, train_items, hard, hard_row if hard is not None else None, rng)
                opt.zero_grad()
                loss.backward()
                opt.step()
                tot += loss.item()
                steps += 1
                if p["max_steps_per_epoch"] and steps >= p["max_steps_per_epoch"]:
                    break
            # ---- validation (full-graph inference on the training graph = val eval graph)
            self.prepare(train_graph, "val")
            ndcg = ev.evaluate(self(val), val)["metrics"]["ndcg@20"]
            self.history.append({"epoch": epoch, "loss": tot / max(steps, 1), "val_ndcg@20": ndcg,
                                 "time_s": time.time() - t0})
            log.info(f"GNN epoch {epoch}: loss={tot / max(steps, 1):.4f} val ndcg@20={ndcg:.4f} "
                     f"({time.time() - t0:.0f}s, {steps} steps)")
            if ndcg > best:
                best, bad = ndcg, 0
                best_state = copy.deepcopy({k: v.cpu() for k, v in self.net.state_dict().items()})
                best_epoch = epoch
            else:
                bad += 1
                if bad >= p["patience"]:
                    log.info(f"Early stopping at epoch {epoch} (best {best_epoch}: {best:.4f})")
                    break
        self.net.load_state_dict(best_state)
        self.net.cpu()
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state": best_state, "history": self.history, "best_epoch": best_epoch, "params": p}, ckpt)
        return self

    def _step(self, bi, Q, hists, tgts, loader, train_items, hard, hard_row, rng):
        p = self.params
        dev = self.dev
        # pairs (query, target)
        q_nodes = Q[bi]
        h_list = [hists[i] for i in bi]
        t_list = [tgts[i] for i in bi]
        pair_q = np.repeat(np.arange(len(bi)), [len(t) for t in t_list])
        pair_t = np.concatenate(t_list)
        neg = rng.choice(train_items, size=p["num_shared_neg"], replace=False)
        hard_ids = None
        if hard is not None:
            k = p["hard_neg_per_query"]
            cols = rng.integers(0, hard.shape[1], size=(len(bi), k))
            hard_ids = hard[hard_row[q_nodes][:, None], cols]                 # (B, k), -1 = none
        h_flat = np.concatenate(h_list)
        seeds = np.unique(np.concatenate([pair_t, neg, h_flat] + ([hard_ids[hard_ids >= 0]] if hard_ids is not None else [])))
        # ---- sample the items' neighbourhoods on the masked graph
        out = loader.collate_fn(torch.as_tensor(seeds))
        batch = out if loader.filter_per_worker else loader.filter_fn(out)
        n_id = batch.n_id
        assert torch.equal(n_id[:len(seeds)], torch.as_tensor(seeds))
        loc = {int(g): i for i, g in enumerate(seeds)}
        x = self.X[n_id].to(dev)
        ei, et = batch.edge_index.to(dev), batch.edge_type.to(dev)
        hs = self.net.items(x, trimmed_agg(ei, et, batch.num_sampled_nodes, batch.num_sampled_edges,
                                           len(self.net.layers)), n_id=n_id.to(dev))
        z = self.net.project(hs[-1][:len(seeds)])
        # ---- cold-start query embeddings from text + history items' layer embeddings
        h_loc = torch.as_tensor([loc[int(h)] for h in h_flat], device=dev)
        seg = torch.as_tensor(np.repeat(np.arange(len(bi)), [len(h) for h in h_list]), device=dev)
        xq = self.X[torch.as_tensor(q_nodes)].to(dev)
        xs = x[:len(seeds)]
        q = self.net.queries(xq, [h[h_loc] for h in hs[:-1]], seg)
        tw = self.net.text_w
        pair = lambda qi, zi: (q[qi] * z[zi]).sum(-1) + (tw * (xq[qi] * xs[zi]).sum(-1) if tw is not None else 0)
        # ---- scores
        tau = p["tau"]
        t_loc = torch.as_tensor([loc[int(t)] for t in pair_t], device=dev)
        pq = torch.as_tensor(pair_q, device=dev)
        s_pos = pair(pq, t_loc) / tau                                              # (P,)
        n_loc = torch.as_tensor([loc[int(v)] for v in neg], device=dev)
        s_neg = self.net.score(q, z[n_loc], xq, xs[n_loc])[pq] / tau               # (P, S)
        yr = torch.as_tensor(self.year)
        valid = (yr[torch.as_tensor(neg)].unsqueeze(0) < yr[torch.as_tensor(q_nodes)][pair_q].unsqueeze(1)).to(dev)
        if hard_ids is not None:
            hid = torch.as_tensor(hard_ids)
            hl = torch.as_tensor([loc.get(int(v), 0) for v in hid.view(-1)], device=dev).view(hid.shape)
            s_hard = pair(torch.arange(len(bi), device=dev).unsqueeze(1).expand_as(hl), hl)[pq] / tau  # (P, k)
            s_neg = torch.cat([s_neg, s_hard], 1)
            valid = torch.cat([valid, (hid >= 0)[torch.as_tensor(pair_q)].to(dev)], 1)
        if p["loss"] == "bpr":
            l = F.softplus(s_neg - s_pos.unsqueeze(1))
            return (l * valid).sum() / valid.sum().clamp(min=1)
        if p["loss"] == "softmax":  # sampled softmax / InfoNCE over the same negatives
            logits = torch.cat([s_pos.unsqueeze(1), s_neg.masked_fill(~valid, -torch.inf)], 1)
            return F.cross_entropy(logits, torch.zeros(len(logits), dtype=torch.long, device=dev))
        # BCE: positives vs every valid negative
        lp = F.binary_cross_entropy_with_logits(s_pos, torch.ones_like(s_pos))
        ln = F.binary_cross_entropy_with_logits(s_neg, torch.zeros_like(s_neg), reduction="none")
        return lp + (ln * valid).sum() / valid.sum().clamp(min=1)

    # ------------------------------------------------------------ inference
    @torch.no_grad()
    def prepare(self, graph: CitationGraph, which: str):
        super().prepare(graph, which)
        was_training = self.net.training
        self.net.eval()
        dev = next(self.net.parameters()).device
        self.net.cpu()
        self.hs = self.net.items(self.X, sparse_agg(graph.A), n_id=torch.arange(self.X.shape[0]))
        self.Z = self.net.project(self.hs[-1])
        self.net.to(dev)
        self.net.train(was_training)
        return self

    @torch.no_grad()
    def score_batch(self, qs, rows):
        was_training = self.net.training
        self.net.eval()
        dev = next(self.net.parameters()).device
        self.net.cpu()
        sizes = np.diff(qs.hist_ptr)[rows]
        hidx = torch.as_tensor(np.concatenate([qs.history(i) for i in rows]))
        seg = torch.as_tensor(np.repeat(np.arange(len(rows)), sizes))
        q = self.net.queries(self.X[torch.as_tensor(qs.node[rows])], [h[hidx] for h in self.hs[:-1]], seg)
        self.net.to(dev)
        self.net.train(was_training)
        return self.net.score(q, self.Z, self.X[torch.as_tensor(qs.node[rows])], self.X)
