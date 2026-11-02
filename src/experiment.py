"""Shared experiment driver: fit -> tune query-time params on validation -> evaluate test.

A model spec (from configs/*.yaml) looks like
    name: tfidf                  # registry key in src/models/__init__.py
    label: tfidf                 # results go to results/runs/<label>/<variant>.json
    params: {...}                # constructor params
    seeds: [0]                   # >1 seed for trained models: mean +- std is reported
    variants:                    # optional; default is a single variant "default"
      combo: {params: {query_mode: combo}, tune: {alpha: [0.25, 0.5, 0.75]}}
Grid tuning maximises validation NDCG@20 (optionally on a subset of validation queries).
"""
from __future__ import annotations

import copy
import itertools
import time
from pathlib import Path

import numpy as np

from .data import Split
from .evaluate import Evaluator
from .graph import CitationGraph, build_eval_graph, build_train_graph
from .models import build_model
from .utils import get_logger, resolve, save_json, set_seed

log = get_logger(__name__)
TUNE_METRIC = "ndcg@20"


class Context:
    """Everything an experiment needs, built once and shared across models."""

    def __init__(self, d, split: Split, features, cfg: dict, evaluator: Evaluator):
        self.d, self.split, self.features, self.cfg, self.ev = d, split, features, cfg, evaluator
        self._graphs: dict[str, CitationGraph] = {}

    def graph(self, which: str) -> CitationGraph:
        if which not in self._graphs:
            self._graphs[which] = build_train_graph(self.d, self.split) if which == "train" \
                else build_eval_graph(self.d, self.split, which)
        return self._graphs[which]


def _grid(tune: dict | None) -> list[dict]:
    if not tune:
        return [{}]
    keys = list(tune)
    return [dict(zip(keys, vals)) for vals in itertools.product(*(tune[k] for k in keys))]


def run_spec(spec: dict, ctx: Context, out_dir: str | Path | None = None, splits=("val", "test"),
             save_topk: int = 10) -> dict:
    label = spec.get("label", spec["name"])
    variants = spec.get("variants") or {"default": {}}
    seeds = spec.get("seeds", [ctx.cfg["seed"]])
    tune_q = spec.get("tune_queries", ctx.cfg.get("tune_queries"))
    out_dir = resolve(out_dir or Path(ctx.cfg["paths"]["results"]) / "runs") / label
    val_qs = ctx.split.val
    tune_qs = val_qs if not tune_q or tune_q >= len(val_qs) else \
        val_qs.subset(np.sort(np.random.default_rng(0).choice(len(val_qs), tune_q, replace=False)))

    per_seed: dict[str, list] = {v: [] for v in variants}
    for seed in seeds:
        set_seed(seed)
        t0 = time.time()
        model = build_model(spec["name"], **{**spec.get("params", {}), "seed": seed})
        model.fit(ctx.graph("train"), ctx.features, ctx.split)
        fit_time = time.time() - t0
        log.info(f"[{label}] seed={seed} fit in {fit_time:.1f}s")

        chosen: dict[str, dict] = {}
        records: dict[str, dict] = {v: {"seed": seed, "fit_time_s": fit_time} for v in variants}
        for which in splits:
            qs = ctx.split[which]
            model.prepare(ctx.graph(which), which)
            for vname, vspec in variants.items():
                base = {**vspec.get("params", {})}
                if which == "val":
                    grid = _grid(vspec.get("tune"))
                    if len(grid) > 1:
                        results = []
                        for g in grid:
                            model.set_params(**base, **g)
                            m = ctx.ev.evaluate(model(tune_qs), tune_qs)["metrics"]
                            results.append({"params": g, TUNE_METRIC: m[TUNE_METRIC]})
                            log.info(f"[{label}/{vname}] tune {g}: {TUNE_METRIC}={m[TUNE_METRIC]:.4f}")
                        best = max(results, key=lambda r: r[TUNE_METRIC])
                        chosen[vname] = best["params"]
                        records[vname]["tuning"] = results
                    else:
                        chosen[vname] = grid[0]
                model.set_params(**base, **chosen.get(vname, {}))
                if which == "val":  # record the parameters actually used (after applying the choice)
                    records[vname]["params"] = {**copy.deepcopy(model.params), **base, **chosen[vname]}
                t1 = time.time()
                out = ctx.ev.evaluate(model(qs), qs, return_per_query=True, return_topk=save_topk)
                records[vname][which] = out["metrics"]
                records[vname][f"{which}_eval_time_s"] = time.time() - t1
                log.info(f"[{label}/{vname}] seed={seed} {which}: " +
                         " ".join(f"{k}={v:.4f}" for k, v in out["metrics"].items()))
                if which == "test":
                    out_dir.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(out_dir / f"{vname}_seed{seed}_test_perquery.npz",
                                        query=qs.node, topk=out["topk"],
                                        first_rank=np.array([r.min() for r in out["target_ranks"]]),
                                        target_rank=np.concatenate(out["target_ranks"]), target_ptr=qs.tgt_ptr,
                                        target_node=qs.tgt_idx,
                                        **{k.replace("@", "_at_"): v for k, v in out["per_query"].items()})
        for v in variants:
            if getattr(model, "history", None):
                records[v]["training_history"] = model.history
            per_seed[v].append(records[v])
        if hasattr(model, "close"):
            model.close()

    summary = {}
    for vname, recs in per_seed.items():
        s = {"model": spec["name"], "label": label, "variant": vname, "seeds": seeds, "runs": recs}
        for which in splits:
            keys = recs[0][which].keys()
            s[which] = {k: float(np.mean([r[which][k] for r in recs])) for k in keys}
            s[f"{which}_std"] = {k: float(np.std([r[which][k] for r in recs])) for k in keys}
        save_json(s, out_dir / f"{vname}.json")
        summary[vname] = s
    return summary
