"""Ablations, breakdowns and qualitative examples, reusing validation-tuned models.

    python scripts/run_ablations.py --parts breakdowns qualitative timemask histfrac minrefs
    python scripts/run_ablations.py --parts histfrac --models popularity/default ppr/default

Every model is rebuilt with the hyper-parameters chosen on validation (read from
results/runs/<label>/<variant>.json); nothing is re-tuned on test. Outputs go to
results/ablations/*.json and results/qualitative.json.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from scripts.run_baselines import build_context
from src.data import load_or_build_split
from src.evaluate import Evaluator
from src.experiment import Context
from src.models import build_model
from src.utils import ROOT, base_argparser, get_logger, load_config, resolve, save_json

log = get_logger("run_ablations")

DEFAULT_MODELS = ["popularity/default", "minilm/combo", "ppr/default", "fusion/text_graph",
                  "fusion/text_graph_pop", "gnn/default"]


def load_tuned(cfg, key: str, seed: int | None = None):
    """Rebuild <label>/<variant> with its validation-chosen parameters (fit not yet called)."""
    label, variant = key.split("/")
    path = resolve(cfg["paths"]["results"]) / "runs" / label / f"{variant}.json"
    if not path.exists():
        return None
    r = json.loads(path.read_text())
    params = dict(r["runs"][0]["params"])
    weights = {k: params.pop(k) for k in list(params) if k.startswith("w_")}
    params["seed"] = r["runs"][0]["seed"] if seed is None else seed
    m = build_model(r["model"], **params)
    if weights:
        m.set_params(**weights)
    return m


def fitted(ctx: Context, cfg, keys):
    out = {}
    for key in keys:
        m = load_tuned(cfg, key)
        if m is None:
            log.info(f"skip {key}: no results yet")
            continue
        m.fit(ctx.graph("train"), ctx.features, ctx.split)
        out[key] = m
    return out


def save_merged(obj: dict, path: Path, model_field: str | None = "models"):
    """Merge per-model entries into an existing results file so separate runs can add models."""
    if path.exists():
        old = json.loads(path.read_text())
        if model_field:
            obj[model_field] = {**old.get(model_field, {}), **obj[model_field]}
        else:
            obj = {**old, **obj}
    save_json(obj, path)


def bucketize(x, edges, labels):
    idx = np.digitize(x, edges[1:-1], right=False)
    return [labels[i] for i in idx]


# --------------------------------------------------------------------------- parts
def part_breakdowns(ctx, cfg, models):
    """Per-query metrics by query reference count / history connectivity; per-target hit
    rate by target popularity; popularity of what each model recommends."""
    qs = ctx.split.test
    g = ctx.graph("test")
    indeg = g.in_degree
    nref = np.diff(qs.hist_ptr) + np.diff(qs.tgt_ptr)
    hist_conn = np.array([np.mean(indeg[qs.history(i)]) for i in range(len(qs))])
    q_ref_bucket = bucketize(nref, [0, 8, 12, 20, 10 ** 9], ["5-7", "8-11", "12-19", "20+"])
    hc_edges = [-1, *np.quantile(hist_conn, [0.25, 0.5, 0.75]).tolist(), 10 ** 9]
    hc_labels = [f"Q{i + 1}" for i in range(4)]
    hc_bucket = bucketize(hist_conn, hc_edges, hc_labels)
    t_pop = indeg[qs.tgt_idx]
    tp_labels = ["0", "1-4", "5-19", "20-99", "100+"]
    t_bucket = bucketize(t_pop, [-1, 1, 5, 20, 100, 10 ** 9], tp_labels)
    pct = np.argsort(np.argsort(indeg)) / (len(indeg) - 1)          # popularity percentile
    top1 = indeg >= np.quantile(indeg, 0.99)

    res = {"bucket_defs": {
        "query_refs": {b: int(sum(1 for x in q_ref_bucket if x == b)) for b in ["5-7", "8-11", "12-19", "20+"]},
        "history_connectivity": {"edges_mean_indegree": hc_edges[1:-1],
                                 **{b: int(sum(1 for x in hc_bucket if x == b)) for b in hc_labels}},
        "target_popularity": {b: int(sum(1 for x in t_bucket if x == b)) for b in tp_labels}},
        "targets_popularity_reference": {"mean_percentile_of_targets": float(pct[qs.tgt_idx].mean()),
                                         "frac_targets_in_top1pct": float(top1[qs.tgt_idx].mean())},
        "models": {}}
    for key, m in models.items():
        m.prepare(g, "test")
        out = ctx.ev.evaluate(m(qs), qs, return_per_query=True, return_topk=10)
        pq, ranks, top = out["per_query"], np.concatenate(out["target_ranks"]), out["topk"]
        r = {"overall": out["metrics"], "by_query_refs": {}, "by_history_connectivity": {},
             "by_target_popularity": {}}
        for name, buckets, labels in (("by_query_refs", q_ref_bucket, ["5-7", "8-11", "12-19", "20+"]),
                                      ("by_history_connectivity", hc_bucket, hc_labels)):
            arr = np.array(buckets)
            for b in labels:
                sel = arr == b
                r[name][b] = {k: float(pq[k][sel].mean()) for k in ("recall@20", "ndcg@20", "mrr")}
        tb = np.array(t_bucket)
        for b in tp_labels:
            sel = tb == b
            r["by_target_popularity"][b] = {"hit@20": float((ranks[sel] <= 20).mean()),
                                            "hit@50": float((ranks[sel] <= 50).mean())}
        r["recommendation_popularity"] = {
            "mean_percentile_top10": float(pct[top].mean()),
            "frac_top10_in_top1pct": float(top1[top].mean()),
            "unique_items_top10": int(len(np.unique(top))),
            "median_indegree_top10": float(np.median(indeg[top]))}
        res["models"][key] = r
        log.info(f"[breakdowns] {key}: {r['recommendation_popularity']}")
    save_merged(res, resolve(cfg["paths"]["results"]) / "ablations" / "breakdowns.json")


def part_timemask(ctx, cfg, models):
    qs = ctx.split.test
    ev_off = Evaluator(ctx.d.year, cfg["eval"]["ks"], False, cfg["eval"]["tie_seed"], cfg["eval"]["batch_size"],
                       ctx.ev.device)
    res = {}
    for key, m in models.items():
        m.prepare(ctx.graph("test"), "test")
        res[key] = {"time_mask_on": ctx.ev.evaluate(m(qs), qs)["metrics"],
                    "time_mask_off": ev_off.evaluate(m(qs), qs)["metrics"]}
        log.info(f"[timemask] {key}: ndcg@20 on={res[key]['time_mask_on']['ndcg@20']:.4f} "
                 f"off={res[key]['time_mask_off']['ndcg@20']:.4f}")
    save_merged(res, resolve(cfg["paths"]["results"]) / "ablations" / "time_mask.json", model_field=None)


def _split_variant(ctx, cfg, models, name, overrides: dict, out_name):
    """Evaluate fixed (val-tuned at the default split) models on a different query split."""
    import copy
    c2 = copy.deepcopy(cfg)
    c2["split"].update(overrides)
    split = load_or_build_split(ctx.d, c2)
    ctx2 = Context(ctx.d, split, ctx.features, c2, ctx.ev)
    res = {"split": c2["split"], "num_test_queries": len(split.test), "models": {}}
    for key, m in models.items():
        # graph-dependent statistics are recomputed on this split's leak-free graphs; learned
        # models were trained on the default split's training graph, which only differs from
        # this one in held-out query edges that are absent from both.
        m.prepare(ctx2.graph("test"), "test")
        res["models"][key] = ctx2.ev.evaluate(m(split.test), split.test)["metrics"]
        log.info(f"[{name}] {overrides} {key}: ndcg@20={res['models'][key]['ndcg@20']:.4f}")
    save_merged(res, resolve(cfg["paths"]["results"]) / "ablations" / out_name)


def part_histfrac(ctx, cfg, models):
    for frac in (0.25, 0.75):
        _split_variant(ctx, cfg, models, "histfrac", {"history_frac": frac}, f"history_frac_{frac}.json")


def part_minrefs(ctx, cfg, models):
    for mr in (3, 10):
        _split_variant(ctx, cfg, models, "minrefs", {"min_refs": mr}, f"min_refs_{mr}.json")


def part_qualitative(ctx, cfg, models, n=5, seed=7):
    key = "fusion/text_graph_pop" if "fusion/text_graph_pop" in models else next(iter(models))
    m = models[key]
    qs = ctx.split.test
    rows = np.sort(np.random.default_rng(seed).choice(len(qs), n, replace=False))
    m.prepare(ctx.graph("test"), "test")
    top, _ = m.score(qs, k=10, evaluator=ctx.ev, rows=rows)
    d = ctx.d
    indeg = ctx.graph("test").in_degree
    out = []
    for r, t in zip(rows, top):
        tg = set(qs.targets(r).tolist())
        out.append({
            "query": {"node": int(qs.node[r]), "year": int(qs.year[r]), "title": d.title[qs.node[r]]},
            "history": [{"node": int(h), "year": int(d.year[h]), "title": d.title[h]} for h in qs.history(r)],
            "num_targets": len(tg),
            "recommendations": [{"rank": i + 1, "node": int(c), "year": int(d.year[c]), "title": d.title[c],
                                 "times_cited": int(indeg[c]), "hit": int(c) in tg} for i, c in enumerate(t)],
            "missed_targets": [{"node": int(x), "title": d.title[x]} for x in sorted(tg - set(t.tolist()))][:5],
        })
    save_json({"model": key, "examples": out}, resolve(cfg["paths"]["results"]) / "qualitative.json")
    log.info(f"[qualitative] wrote {n} examples using {key}")


PARTS = {"breakdowns": part_breakdowns, "timemask": part_timemask, "histfrac": part_histfrac,
         "minrefs": part_minrefs, "qualitative": part_qualitative}


def main():
    ap = base_argparser(__doc__)
    ap.add_argument("--parts", nargs="+", default=list(PARTS), choices=list(PARTS))
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    args = ap.parse_args()
    cfg = load_config([ROOT / "configs" / "baselines.yaml", *args.config], args.overrides)
    ctx = build_context(cfg, "run_ablations")
    models = fitted(ctx, cfg, args.models)
    for p in args.parts:
        PARTS[p](ctx, cfg, models)


if __name__ == "__main__":
    main()
