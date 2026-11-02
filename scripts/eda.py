"""Milestone 1: dataset statistics, text coverage, split construction and summary.

    python scripts/eda.py [--set split.min_refs=5 ...]
Writes results/eda.json and results/figures/eda_*.png, and saves the split to cache/splits/.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.data import eligible_references, load_arxiv, load_or_build_split
from src.graph import build_eval_graph, build_train_graph
from src.plotting import style
from src.utils import base_argparser, get_logger, load_config, resolve, save_json, set_seed

log = get_logger("eda")


def describe(x: np.ndarray) -> dict:
    x = np.asarray(x)
    return {"mean": float(x.mean()), "median": float(np.median(x)), "p90": float(np.percentile(x, 90)),
            "p99": float(np.percentile(x, 99)), "max": int(x.max()), "frac_zero": float((x == 0).mean())}


def main():
    args = base_argparser(__doc__).parse_args()
    cfg = load_config(args.config, args.overrides)
    set_seed(cfg["seed"])
    d = load_arxiv(cfg["paths"]["data_root"])
    s, t = d.edge_index
    N, E = d.num_nodes, d.edge_index.shape[1]
    in_deg = np.bincount(t, minlength=N)
    out_deg = np.bincount(s, minlength=N)
    years, counts = np.unique(d.year, return_counts=True)
    R = eligible_references(d)
    n_elig = np.diff(R.indptr)

    split = load_or_build_split(d, cfg)
    g = build_train_graph(d, split)
    gt = build_eval_graph(d, split, "test")

    def qstats(qs):
        nh, nt = np.diff(qs.hist_ptr), np.diff(qs.tgt_ptr)
        return {"num_queries": len(qs), "history_size": describe(nh), "target_size": describe(nt),
                "total_targets": int(nt.sum())}

    stats = {
        "num_nodes": N, "num_edges": E,
        "nodes_with_text": int(d.has_text.sum()), "nodes_missing_text": int((~d.has_text).sum()),
        "papers_per_year": {int(y): int(c) for y, c in zip(years, counts)},
        "edge_direction": {"citing_newer": float((d.year[s] > d.year[t]).mean()),
                           "same_year": float((d.year[s] == d.year[t]).mean()),
                           "citing_older_(noise)": float((d.year[s] < d.year[t]).mean()),
                           "self_loops": int((s == t).sum())},
        "in_degree": describe(in_deg), "out_degree": describe(out_deg),
        "eligible_refs_fraction_of_all_refs": float(R.nnz / E),
        "eligible_refs_per_paper": describe(n_elig),
        "train_graph": {"num_edges": g.num_edges, "frac_of_all_edges": g.num_edges / E},
        "test_eval_graph": {"num_edges": gt.num_edges, "frac_of_all_edges": gt.num_edges / E},
        "split": {k: qstats(split[k]) for k in ("train", "val", "test")},
        "eligible_queries_before_cap": {
            "val": int(((d.year == cfg["split"]["val_year"]) & (n_elig >= cfg["split"]["min_refs"])).sum()),
            "test": int(((d.year == cfg["split"]["test_year"]) & (n_elig >= cfg["split"]["min_refs"])).sum())},
        "split_config": cfg["split"],
    }
    # text length
    tl = np.array([len(d.title[i].split()) + len(d.abstract[i].split()) for i in range(N)])
    stats["text_words"] = describe(tl)
    save_json(stats, resolve(cfg["paths"]["results"]) / "eda.json")

    # ---------------- figures
    plt = style()
    fig_dir = resolve(cfg["paths"]["results"]) / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 3.2))
    sel = years >= 2000
    yc = cfg["split"]
    colors = ["#2a78d6" if y <= yc["train_max_year"] else "#eb6834" if y == yc["val_year"]
              else "#1baf7a" if y == yc["test_year"] else "#b5b3ad" for y in years[sel]]
    ax.bar(years[sel], counts[sel], color=colors, width=0.8)
    ax.set_title("Papers per year (blue train era, orange val 2018, green test 2019)")
    ax.set_xlabel("year"); ax.set_ylabel("papers")
    fig.tight_layout(); fig.savefig(fig_dir / "eda_papers_per_year.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 3.6))
    for deg, name in ((in_deg, "in-degree (times cited)"), (out_deg, "out-degree (references)")):
        v, c = np.unique(deg[deg > 0], return_counts=True)
        ax.loglog(v, c / N, marker="o", markersize=3, linewidth=0, label=name)
    ax.set_title("Degree distributions (full directed graph)")
    ax.set_xlabel("degree"); ax.set_ylabel("fraction of papers"); ax.legend()
    fig.tight_layout(); fig.savefig(fig_dir / "eda_degree_distribution.png"); plt.close(fig)

    log.info(f"N={N} E={E} missing_text={stats['nodes_missing_text']} train_edges={g.num_edges} test_eval_edges={gt.num_edges}")
    for k in ("train", "val", "test"):
        q = stats["split"][k]
        log.info(f"{k}: {q['num_queries']} queries, mean history {q['history_size']['mean']:.1f}, "
                 f"mean targets {q['target_size']['mean']:.1f}")


if __name__ == "__main__":
    main()
