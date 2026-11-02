"""Collect results/runs/<label>/<variant>.json into tables.

    python scripts/make_tables.py [--runs results/runs] [--split test]
Writes results/tables/main_<split>.{csv,md}. Models with >1 seed show mean ± std.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.utils import base_argparser, load_config, resolve

FAMILY = {"popularity": "Popularity", "tfidf": "Content", "bm25": "Content", "ogb_features": "Content", "ogb_centered": "Content",
          "minilm": "Content", "specter2": "Content", "cocitation": "Graph", "ppr": "Graph", "node2vec": "Graph",
          "fusion": "Hybrid", "gnn": "Hybrid", "lightgcn": "Graph"}
FAMILY_ORDER = ["Popularity", "Content", "Graph", "Hybrid", "Other"]
METRICS = ["recall@10", "recall@20", "recall@50", "ndcg@10", "ndcg@20", "ndcg@50", "mrr"]


def collect(runs_dir: Path, split: str) -> pd.DataFrame:
    rows = []
    for f in sorted(runs_dir.glob("*/*.json")):
        r = json.loads(f.read_text())
        if split not in r:
            continue
        label = r["label"]
        row = {"family": FAMILY.get(label.split("_seed")[0], FAMILY.get(r["model"], "Other")),
               "model": label, "variant": r["variant"], "seeds": len(r["seeds"])}
        params = r["runs"][0].get("params", {})
        tuned = {k: v for k, v in params.items() if k in ("alpha", "w_content", "w_graph", "w_pop")}
        row["chosen"] = ", ".join(f"{k}={v}" for k, v in tuned.items()) if r["runs"][0].get("tuning") else ""
        for m in METRICS:
            row[m] = r[split][m]
            row[m + "_std"] = r.get(f"{split}_std", {}).get(m, 0.0)
        rows.append(row)
    df = pd.DataFrame(rows)
    if len(df):
        df["_f"] = df["family"].map(lambda x: FAMILY_ORDER.index(x) if x in FAMILY_ORDER else 99)
        df = df.sort_values(["_f", "model", "variant"]).drop(columns="_f").reset_index(drop=True)
    return df


def to_markdown(df: pd.DataFrame, metrics=METRICS) -> str:
    best = {m: df[m].max() for m in metrics}
    head = "| Family | Model | Variant | " + " | ".join(m.replace("recall", "R").replace("ndcg", "N").upper()
                                                      for m in metrics) + " |"
    lines = [head, "|" + "---|" * (3 + len(metrics))]
    for _, r in df.iterrows():
        cells = []
        for m in metrics:
            s = f"{r[m]:.4f}"
            if r["seeds"] > 1:
                s += f" ± {r[m + '_std']:.4f}"
            if abs(r[m] - best[m]) < 1e-12:
                s = f"**{s}**"
            cells.append(s)
        var = r["variant"] + (f" ({r['chosen']})" if r.get("chosen") else "")
        lines.append(f"| {r['family']} | {r['model']} | {var} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    ap = base_argparser(__doc__)
    ap.add_argument("--runs", default=None)
    ap.add_argument("--split", default="test")
    ap.add_argument("--name", default="main")
    args = ap.parse_args()
    cfg = load_config(args.config, args.overrides)
    runs = resolve(args.runs or Path(cfg["paths"]["results"]) / "runs")
    df = collect(runs, args.split)
    out = resolve(cfg["paths"]["results"]) / "tables"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / f"{args.name}_{args.split}.csv", index=False)
    md = to_markdown(df)
    (out / f"{args.name}_{args.split}.md").write_text(md + "\n")
    print(md)


if __name__ == "__main__":
    main()
