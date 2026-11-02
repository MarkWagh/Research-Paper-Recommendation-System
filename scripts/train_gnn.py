"""Train and evaluate the GNN hybrid (fit -> early stopping on val NDCG@20 -> test).

    python scripts/train_gnn.py                                   # 3 seeds, val + test
    python scripts/train_gnn.py --dev --set gnn.params.epochs=6   # seed 0, training curve only
    python scripts/train_gnn.py --set gnn.label=gnn_bpr gnn.params.loss=bpr
Trained weights are cached in cache/gnn/ (keyed by hyper-parameters and seed).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_baselines import build_context
from src.experiment import run_spec
from src.models import build_model
from src.utils import ROOT, base_argparser, load_config, resolve, save_json


def main():
    ap = base_argparser(__doc__)
    ap.add_argument("--dev", action="store_true", help="fit seed 0 only and save the training curve")
    args = ap.parse_args()
    cfg = load_config([ROOT / "configs" / "gnn.yaml", *args.config], args.overrides)
    ctx = build_context(cfg, "train_gnn")
    spec = cfg["gnn"]
    if args.dev:
        m = build_model(spec["name"], **{**spec["params"], "seed": spec["seeds"][0]})
        m.fit(ctx.graph("train"), ctx.features, ctx.split)
        best = max(m.history, key=lambda h: h["val_ndcg@20"])
        save_json({"params": spec["params"], "history": m.history, "best": best},
                  resolve(cfg["paths"]["results"]) / "gnn_dev" / f"{spec['label']}.json")
        return
    run_spec(spec, ctx, cfg.get("out_dir"))


if __name__ == "__main__":
    main()
