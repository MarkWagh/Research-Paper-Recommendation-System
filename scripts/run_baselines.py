"""Run baseline models end to end (fit -> tune on val -> test) and save results.

    python scripts/run_baselines.py --models popularity
    python scripts/run_baselines.py --config configs/baselines.yaml --models tfidf bm25
    python scripts/run_baselines.py --set split.max_test_queries=null      # full test set
Model specs live under ``models:`` in configs/baselines.yaml (see src/experiment.py).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import load_arxiv, load_or_build_split
from src.evaluate import Evaluator
from src.experiment import Context, run_spec
from src.features import Features
from src.utils import ROOT, base_argparser, get_logger, load_config, resolve, set_seed


def build_context(cfg: dict, log_name: str) -> Context:
    log = get_logger(log_name, resolve(cfg["paths"]["results"]) / "logs" / f"{log_name}.log")
    set_seed(cfg["seed"])
    d = load_arxiv(cfg["paths"]["data_root"])
    split = load_or_build_split(d, cfg)
    ev = Evaluator.from_config(d.year, cfg)
    log.info(f"eval device={ev.device} | val={len(split.val)} test={len(split.test)} queries")
    return Context(d, split, Features(d, cfg), cfg, ev)


def main():
    ap = base_argparser(__doc__)
    ap.add_argument("--models", nargs="*", default=None, help="labels from the config's models list (default: all)")
    args = ap.parse_args()
    cfg = load_config([ROOT / "configs" / "baselines.yaml", *args.config], args.overrides)
    ctx = build_context(cfg, "run_baselines")
    specs = cfg["models"]
    if args.models:
        specs = [s for s in specs if s.get("label", s["name"]) in args.models or s["name"] in args.models]
    else:
        specs = [s for s in specs if not s.get("skip")]
    for spec in specs:
        run_spec(spec, ctx, cfg.get("out_dir"))


if __name__ == "__main__":
    main()
