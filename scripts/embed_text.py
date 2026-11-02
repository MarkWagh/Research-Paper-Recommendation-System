"""Compute and cache text embeddings for every paper (resumable).

    python scripts/embed_text.py --models specter2 minilm [--set device=cpu]
"""
import os
import sys
from pathlib import Path

# Cap the MPS allocator (Apple GPU shares RAM with the CPU); ignored on other backends.
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.5")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.4")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import load_arxiv
from src.features import text_embeddings
from src.utils import base_argparser, get_logger, load_config, set_seed

log = get_logger("embed_text")

if __name__ == "__main__":
    ap = base_argparser(__doc__)
    ap.add_argument("--models", nargs="+", default=["specter2"])
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()
    cfg = load_config(args.config, args.overrides)
    set_seed(cfg["seed"])
    d = load_arxiv(cfg["paths"]["data_root"])
    for m in args.models:
        e = text_embeddings(d, m, cfg["paths"]["cache"], cfg.get("device", "auto"), args.batch_size)
        log.info(f"{m}: {e.shape}")
