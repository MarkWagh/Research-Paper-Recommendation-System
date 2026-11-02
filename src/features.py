"""Node feature matrices, with on-disk caching of expensive text embeddings.

Available feature names:
  ogb       - the 128-d OGB skip-gram features (provided)
  ogb_centered - the same, mean-centred before normalising (raw vectors are highly
                 anisotropic: the mean unit vector has norm ~0.88, which washes out cosine)
  specter2  - allenai/specter2_base, CLS pooling, "title [SEP] abstract", max 512 tokens
  minilm    - sentence-transformers/all-MiniLM-L6-v2, mean pooling (fast option)
Dense features are returned L2-normalised float32 arrays.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .data import ArxivData
from .utils import Timer, get_device, get_logger, resolve

log = get_logger(__name__)

TEXT_MODELS = {
    "specter2": {"hf": "allenai/specter2_base", "pooling": "cls", "max_len": 512, "sep": True},
    "minilm": {"hf": "sentence-transformers/all-MiniLM-L6-v2", "pooling": "mean", "max_len": 256, "sep": False},
}


def l2norm(x: np.ndarray) -> np.ndarray:
    return (x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)).astype(np.float32)


def _encode(texts: list[str], spec: dict, device: torch.device, batch_size: int, ckpt: Path) -> np.ndarray:
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(spec["hf"])
    model = AutoModel.from_pretrained(spec["hf"]).to(device).eval()
    if device.type in ("cuda", "mps"):
        model = model.half()  # halves activation memory; embeddings are re-normalised in fp32
    # Sort by length so batches have little padding; restore the order at the end.
    order = np.argsort([-len(t) for t in texts], kind="stable")
    n = len(texts)
    dim = model.config.hidden_size
    out = np.zeros((n, dim), dtype=np.float32)
    done = 0
    if ckpt.exists():  # resume an interrupted run
        z = np.load(ckpt)
        done = int(z["done"])
        out[order[:done]] = z["emb"][:done]
        log.info(f"Resuming {spec['hf']} from {done}/{n}")
    t_save = done
    with torch.inference_mode():
        for s in range(done, n, batch_size):
            idx = order[s:s + batch_size]
            enc = tok([texts[i] for i in idx], padding=True, truncation=True, max_length=spec["max_len"],
                      return_tensors="pt").to(device)
            h = model(**enc).last_hidden_state
            if spec["pooling"] == "cls":
                e = h[:, 0]
            else:
                m = enc["attention_mask"].unsqueeze(-1).to(h.dtype)
                e = (h * m).sum(1) / m.sum(1).clamp(min=1)
            out[idx] = e.float().cpu().numpy()
            done = s + len(idx)
            step = s // batch_size
            if device.type == "mps" and step % 50 == 0:
                torch.mps.empty_cache()  # the MPS caching allocator otherwise grows into swap
            if step % 100 == 0:
                log.info(f"{spec['hf']}: {done}/{n}")
            if done - t_save >= 10000 or done == n:
                np.savez(ckpt, emb=out[order[:done]], done=done)
                t_save = done
                log.info(f"{spec['hf']}: {done}/{n}")
    return out


def text_embeddings(d: ArxivData, name: str, cache_dir: str | Path = "cache", device: str = "auto",
                    batch_size: int = 64) -> np.ndarray:
    path = resolve(cache_dir) / "embeddings" / f"{name}.npy"
    if path.exists():
        return np.load(path)
    spec = TEXT_MODELS[name]
    path.parent.mkdir(parents=True, exist_ok=True)
    sep = " [SEP] " if spec["sep"] else ". "
    texts = [f"{d.title[i]}{sep}{d.abstract[i]}" for i in range(d.num_nodes)]
    with Timer(log, f"Encoding {d.num_nodes} papers with {spec['hf']}"):
        emb = _encode(texts, spec, get_device(device), batch_size, path.with_suffix(".partial.npz"))
    emb = l2norm(emb)
    np.save(path, emb.astype(np.float16))
    path.with_suffix(".partial.npz").unlink(missing_ok=True)
    return emb


class Features:
    """Lazy, cached access to node feature matrices by name."""

    def __init__(self, d: ArxivData, cfg: dict):
        self.d, self.cfg = d, cfg
        self._cache: dict[str, np.ndarray] = {}

    def __getitem__(self, name: str) -> np.ndarray:
        if name not in self._cache:
            if name == "ogb":
                self._cache[name] = l2norm(self.d.x)
            elif name == "ogb_centered":
                self._cache[name] = l2norm(self.d.x - self.d.x.mean(0))
            elif name in TEXT_MODELS:
                emb = text_embeddings(self.d, name, self.cfg["paths"]["cache"], self.cfg.get("device", "auto"))
                self._cache[name] = l2norm(emb.astype(np.float32))
            else:
                raise KeyError(f"unknown feature {name!r}")
        return self._cache[name]

    @property
    def year(self) -> np.ndarray:
        return self.d.year
