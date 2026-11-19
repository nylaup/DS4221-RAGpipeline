"""Sentence embeddings with all-MiniLM-L6-v2 (384-d, unit-normalized)."""

from __future__ import annotations

import logging
import os
from functools import lru_cache

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("torch.distributed.elastic").setLevel(logging.ERROR)

import numpy as np

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_TAG = "all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def model():
    from sentence_transformers import SentenceTransformer
    from transformers.utils import logging as hf_logging
    hf_logging.set_verbosity_error()
    hf_logging.disable_progress_bar()
    return SentenceTransformer(MODEL_NAME, device="cpu")


def embed(texts: list[str], batch_size: int = 64, progress: bool = False) -> np.ndarray:
    return model().encode(texts, batch_size=batch_size, normalize_embeddings=True,
                          convert_to_numpy=True, show_progress_bar=progress).astype(np.float32)


def embed_one(text: str) -> np.ndarray:
    return embed([text])[0]


def count_tokens(texts: list[str]) -> list[int]:
    """Word-piece count before truncation (the model keeps the first 256)."""
    tok = model().tokenizer
    return [len(ids) for ids in tok(texts, add_special_tokens=True, truncation=False)["input_ids"]]


def max_seq_length() -> int:
    return model().max_seq_length
