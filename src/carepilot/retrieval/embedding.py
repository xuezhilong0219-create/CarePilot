import os
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

import numpy as np
from tqdm.auto import tqdm


class Embedder(Protocol):
    name: str

    def encode(self, texts: Sequence[str]) -> np.ndarray: ...


class FastEmbedEmbedder:
    def __init__(
        self, model_name: str = "BAAI/bge-small-zh-v1.5",
        cache_dir: Path = Path("data/models"), batch_size: int = 32,
        threads: int = 4, show_progress: bool = True,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        # huggingface_hub 在导入时读取 endpoint，必须先设置再导入 fastembed。
        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
        from fastembed import TextEmbedding

        self.name = model_name
        self.batch_size = batch_size
        self.show_progress = show_progress
        supported = {entry["model"]: entry for entry in TextEmbedding.list_supported_models()}
        if model_name not in supported:
            raise ValueError(f"Unsupported embedding model: {model_name}")
        self.dimension = supported[model_name]["dim"]
        self._model = TextEmbedding(model_name=model_name, cache_dir=str(cache_dir), threads=threads)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)
        rows = list(tqdm(
            self._model.embed(texts, batch_size=self.batch_size),
            total=len(texts), desc="Embedding", unit="chunk",
            disable=not self.show_progress or len(texts) == 1,
        ))
        matrix = np.asarray(rows, dtype=np.float32)
        if matrix.shape != (len(texts), self.dimension) or not np.isfinite(matrix).all():
            raise ValueError("Embedding model returned invalid vectors")
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        if np.any(norms == 0):
            raise ValueError("Embedding model returned a zero vector")
        return matrix / norms
