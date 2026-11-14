import hashlib
from collections.abc import Sequence

import numpy as np
import pytest

from carepilot.retrieval.base import Chunk


class HashEmbedder:
    name = "test/sha256"

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        rows = [np.frombuffer(hashlib.sha256(text.encode()).digest(), dtype=np.uint8).astype(np.float32) - 127.5 for text in texts]
        matrix = np.asarray(rows, dtype=np.float32).reshape(len(texts), 32)
        return matrix / np.linalg.norm(matrix, axis=1, keepdims=True) if len(texts) else matrix


@pytest.fixture
def embedder() -> HashEmbedder:
    return HashEmbedder()


@pytest.fixture
def chunks() -> list[Chunk]:
    return [Chunk(f"doc{i}#0", f"标题{i}", f"文档正文 {i}", f"doc{i}") for i in range(5)]
