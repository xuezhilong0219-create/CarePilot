import hashlib
import json
import os
import tempfile
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from .base import Chunk, Hit
from .embedding import Embedder


def _fingerprint(chunks: Sequence[Chunk]) -> str:
    digest = hashlib.sha256()
    for chunk in chunks:
        digest.update(json.dumps([chunk.id, chunk.text], ensure_ascii=False).encode("utf-8"))
    return digest.hexdigest()


class VectorIndex:
    def __init__(
        self, embeddings: np.ndarray, chunk_ids: Sequence[str], embedder_name: str,
        corpus_fingerprint: str | None = None,
    ) -> None:
        matrix = np.asarray(embeddings, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[0] != len(chunk_ids) or matrix.shape[1] == 0:
            raise ValueError("Vector shape does not match chunk count or dimension")
        if len(set(chunk_ids)) != len(chunk_ids):
            raise ValueError("Vector index requires unique chunk IDs")
        if not np.isfinite(matrix).all() or not np.allclose(np.linalg.norm(matrix, axis=1), 1, atol=1e-4):
            raise ValueError("Embeddings must be finite, nonzero, normalized vectors")
        self.embeddings = np.ascontiguousarray(matrix)
        self.chunk_ids = list(chunk_ids)
        self.embedder_name = embedder_name
        self.corpus_fingerprint = corpus_fingerprint

    @classmethod
    def build(cls, chunks: Sequence[Chunk], embedder: Embedder) -> "VectorIndex":
        ids = [c.id for c in chunks]
        if len(set(ids)) != len(ids):
            raise ValueError("Vector index requires unique chunk IDs")
        return cls(embedder.encode([c.text for c in chunks]), ids, embedder.name, _fingerprint(chunks))

    def validate(self, chunks: Sequence[Chunk], embedder_name: str) -> None:
        if self.embedder_name != embedder_name:
            raise ValueError(f"Embedding model mismatch ({self.embedder_name} != {embedder_name}); rebuild the index")
        if self.chunk_ids != [c.id for c in chunks]:
            raise ValueError("Knowledge base chunk count, IDs, or order changed; rebuild the index")
        if self.corpus_fingerprint is not None and self.corpus_fingerprint != _fingerprint(chunks):
            raise ValueError("Knowledge base content changed; rebuild the index")

    def save(self, index_dir: Path) -> None:
        index_dir = Path(index_dir)
        index_dir.mkdir(parents=True, exist_ok=True)
        metadata = dict(
            version=1, embedder_name=self.embedder_name, chunk_count=len(self.chunk_ids),
            dimension=self.embeddings.shape[1], chunk_ids=self.chunk_ids,
            corpus_fingerprint=self.corpus_fingerprint,
            embeddings_sha256=hashlib.sha256(self.embeddings.tobytes()).hexdigest(),
        )
        temporary: list[Path] = []
        try:
            with tempfile.NamedTemporaryFile(dir=index_dir, suffix=".npy", delete=False) as target:
                temporary.append(Path(target.name))
                np.save(target, self.embeddings, allow_pickle=False)
            with tempfile.NamedTemporaryFile(dir=index_dir, mode="w", encoding="utf-8", suffix=".json", delete=False) as target:
                temporary.append(Path(target.name))
                json.dump(metadata, target, ensure_ascii=False)
            # 两文件替换间中断时，加载端的摘要校验会拒绝混合版本。
            os.replace(temporary[0], index_dir / "embeddings.npy")
            os.replace(temporary[1], index_dir / "chunk_ids.json")
        finally:
            for path in temporary:
                path.unlink(missing_ok=True)

    @classmethod
    def load(cls, index_dir: Path, chunks: Sequence[Chunk], embedder_name: str) -> "VectorIndex":
        index_dir = Path(index_dir)
        try:
            metadata = json.loads((index_dir / "chunk_ids.json").read_text(encoding="utf-8"))
            matrix = np.load(index_dir / "embeddings.npy", allow_pickle=False)
            if metadata["version"] != 1 or matrix.dtype != np.float32 or matrix.shape != (metadata["chunk_count"], metadata["dimension"]):
                raise ValueError("Unsupported format or inconsistent vector dimensions")
            if hashlib.sha256(matrix.tobytes()).hexdigest() != metadata["embeddings_sha256"]:
                raise ValueError("Vector checksum mismatch")
            index = cls(matrix, metadata["chunk_ids"], metadata["embedder_name"], metadata["corpus_fingerprint"])
            index.validate(chunks, embedder_name)
            return index
        except (OSError, ValueError, KeyError, TypeError, EOFError) as error:
            raise ValueError(f"Invalid vector index at {index_dir}: {error}. Rebuild the index.") from error

    def search(self, query_vector: np.ndarray, k: int = 4) -> list[tuple[int, float]]:
        if k <= 0 or not self.chunk_ids:
            return []
        query = np.asarray(query_vector, dtype=np.float32)
        if query.shape != (self.embeddings.shape[1],):
            raise ValueError("Query vector dimension does not match the index")
        norm = np.linalg.norm(query)
        if not np.isfinite(query).all() or norm == 0:
            raise ValueError("Query vector must be finite and nonzero")
        scores = self.embeddings @ (query / norm)
        count = min(k, len(scores))
        candidates = np.argpartition(-scores, count - 1)[:count]
        # 包含边界同分项，用原始顺序打破平分，保证落盘前后结果稳定。
        candidates = np.flatnonzero(scores >= scores[candidates].min())
        order = np.lexsort((candidates, -scores[candidates]))[:count]
        return [(int(i), float(scores[i])) for i in candidates[order]]


class VectorRetriever:
    def __init__(self, chunks: Sequence[Chunk], embedder: Embedder, index: VectorIndex | None = None) -> None:
        self.chunks = list(chunks)
        self.embedder = embedder
        self.index = index if index is not None else VectorIndex.build(chunks, embedder)
        self.index.validate(chunks, embedder.name)

    def search(self, query: str, k: int = 4) -> list[Hit]:
        if k <= 0 or not query.strip() or not self.chunks:
            return []
        vector = self.embedder.encode([query])[0]
        return [Hit(self.chunks[i], score) for i, score in self.index.search(vector, k)]
