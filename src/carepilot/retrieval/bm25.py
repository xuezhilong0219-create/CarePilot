import math
import re
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

import jieba
import numpy as np

from .base import Chunk, Hit

_SEGMENT = re.compile(r"[A-Za-z0-9_]+(?:[.\-:/+][A-Za-z0-9_]+)*|[\u3400-\u9fff]+")


@lru_cache(maxsize=1)
def _jieba() -> jieba.Tokenizer:
    tokenizer = jieba.Tokenizer()
    tokenizer.load_userdict(str(Path(__file__).with_name("userdict.txt")))
    return tokenizer


def tokenize(text: str, tokenizer: str = "jieba") -> list[str]:
    if tokenizer not in ("jieba", "bigram"):
        raise ValueError(f"Unknown tokenizer: {tokenizer}")
    tokens: list[str] = []
    for match in _SEGMENT.finditer(text):
        segment = match[0]
        if segment.isascii():
            tokens.append(segment.lower())
        elif tokenizer == "jieba":
            tokens.extend(_jieba().lcut_for_search(segment))
        elif len(segment) == 1:
            tokens.append(segment)
        else:
            tokens.extend(segment[i:i + 2] for i in range(len(segment) - 1))
    return tokens


class BM25Retriever:
    def __init__(
        self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75, *,
        tokenizer: str = "jieba", title_weight: float = 2.0,
    ) -> None:
        if not math.isfinite(k1) or k1 <= 0 or not 0 <= b <= 1 or not math.isfinite(title_weight) or title_weight < 1:
            raise ValueError("Require k1 > 0, 0 <= b <= 1, and title_weight >= 1")
        tokenize("", tokenizer)
        self.chunks = list(chunks)
        self.tokenizer = tokenizer
        self.k1, self.b = k1, b
        postings: dict[str, list[tuple[int, float]]] = defaultdict(list)
        lengths = []
        for i, chunk in enumerate(chunks):
            counts = Counter(tokenize(chunk.text, tokenizer))
            if title_weight > 1:
                first_line = chunk.text.split("\n", 1)[0]
                heading = first_line if chunk.title in first_line else chunk.title
                for term, count in Counter(tokenize(heading, tokenizer)).items():
                    counts[term] += count * (title_weight - 1)
            lengths.append(sum(counts.values()))
            for term, frequency in counts.items():
                postings[term].append((i, frequency))
        self.lens = np.asarray(lengths, dtype=np.float64)
        self.avg_len = float(self.lens.mean()) if lengths else 0.0
        n = len(chunks)
        self.idf = {term: math.log(1 + (n - len(rows) + .5) / (len(rows) + .5)) for term, rows in postings.items()}
        self.postings: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for term, rows in postings.items():
            ids = np.asarray([row[0] for row in rows], dtype=np.intp)
            tf = np.asarray([row[1] for row in rows], dtype=np.float64)
            norm = tf + k1 * (1 - b + b * self.lens[ids] / self.avg_len)
            self.postings[term] = ids, self.idf[term] * tf * (k1 + 1) / norm

    def search(self, query: str, k: int = 4) -> list[Hit]:
        if k <= 0 or not self.chunks:
            return []
        scores = np.zeros(len(self.chunks), dtype=np.float64)
        for term in tokenize(query, self.tokenizer):
            if term in self.postings:
                ids, contributions = self.postings[term]
                scores[ids] += contributions
        candidates = np.flatnonzero(scores > 0)
        if len(candidates) > k:
            threshold = np.partition(scores[candidates], -k)[-k]
            candidates = candidates[scores[candidates] >= threshold]
        order = np.lexsort((candidates, -scores[candidates]))[:k]
        return [Hit(self.chunks[i], float(scores[i])) for i in candidates[order]]
