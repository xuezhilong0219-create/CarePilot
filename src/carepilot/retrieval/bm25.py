import math
import re
from collections import Counter

import jieba

from .base import Chunk, Hit

_TOKEN = re.compile(r"[\w一-鿿]+")


def tokenize(text: str) -> list[str]:
    # jieba 切中文词，同时保留型号 / 错误码这类英文数字串（小写化）
    return [t.lower() for t in jieba.lcut_for_search(text) if _TOKEN.fullmatch(t)]


class BM25Retriever:
    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75):
        self.chunks = chunks
        self.k1, self.b = k1, b
        self.docs = [Counter(tokenize(c.text)) for c in chunks]
        self.lens = [sum(d.values()) for d in self.docs]
        self.avg_len = (sum(self.lens) / len(self.lens)) if self.lens else 0.0
        df: Counter[str] = Counter()
        for d in self.docs:
            df.update(d.keys())
        n = len(chunks)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def search(self, query: str, k: int = 4) -> list[Hit]:
        q_tokens = tokenize(query)
        scored: list[Hit] = []
        for chunk, doc, length in zip(self.chunks, self.docs, self.lens):
            score = 0.0
            for t in q_tokens:
                tf = doc.get(t, 0)
                if not tf:
                    continue
                norm = tf + self.k1 * (1 - self.b + self.b * length / self.avg_len)
                score += self.idf[t] * tf * (self.k1 + 1) / norm
            if score > 0:
                scored.append(Hit(chunk, score))
        scored.sort(key=lambda h: h.score, reverse=True)
        return scored[:k]
