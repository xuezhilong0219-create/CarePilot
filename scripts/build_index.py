import argparse
from collections.abc import Sequence
from pathlib import Path
from time import perf_counter

from carepilot.retrieval.chunking import load_kb
from carepilot.retrieval.embedding import FastEmbedEmbedder
from carepilot.retrieval.vector import VectorIndex


def _values(items: list[str]) -> list[str]:
    return [value.strip() for item in items for value in item.split(",") if value.strip()]


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build a local CarePilot vector index")
    parser.add_argument("--kb-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--exclude-dirs", nargs="*", default=[])
    parser.add_argument("--exclude-flags", nargs="*", default=[])
    args = parser.parse_args(argv)
    started = perf_counter()
    chunks = load_kb(args.kb_dir, _values(args.exclude_dirs), _values(args.exclude_flags))
    loading = perf_counter()
    embedder = FastEmbedEmbedder()
    loaded = perf_counter()
    index = VectorIndex.build(chunks, embedder)
    encoded = perf_counter()
    index.save(args.out)
    print(f"chunks={len(chunks)} model={embedder.name} model_load_s={loaded - loading:.3f}")
    print(f"encode_s={encoded - loaded:.3f} chunks_per_second={len(chunks) / max(encoded - loaded, 1e-9):.2f}")
    print(f"total_s={perf_counter() - started:.3f} index_dir={args.out}")


if __name__ == "__main__":
    main()
