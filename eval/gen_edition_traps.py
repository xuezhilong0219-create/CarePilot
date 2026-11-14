"""生成产品线陷阱评测集（A2）：企业版 / 极速版同名文档。

用法：
    python eval/gen_edition_traps.py --kb-dir data/kb_real --n-pairs 30 --seed 42 \
        --model deepseek-v4.1-flash-1017 --concurrency 8 \
        --out eval/real/dataset_edition_traps.jsonl [--dry-run]

真实运行需要 .env 里的 LLM 配置；单测只用假模型。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from carepilot.evaluation.edition_traps import (  # noqa: E402
    EditionTrap,
    build_trap_prompt,
    discriminator_pairs,
    dump_traps,
    find_edition_pairs,
    run_edition_traps,
    sample_pairs,
    write_review,
)
from carepilot.evaluation.gen_core import (  # noqa: E402
    CachedGenerator,
    load_manifest,
    read_doc,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="生成企业版/极速版同名文档的陷阱评测集")
    parser.add_argument("--kb-dir", type=Path, default=Path("data/kb_real"))
    parser.add_argument("--n-pairs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", default="deepseek-v4.1-flash-1017")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument(
        "--out", type=Path, default=Path("eval/real/dataset_edition_traps.jsonl")
    )
    parser.add_argument("--dry-run", action="store_true", help="不调用 LLM，只打印配对统计和提示词样例")
    return parser


def make_generator(model: str, cache_dir: Path) -> CachedGenerator:
    from carepilot.config import get_settings
    from carepilot.llm import StructuredLLM, get_llm

    settings = get_settings().model_copy(update={"llm_fast_model": model})

    def invoke(schema, temperature, messages):
        llm = get_llm(settings, "fast").model_copy(update={"temperature": temperature})
        return StructuredLLM(llm, schema).invoke(messages)

    return CachedGenerator(cache_dir, invoke)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    kb_dir = Path(args.kb_dir)
    out = Path(args.out)
    cache_dir = out.parent / "cache" / "gen"
    review_path = out.parent / "dataset_edition_traps_review.md"

    try:
        entries = load_manifest(kb_dir)
        pairs = find_edition_pairs(entries)
        candidates = discriminator_pairs(pairs, kb_dir)
    except (FileNotFoundError, ValueError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 2

    selected = sample_pairs(candidates, args.n_pairs, args.seed)
    print(f"同名配对数（企业版/极速版）：{len(pairs)}")
    print(f"可区分配对数：{len(candidates)}")
    print(f"本次抽中配对数：{len(selected)}")
    for enterprise, _, _, _ in selected:
        print(f"  {enterprise.title_key}  ({enterprise.title})")

    if args.dry_run:
        if selected:
            enterprise, _, enterprise_lines, _ = selected[0]
            prompt = build_trap_prompt(
                enterprise,
                "企业版",
                enterprise_lines,
                read_doc(kb_dir, enterprise),
            )
        else:
            prompt = "（没有可区分的配对，无法给出提示词样例）"
        print("\n=== 一条完整的提示词样例 ===")
        print(prompt)
        return 0

    try:
        generator = make_generator(args.model, cache_dir)
    except Exception as exc:  # 缺 .env / 配置不合法时直接说清楚
        print(f"错误：初始化 LLM 失败（检查 .env 里的 LLM_API_KEY / LLM_BASE_URL）：{exc}", file=sys.stderr)
        return 2

    work_pairs = [(enterprise, express) for enterprise, express, _, _ in selected]
    traps, stats = run_edition_traps(
        work_pairs, kb_dir, generator, args.model, workers=max(1, args.concurrency)
    )
    dump_traps(out, traps)
    write_review(review_path, traps, args.seed)

    print(f"生成完成：{len(traps)} 条（{len(traps) // 2} 组配对）")
    print(f"实际 LLM 调用 {generator.calls} 次（缓存命中不计），缓存目录 {cache_dir}")
    if stats:
        print("拒收原因统计：")
        for reason, count in stats.most_common():
            print(f"  {reason}: {count}")
    print(f"数据集：{out}")
    print(f"人工抽查：{review_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
