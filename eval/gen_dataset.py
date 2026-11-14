"""反向合成评估集：抽样文档 → 用 LLM 以客户口吻写工单 → 校验后落盘。

用法见 .team/tasks/E2.md。真实运行需要 .env 里的 LLM 配置；单测只用假模型。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 允许 `python eval/gen_dataset.py` 直接从仓库根目录运行，不必先安装包
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from carepilot.evaluation.gen_core import (  # noqa: E402
    SYSTEM_ANSWERABLE,
    CachedGenerator,
    ManifestEntry,
    build_answerable_prompt,
    load_manifest,
    plan_samples,
    read_doc,
    run_answerables,
    run_unanswerables,
    assemble_dataset,
    write_review,
)
from carepilot.evaluation.schema import dump_dataset  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="从清洗后的知识库反向合成评估集")
    parser.add_argument("--kb-dir", type=Path, default=Path("data/kb_real"))
    parser.add_argument("--n-answerable", type=int, default=120)
    parser.add_argument("--n-unanswerable", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=Path("eval/real/dataset.jsonl"))
    parser.add_argument("--model", default="qwen3.8-flash")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true", help="不调用 LLM，只打印抽样和提示词样例")
    return parser


def print_plan(plan, kb_dir: Path, n_answerable: int) -> None:
    print(f"目录名额（共 {sum(plan.quotas.values())} 篇）：")
    for top_dir in sorted(plan.quotas):
        print(f"  {top_dir}: {plan.quotas[top_dir]} 篇")
    print("抽到的文档：")
    for entry in plan.selected:
        print(f"  [{entry.top_dir}] {entry.title}  ({entry.doc_id}, {entry.chars} 字)")

    sample = plan.selected[0] if plan.selected else None
    if sample is not None:
        prompt = build_answerable_prompt(sample, read_doc(kb_dir, sample))
    else:
        prompt = build_answerable_prompt(
            ManifestEntry(doc_id="example", title="示例标题", title_key="示例标题", top_dir="示例"),
            "示例正文：这里放清洗后的 markdown 内容。",
        )
    print("\n=== SYSTEM 提示词 ===")
    print(SYSTEM_ANSWERABLE)
    print("\n=== 一条完整的可答样本提示词 ===")
    print(prompt)

    # 粗估：可答样本平均按 1.25 次调用（留重试余量），负样本固定 2 次；
    # 中文按约 1.5 字/token，输出按每条 300 token 计。
    calls = int(n_answerable * 1.25) + 2
    tokens = int(calls * (len(prompt) / 1.5 + 300))
    print(f"\n预计 LLM 调用：约 {calls} 次（可答 {n_answerable} 条 + 负样本 2 类 + 重试余量）")
    print(f"预计 token：约 {tokens:,}（按输入 {len(prompt)} 字、每条输出 300 token 粗估）")


def make_generator(args: argparse.Namespace, cache_dir: Path) -> CachedGenerator:
    from carepilot.config import get_settings
    from carepilot.llm import StructuredLLM, get_llm

    settings = get_settings().model_copy(update={"llm_fast_model": args.model})

    def invoke(schema, temperature, messages):
        llm = get_llm(settings, "fast").model_copy(update={"temperature": temperature})
        return StructuredLLM(llm, schema).invoke(messages)

    return CachedGenerator(cache_dir, invoke)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    kb_dir = Path(args.kb_dir)
    out = Path(args.out)
    cache_dir = out.parent / "cache" / "gen"
    review_path = out.parent / "dataset_review.md"

    try:
        entries = load_manifest(kb_dir)
        plan = plan_samples(entries, args.n_answerable, args.seed)
    except (FileNotFoundError, ValueError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 2

    if args.dry_run:
        print_plan(plan, kb_dir, args.n_answerable)
        return 0

    try:
        generator = make_generator(args, cache_dir)
    except Exception as exc:  # 缺 .env / 配置不合法时直接说清楚
        print(f"错误：初始化 LLM 失败（检查 .env 里的 LLM_API_KEY / LLM_BASE_URL）：{exc}", file=sys.stderr)
        return 2

    workers = max(1, args.concurrency)
    try:
        attempts, stats = run_answerables(
            plan, kb_dir, generator, args.model, args.n_answerable, workers=workers
        )
        answerables = [(a.entry, a.ticket) for a in attempts if a.ticket is not None]
        unanswerables = run_unanswerables(
            generator, args.model, args.n_unanswerable, workers=workers
        )
    except ValueError as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 2

    if len(unanswerables) < args.n_unanswerable:
        print(
            f"警告：负样本只生成出 {len(unanswerables)} 条，少于要求的 {args.n_unanswerable} 条",
            file=sys.stderr,
        )

    items = assemble_dataset(answerables, unanswerables)
    dump_dataset(items, out)
    write_review(review_path, items, kb_dir, args.seed)

    print(f"生成完成：可答 {len(answerables)} 条，不可答 {len(unanswerables)} 条，共 {len(items)} 条")
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
