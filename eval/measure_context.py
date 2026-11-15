"""对话历史的上下文预算：改动前后各塞多少 token，跑这个脚本看数。

为什么单独一个脚本：`eval_all.py` 的评估集是单轮的，多轮历史不进提示词，
所以端到端指标对这次改动是盲的。这里不再造一套 LLM 评估（那要新的标注），
只把「同样一段会话，旧实现和新实现分别往提示词里塞多少」算清楚——
这是确定性计算，不需要模型，也不需要密钥。

    uv run python eval/measure_context.py
"""

from __future__ import annotations

import argparse

from carepilot.graph.context import (
    DEFAULT_BUDGET_TOKENS,
    TURN_CHARS,
    VERBATIM_RECENT,
    compress_history,
    estimate_tokens,
)

#: 旧实现的窗口：最近 6 轮、每条 300 字
OLD_LIMIT = 6
OLD_CHARS = 300


def old_render(history: list[dict]) -> str:
    rows = history[-OLD_LIMIT:]
    return "\n".join(f"{'客户' if h['role'] == 'customer' else '客服'}：{h['text'][:OLD_CHARS]}" for h in rows)


def turn(role: str, n: int) -> dict:
    """一轮 n 字的典型内容：客户描述现象 + 客套，或者是坐席的标准答复。"""
    body = "订单导不出来，一直提示网络异常，我换了三个浏览器都一样，麻烦看下。" if role == "customer" \
        else "我这边查一下，您先按这个步骤操作：设置-导出-重新授权，好了再试一次。"
    return {"role": role, "text": (body * ((n // len(body)) + 1))[:n]}


def conversation(turns: int, chars: int) -> list[dict]:
    return [turn("customer" if i % 2 == 0 else "agent", chars) for i in range(turns)]


CASES = [
    ("3 轮短会话（正常工单）", 4, 250),
    ("10 轮普通会话", 10, 300),
    ("30 轮长会话", 30, 300),
    ("30 轮贴报错（每轮 900 字）", 30, 900),
    ("客户连贴 12 条日志", 12, 1500),
    ("100 轮马拉松会话", 100, 400),
]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="对比上下文预算改动前后的提示词体积")
    ap.add_argument("--budget", type=int, default=DEFAULT_BUDGET_TOKENS, help=f"token 预算（默认 {DEFAULT_BUDGET_TOKENS}）")
    args = ap.parse_args(argv)

    print(f"预算 {args.budget} token；最近 {VERBATIM_RECENT} 轮逐字（每轮上限 {TURN_CHARS} 字），更早的压成一行摘要\n")
    head = (f"{'会话':<24}{'轮':>4}{'旧 token':>9}{'旧覆盖':>7}{'新 token':>9}{'新覆盖':>7}{'丢弃':>5}"
            f"{'旧丢字%':>8}{'新丢字%':>8}")
    print(head)
    print("-" * len(head))

    for name, turns, chars in CASES:
        history = conversation(turns, chars)
        old_text = old_render(history)
        view = compress_history(history, budget_tokens=args.budget)
        r = view.report
        # 「丢字」= 没进提示词的原文字符占比，衡量的是信息损失，不是省钱；
        # 只算正文（去掉「客户：」前缀和省略号），否则行前缀会把比例算成负数
        old_kept = sum(min(len(h["text"]), OLD_CHARS) for h in history[-OLD_LIMIT:])
        payloads = [line.split("：", 1)[1].rstrip("…") for line in view.text.splitlines()]
        new_kept = sum(len(x) for x in payloads)
        total = sum(len(h["text"]) for h in history)
        print(f"{name:<24}{len(history):>4}{estimate_tokens(old_text):>9}{min(len(history), OLD_LIMIT):>7}"
              f"{r['tokens_after']:>9}{r['verbatim'] + r['digested']:>7}{r['dropped']:>5}"
              f"{(1 - old_kept / total) * 100:>7.0f}%{(1 - new_kept / total) * 100:>7.0f}%")

    print("\n旧实现：固定最近 6 轮、每条 300 字，超出的轮次和字数静默丢弃——20 轮以后的工单它压根没看过。")
    print("新实现：每轮都进提示词（最近 3 轮逐字 600 字，更早的一行摘要），token 有上界、不随会话长度线性增长。")
    print("注：token 是估算值（中文按字、ASCII 按 4 字符），只用来比相对大小，不能当账单。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
