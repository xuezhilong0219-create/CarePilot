# 本机没有 make；主入口是跨平台的 eval/eval_all.py，这里只做转发。
.PHONY: quick full

quick:
	uv run python eval/eval_all.py --quick

full:
	uv run python eval/eval_all.py --full
