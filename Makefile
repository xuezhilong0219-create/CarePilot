# 本机没有 make；真正的入口是跨平台的 scripts/check.py 和 eval/eval_all.py，这里只做转发。
.PHONY: check lint quick full

check:
	uv run python scripts/check.py

lint:
	uv run python scripts/check.py --skip tests

quick:
	uv run python eval/eval_all.py --quick

full:
	uv run python eval/eval_all.py --full
