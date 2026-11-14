"""按任务 G 的五处故障逐个修改源文件、运行断言、恢复原始字节。"""
from pathlib import Path
import os
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
GATEWAY = ROOT / "src/carepilot/gateway/gateway.py"


def main() -> None:
    mutations = [
        ("1_remove_insert_or_ignore", GATEWAY,
         "INSERT OR IGNORE INTO action_ledger", "INSERT INTO action_ledger",
         "tests/gateway/test_ledger.py::test_two_thread_barrier_fifty_rounds_execute_once"),
        ("2_skip_pending_check", GATEWAY,
         'if row["status"] != "pending":', 'if False:',
         "tests/gateway/test_approvals.py::test_approve_requires_pending_including_already_approved"),
        ("3_confirm_executes_again", GATEWAY,
         'spec, args = inputs\n            with self.ledger.transaction() as db:\n                self._event(db, ctx, row["tool"], self._summary(spec, args), "confirmed"',
         'spec, args = inputs\n            if row["status"] == "confirmed":\n                spec.fn(ctx, args)\n            with self.ledger.transaction() as db:\n                self._event(db, ctx, row["tool"], self._summary(spec, args), "confirmed"',
         "tests/gateway/test_approvals.py::test_confirm_replays_saved_result_and_parameters"),
        ("4_plaintext_audit_parameters", GATEWAY,
         'value = spec.summarize_args(args) if spec and spec.summarize_args else args.model_dump(mode="json")\n            return safe_summary(value)',
         'return canonical_json(args.model_dump(mode="json"))',
         "tests/gateway/test_audit.py::test_sensitive_args_and_results_never_enter_any_audit_column"),
        ("5_retry_without_verify", GATEWAY,
         'status, code = "needs_manual", "reconciled_needs_manual"',
         'status, code = "abandoned", "reconciled_retry_allowed"',
         "tests/gateway/test_crash_recovery.py::test_no_verify_recovery_never_retries_unknown_effect"),
    ]
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    for label, path, old, new, test in mutations:
        original = path.read_bytes()
        source = original.decode("utf-8")
        if old not in source:
            raise AssertionError(f"mutation target missing: {label}")
        try:
            path.write_text(source.replace(old, new, 1), encoding="utf-8")
            result = subprocess.run([sys.executable, "-m", "pytest", "-q", test],
                                    cwd=ROOT, env=env, capture_output=True, text=True, encoding="utf-8", timeout=60)
            print(f"MUTATION {label}: exit={result.returncode}", flush=True)
            print(result.stdout, flush=True)
            if result.returncode != 1 or "FAILED " not in result.stdout:
                raise AssertionError(f"expected assertion failure for {label}: {result.stderr}")
        finally:
            path.write_bytes(original)
            for cache in path.parent.glob(f"__pycache__/{path.stem}.*.pyc"):
                cache.unlink()
        assert path.read_bytes() == original
    print("5/5 mutations detected; all original source bytes restored", flush=True)


if __name__ == "__main__":
    main()
