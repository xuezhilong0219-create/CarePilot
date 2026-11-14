import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import time

import pytest

from carepilot.gateway.spec import Risk


CHILD = Path(__file__).with_name("crash_child.py")
ROOT = CHILD.parents[2]


def child_env() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    env["PYTHONIOENCODING"] = "utf-8"
    env.pop("CAREPILOT_CRASH_AT", None)
    env.pop("CAREPILOT_CRASH_MARKER", None)
    return env


def wait_marker(path: Path, process: subprocess.Popen) -> None:
    deadline = time.monotonic() + 20
    while not path.exists():
        if process.poll() is not None:
            stdout, stderr = process.communicate()
            pytest.fail(f"child exited before marker: {stdout} {stderr}")
        if time.monotonic() > deadline:
            pytest.fail("child never reached crash point")
        time.sleep(0.01)


def kill_at(tmp_path: Path, point: str, *, no_verify: bool = False, risk: Risk = Risk.WRITE_INTERNAL):
    db = tmp_path / "gateway.sqlite"
    effects = tmp_path / "effects.jsonl"
    marker = tmp_path / "crash.marker"
    env = child_env()
    env.update(CAREPILOT_CRASH_AT=point, CAREPILOT_CRASH_MARKER=str(marker))
    command = [sys.executable, str(CHILD), "call", str(db), str(effects), "--risk", str(int(risk))]
    if no_verify:
        command.append("--no-verify")
    process = subprocess.Popen(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8")
    try:
        wait_marker(marker, process)
        process.kill()
        process.communicate(timeout=10)
        assert process.returncode != 0
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate(timeout=10)
    return db, effects


@pytest.mark.parametrize("point", ["before_effect", "after_effect"])
def test_real_process_kill_then_new_process_reconciles_exactly_once(tmp_path, point):
    db, effects = kill_at(tmp_path, point)
    with sqlite3.connect(db) as connection:
        assert connection.execute("SELECT status FROM action_ledger").fetchone()[0] == "started"
        assert connection.execute("SELECT code FROM audit_log").fetchone()[0] == "execution_intent"
    recovery = subprocess.run([sys.executable, str(CHILD), "recover", str(db), str(effects)], env=child_env(), capture_output=True, text=True, encoding="utf-8", timeout=20, check=True)
    payload = json.loads(recovery.stdout)
    assert payload["outcome"]["result"] == {"sent": "one"}
    assert payload["outcome"]["replayed"] == (point == "after_effect")
    expected = "reconciled_adopted" if point == "after_effect" else "reconciled_retry_allowed"
    assert payload["reconciled"][0]["code"] == expected
    assert len(effects.read_text(encoding="utf-8").splitlines()) == 1


def test_no_verify_recovery_never_retries_unknown_effect(tmp_path):
    db, effects = kill_at(tmp_path, "after_effect", no_verify=True)
    recovery = subprocess.run([sys.executable, str(CHILD), "recover", str(db), str(effects), "--no-verify"], env=child_env(), capture_output=True, text=True, encoding="utf-8", timeout=20, check=True)
    payload = json.loads(recovery.stdout)
    assert payload["reconciled"][0]["status"] == "needs_manual"
    assert payload["outcome"]["code"] == "needs_manual"
    assert len(effects.read_text(encoding="utf-8").splitlines()) == 1
    with sqlite3.connect(db) as connection:
        assert connection.execute("SELECT status FROM action_ledger").fetchone()[0] == "started"


@pytest.mark.parametrize("risk", [Risk.STATE_CHANGE, Risk.EXTERNAL_SEND])
@pytest.mark.parametrize("point", ["before_effect", "after_effect"])
def test_approved_actions_recover_in_new_process(tmp_path, risk, point):
    db, effects = kill_at(tmp_path, point, risk=risk)
    recovery = subprocess.run(
        [sys.executable, str(CHILD), "recover", str(db), str(effects), "--risk", str(int(risk))],
        env=child_env(), capture_output=True, text=True, encoding="utf-8", timeout=20, check=True,
    )
    payload = json.loads(recovery.stdout)
    assert payload["outcome"]["result"] == {"sent": "one"}
    assert len(effects.read_text(encoding="utf-8").splitlines()) == 1


def test_multiple_processes_same_file_reserve_and_execute_once(tmp_path):
    db = tmp_path / "gateway.sqlite"
    effects = tmp_path / "effects.jsonl"
    release = tmp_path / "release"
    children = []
    try:
        for index in range(4):
            ready = tmp_path / f"ready-{index}"
            command = [sys.executable, str(CHILD), "concurrent", str(db), str(effects), "--ready", str(ready), "--release", str(release)]
            process = subprocess.Popen(command, env=child_env(), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8")
            children.append((process, ready))
        for process, ready in children:
            wait_marker(ready, process)
        release.write_text("go", encoding="utf-8")
        outcomes = []
        for process, _ in children:
            stdout, stderr = process.communicate(timeout=20)
            assert process.returncode == 0, stderr
            outcomes.append(json.loads(stdout)["outcome"])
        assert sorted(outcome["replayed"] for outcome in outcomes) == [False, True, True, True]
        assert len(effects.read_text(encoding="utf-8").splitlines()) == 1
    finally:
        for process, _ in children:
            if process.poll() is None:
                process.kill()
                process.communicate(timeout=10)
