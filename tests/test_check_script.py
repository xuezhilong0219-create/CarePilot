import importlib.util
from pathlib import Path


spec = importlib.util.spec_from_file_location("check_script", Path(__file__).parents[1] / "scripts" / "check.py")
check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check)


def test_skip_summary_lists_only_checks_that_ran(monkeypatch, capsys):
    calls = []

    def successful_call(command, *, cwd):
        calls.append((command, cwd))
        return 0

    monkeypatch.setattr(check.subprocess, "call", successful_call)

    assert check.main(["--skip", "tests"]) == 0

    output = capsys.readouterr().out
    assert "[tests] 跳过" in output
    assert "门禁通过：lint\n" in output
    assert "lint + tests" not in output
    assert len(calls) == 1


def test_skipping_every_check_is_not_success(monkeypatch, capsys):
    monkeypatch.setattr(check.subprocess, "call", lambda *args, **kwargs: 0)

    assert check.main(["--skip", "lint,tests"]) == 2
    assert "没有执行任何检查" in capsys.readouterr().out
