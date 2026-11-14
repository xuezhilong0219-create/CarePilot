import shutil
import uuid
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]


@pytest.fixture
def work_dir():
    """沙箱可写仓库根目录；用 .tmp_test/ 避免依赖系统临时目录。"""
    path = _REPO / ".tmp_test" / f"sandbox-{uuid.uuid4().hex}"
    path.mkdir(parents=True)
    yield path
    shutil.rmtree(path, ignore_errors=True)
