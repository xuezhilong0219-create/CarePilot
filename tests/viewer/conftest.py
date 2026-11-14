import shutil
import uuid
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def work_tmp():
    # 沙箱里 pytest 的 tmp_path 建在不可写的系统临时目录下，用仓库根的 .tmp_test/
    path = ROOT / ".tmp_test" / f"viewer-{uuid.uuid4().hex}"
    path.mkdir(parents=True)
    yield path
    shutil.rmtree(path, ignore_errors=True)
