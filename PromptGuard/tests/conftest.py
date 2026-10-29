"""Test-only path fixture for the DSH Windows sandbox.

pytest's built-in ``tmp_path`` creates directories with mode 0700.  In this
DSH workspace that mode makes the directory unreadable/unwritable even for the
process that created it (WinError 5).  Keep the standard fixture name used by
the tests, but create a normal-mode directory under ``artifacts/test-tmp``
instead and remove it after the test.
"""
import shutil
import uuid
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def tmp_path():
    base = _PROJECT_ROOT / "artifacts" / "test-tmp"
    path = base / uuid.uuid4().hex
    path.mkdir(parents=True)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)
