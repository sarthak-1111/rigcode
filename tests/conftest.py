from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from mock_server import MockServer  # noqa: E402

_ENV = ("RIGCODE_BASE_URL", "RIGCODE_MODEL", "RIGCODE_API_KEY", "RIGCODE_EFFORT", "RIGCODE_CONTEXT_WINDOW", "RIGCODE_TOOL_MODE")


@pytest.fixture(autouse=True)
def isolated_home(tmp_path, monkeypatch):
    """Never read or write the real ~/.rigcode during tests."""
    home = tmp_path / "rigcode_home"
    monkeypatch.setenv("RIGCODE_HOME", str(home))
    for k in _ENV:
        monkeypatch.delenv(k, raising=False)
    return home


@pytest.fixture
def server():
    s = MockServer()
    s.base_url = s.start()
    yield s
    s.stop()


@pytest.fixture
def project(tmp_path):
    p = tmp_path / "proj"
    p.mkdir()
    return p
