from __future__ import annotations

from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def fixtures_dir() -> Path:
    return FIXTURES_DIR


@pytest.fixture
def peakhurst_image() -> Path:
    return FIXTURES_DIR / "peakhurst.png"


@pytest.fixture
def woolloongabba_image() -> Path:
    return FIXTURES_DIR / "woolloongabba.png"


@pytest.fixture
def keysborough_image() -> Path:
    return FIXTURES_DIR / "keysborough.png"


@pytest.fixture
def katanning_image() -> Path:
    return FIXTURES_DIR / "katanning.png"


@pytest.fixture
def campbelltownlink9_image() -> Path:
    return FIXTURES_DIR / "campbelltownlink9.png"
