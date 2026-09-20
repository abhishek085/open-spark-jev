from __future__ import annotations

import pytest

from os_datagen.config import PipelineConfig
from os_datagen.taskpacks.registry import get_pack, pack_names


@pytest.fixture(scope="session")
def cfg() -> PipelineConfig:
    return PipelineConfig()


@pytest.fixture(params=pack_names())
def pack(request):
    return get_pack(request.param)
