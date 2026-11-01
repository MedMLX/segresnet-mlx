"""Small, immutable golden records from the pinned MONAI implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from fixture_contracts import ModelFixture

if TYPE_CHECKING:
    from segresnet_mlx._types import MLX

import pytest
from reference_data import network_fixture


@pytest.fixture(params=["segresnet_no_decoder", "segresnet_nontrainable", "segresnet_deconv"])
def reference(request: pytest.FixtureRequest) -> ModelFixture:
    """Provide each recorded network topology to the test."""
    return network_fixture(request.param)


@pytest.fixture
def no_decoder_reference() -> ModelFixture:
    """Provide the recorded network without a decoder."""
    return network_fixture("segresnet_no_decoder")


@pytest.fixture
def mx() -> MLX:
    """Import the MLX runtime used by Metal tests."""
    from segresnet_mlx._runtime import import_mlx

    return import_mlx()
