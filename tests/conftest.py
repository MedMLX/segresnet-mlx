"""Small, immutable golden records from the pinned MONAI implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from fixture_contracts import ModelFixture

if TYPE_CHECKING:
    from segresnet_mlx._types import MLX

import pytest
from reference_data import FIXTURES, model_fixture


def _load_reference(name: str) -> ModelFixture:
    """Load one named recorded network fixture."""
    return model_fixture(FIXTURES / f"{name}.npz")


@pytest.fixture(params=["segresnet_no_decoder", "segresnet_nontrainable", "segresnet_deconv"])
def reference(request: pytest.FixtureRequest) -> ModelFixture:
    """Provide each recorded network topology to the test."""
    return _load_reference(request.param)


@pytest.fixture
def no_decoder_reference() -> ModelFixture:
    """Provide the recorded network without a decoder."""
    return _load_reference("segresnet_no_decoder")


@pytest.fixture
def encoder_reference() -> ModelFixture:
    """Provide the recorded nontrainable-upsampling network."""
    return _load_reference("segresnet_nontrainable")


@pytest.fixture
def deconv_reference() -> ModelFixture:
    """Provide the recorded transpose-convolution network."""
    return _load_reference("segresnet_deconv")


@pytest.fixture
def mx() -> MLX:
    """Import the MLX runtime used by Metal tests."""
    from segresnet_mlx._runtime import import_mlx

    return import_mlx()
