"""Strict checkpoint admission against the topology's tensor inventory."""

from __future__ import annotations

from pathlib import Path

import pytest
from fixture_contracts import ModelFixture
from medmlx_core.errors import InvalidInputError
from reference_data import mlx_weights

from segresnet_mlx import load_network

pytestmark = pytest.mark.mlx


@pytest.mark.parametrize("corruption", ["format", "keys", "shape", "dtype", "nan", "inf"])
def test_load_network_rejects_invalid_checkpoints(
    no_decoder_reference: ModelFixture, tmp_path: Path, corruption: str
) -> None:
    """Reject non-safetensors files and missing, wrong-shaped, or non-float32 parameters."""
    from segresnet_mlx._runtime import import_mlx

    mx = import_mlx()
    spec, arrays, _ = no_decoder_reference
    weights = mlx_weights(arrays, mx)
    key = "conv_final.2.conv.bias"
    if corruption == "keys":
        del weights[key]
    elif corruption == "shape":
        weights[key] = mx.zeros((1,), dtype=mx.float32)
    elif corruption == "dtype":
        weights[key] = weights[key].astype(mx.float16)
    elif corruption in ("nan", "inf"):
        weights[key] = weights[key] * 0 + float(corruption)
    destination = tmp_path / ("model.npz" if corruption == "format" else "invalid.safetensors")
    if corruption != "format":
        mx.save_safetensors(str(destination), weights)
    with pytest.raises(InvalidInputError):
        load_network(destination, spec=spec)
