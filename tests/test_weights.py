"""Native safetensors loading and strict checkpoint admission."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import numpy as np
import pytest
from fixture_contracts import ModelFixture
from medmlx_core.errors import InvalidInputError
from medmlx_core.typing import ArrayFactory

from segresnet_mlx import (
    SegResNetSpec,
    forward_patch,
    load_model,
)


@pytest.mark.mlx
def test_load_native_weights_matches_reference(
    no_decoder_reference: ModelFixture, tmp_path: Path
) -> None:
    """Load seeded safetensors and compare resulting logits with the recording."""
    from segresnet_mlx._runtime import import_mlx

    mx = import_mlx()
    spec, arrays, _ = no_decoder_reference
    weights = {
        key.removeprefix("weight::"): cast(ArrayFactory, mx.array)(value)
        for key, value in arrays.items() if key.startswith("weight::")
    }
    path = tmp_path / "model.safetensors"
    mx.save_safetensors(str(path), weights)
    np.testing.assert_allclose(
        forward_patch(load_model(path, spec=spec), arrays["input"]),
        arrays["output"], rtol=0, atol=1e-6,
    )


def test_load_model_rejects_other_formats(tmp_path: Path) -> None:
    """Reject checkpoint paths that are not safetensors."""
    with pytest.raises(InvalidInputError, match="safetensors"):
        load_model(tmp_path / "model.npz", spec=SegResNetSpec(1, 1, 8))


@pytest.mark.parametrize(
    ("corruption", "message"),
    [
        ("keys", "weight keys differ"),
        ("shape", "has shape"),
        ("dtype", "must be finite float32"),
        ("nonfinite", "must be finite float32"),
    ],
)
@pytest.mark.mlx
def test_loaded_weights_reject_corruption(
    no_decoder_reference: ModelFixture,
    tmp_path: Path,
    corruption: str,
    message: str,
) -> None:
    """Reject missing, wrong-shaped, wrong-dtype, and non-finite parameters."""
    from segresnet_mlx._runtime import import_mlx

    mx = import_mlx()

    spec, arrays, _ = no_decoder_reference
    weights = {
        key.removeprefix("weight::"): cast(ArrayFactory, mx.array)(value)
        for key, value in arrays.items()
        if key.startswith("weight::")
    }
    key = "conv_final.2.conv.bias"
    if corruption == "keys":
        del weights[key]
    elif corruption == "shape":
        weights[key] = mx.zeros((1,), dtype=mx.float32)
    elif corruption == "dtype":
        weights[key] = weights[key].astype(mx.float16)
    else:
        weights[key] = mx.full(weights[key].shape, float("inf"), dtype=mx.float32)
    destination = tmp_path / "invalid.safetensors"
    mx.save_safetensors(str(destination), weights)
    with pytest.raises(InvalidInputError, match=message):
        load_model(destination, spec=spec)
