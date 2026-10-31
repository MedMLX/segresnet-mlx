"""Parity at the original MONAI network and operator boundaries."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from fixture_contracts import ModelFixture, NumericArrays

from segresnet_mlx._arrays import NumericArray, as_numpy

if TYPE_CHECKING:
    from segresnet_mlx._types import MLX, Array

import numpy as np
import pytest
from medmlx_core.errors import InvalidInputError
from medmlx_core.ops import conv_transpose3d_ncdhw
from medmlx_core.typing import ArrayFactory

from segresnet_mlx import SegResNet, forward_patch
from segresnet_mlx.network import group_mean

pytestmark = pytest.mark.mlx


def _assert_float32_reference(actual: NumericArray, expected: NumericArray) -> None:
    # Torch CPU and MLX use different float32 convolution/reduction orders.
    # Seeded logit/operator parity allows float32 accumulation-order differences.
    """Compare seeded float32 outputs within the recorded absolute tolerance."""
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-6)


def _weights(arrays: NumericArrays, mx: MLX) -> dict[str, Array]:
    """Transfer recorded network weights to MLX under their original names."""
    return {
        key.removeprefix("weight::"): cast(ArrayFactory, mx.array)(value)
        for key, value in arrays.items()
        if key.startswith("weight::")
    }


def test_network_matches_upstream_reference(reference: ModelFixture, mx: MLX) -> None:
    """Compare seeded network logits with MONAI recordings."""
    spec, arrays, _ = reference
    network = SegResNet(spec, _weights(arrays, mx), mx=mx)
    output = network(cast(ArrayFactory, mx.array)(arrays["input"]))
    mx.eval(output)
    assert output.dtype == mx.float32
    _assert_float32_reference(as_numpy(output), arrays["output"])


def test_single_patch_and_tiled_group_mean_match_upstream(
    no_decoder_reference: ModelFixture,
    mx: MLX,
) -> None:
    """Compare patch logits and tiled group means with MONAI."""
    spec, arrays, _ = no_decoder_reference
    network = SegResNet(spec, _weights(arrays, mx), mx=mx)
    output = forward_patch(network, arrays["input"])
    assert output.dtype == np.float32
    _assert_float32_reference(output, arrays["output"])
    mean = group_mean(cast(ArrayFactory, mx.array)(arrays["group_input"]), mx)
    mx.eval(mean)
    _assert_float32_reference(as_numpy(mean), arrays["group_output"])


def test_encoder_downsampling_and_residuals_match_upstream(
    encoder_reference: ModelFixture,
    mx: MLX,
) -> None:
    """Compare recorded encoder stages within their float32 tolerance."""
    spec, arrays, _ = encoder_reference
    network = SegResNet(spec, _weights(arrays, mx), mx=mx)
    initial = network.conv(
        cast(ArrayFactory, mx.array)(arrays["input"]), "convInit.conv", padding=1
    )
    residual = network.residual(initial, "down_layers.0.1")
    downsampled = network.conv(residual, "down_layers.1.0.conv", padding=1, stride=2)
    encoded = network.residual(downsampled, "down_layers.1.1")
    for name, output in (
        ("encoder_initial", initial),
        ("encoder_residual", residual),
        ("encoder_downsampled", downsampled),
        ("encoder_output", encoded),
    ):
        mx.eval(output)
        # Native Torch/Metal convolutions and reductions round differently. The
        # reported downsampling outlier is 1.66893e-6 (7 ULPs); 2**-19 is the
        # enclosing absolute bound. Public logit assertions retain 1e-6.
        np.testing.assert_allclose(
            as_numpy(output), arrays[name], rtol=0, atol=2**-19, err_msg=name
        )


def test_shared_deconvolution_matches_upstream_reference(
    deconv_reference: ModelFixture, mx: MLX
) -> None:
    """Compare transpose convolution with the recorded MONAI output."""
    _, arrays, _ = deconv_reference
    weights = _weights(arrays, mx)
    output = conv_transpose3d_ncdhw(
        cast(ArrayFactory, mx.array)(arrays["deconv_input"]),
        weights["up_samples.0.1.deconv.weight"],
        weights["up_samples.0.1.deconv.bias"],
        padding=0,
        output_padding=0,
        stride=2,
        mx=mx,
    )
    mx.eval(output)
    _assert_float32_reference(as_numpy(output), arrays["deconv_output"])


@pytest.mark.parametrize("invalid", ["dtype", "nonfinite", "rank", "channels"])
def test_single_patch_rejects_invalid_input(
    no_decoder_reference: ModelFixture,
    invalid: str,
    mx: MLX,
) -> None:
    """Reject invalid patch dtype, finiteness, rank, and channels."""
    spec, arrays, _ = no_decoder_reference
    network = SegResNet(spec, _weights(arrays, mx), mx=mx)
    values: NumericArray = arrays["input"].copy()
    if invalid == "dtype":
        values = values.astype(np.float64)
        error, message = InvalidInputError, "input must be finite float32"
    elif invalid == "nonfinite":
        values.flat[0] = np.nan
        error, message = InvalidInputError, "input must be finite float32"
    elif invalid == "rank":
        values = values[0]
        error, message = ValueError, "must be NCDHW rank-5"
    else:
        values = values[:, :1]
        error, message = ValueError, "has 1 channels; expected 4"
    with pytest.raises(error, match=message):
        forward_patch(network, values)
