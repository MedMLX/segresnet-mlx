"""Parity at the original MONAI network and operator boundaries."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from fixture_contracts import ModelFixture

from segresnet_mlx._arrays import NumericArray, as_numpy

if TYPE_CHECKING:
    from segresnet_mlx._types import MLX

import numpy as np
import pytest
from medmlx_core.ops import conv_transpose3d_ncdhw
from medmlx_core.typing import ArrayFactory
from reference_data import mlx_weights, network_fixture

from segresnet_mlx import SegResNet, forward_patch

pytestmark = pytest.mark.mlx


def _assert_float32_reference(actual: NumericArray, expected: NumericArray) -> None:
    # Torch CPU and MLX use different float32 convolution/reduction orders.
    # Seeded logit/operator parity allows float32 accumulation-order differences.
    """Compare seeded float32 outputs within the recorded absolute tolerance."""
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-6)


def test_network_matches_upstream_reference(reference: ModelFixture, mx: MLX) -> None:
    """Compare seeded network logits with MONAI recordings."""
    spec, arrays, _ = reference
    network = SegResNet(spec, mlx_weights(arrays, mx), mx=mx)
    output = network(cast(ArrayFactory, mx.array)(arrays["input"]))
    mx.eval(output)
    assert output.dtype == mx.float32
    _assert_float32_reference(as_numpy(output), arrays["output"])


def test_single_patch_matches_upstream(no_decoder_reference: ModelFixture, mx: MLX) -> None:
    """Compare ``forward_patch`` logits with MONAI."""
    spec, arrays, _ = no_decoder_reference
    network = SegResNet(spec, mlx_weights(arrays, mx), mx=mx)
    output = forward_patch(network, arrays["input"])
    assert output.dtype == np.float32
    _assert_float32_reference(output, arrays["output"])


def test_encoder_downsampling_and_residuals_match_upstream(mx: MLX) -> None:
    """Compare recorded encoder stages within their float32 tolerance."""
    spec, arrays, _ = network_fixture("segresnet_nontrainable")
    network = SegResNet(spec, mlx_weights(arrays, mx), mx=mx)
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


def test_shared_deconvolution_matches_upstream_reference(mx: MLX) -> None:
    """Compare transpose convolution with the recorded MONAI output."""
    _, arrays, _ = network_fixture("segresnet_deconv")
    weights = mlx_weights(arrays, mx)
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
