"""Full-volume parity against the original MONAI SlidingWindowInferer."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, cast

from fixture_contracts import NumericArrays, PredictorMetadata

if TYPE_CHECKING:
    from segresnet_mlx._types import MLX, Array

import numpy as np
import pytest
from reference_data import FIXTURES, load_fixture, mlx_weights, segresnet_spec

from segresnet_mlx import SegResNet, SlidingWindowConfig, predict_logits

pytestmark = pytest.mark.mlx


@pytest.mark.parametrize(
    "name",
    [
        "brats_standin",
        "brats_actual_roi",
        "renal_standin",
        "renal_actual_roi",
        "brats_strided_reduction",
        "renal_strided_reduction",
        "renal_single_window_batch",
        "brats_no_decoder_network",
        "renal_no_decoder_network",
        "brats_decoder_network",
        "renal_decoder_network",
        "wholebody_standin",
        "wholebody_actual_roi",
        "wholebody_strided_reduction",
        "wholebody_no_decoder_network",
        "wholebody_decoder_network",
    ],
)
def test_predictor_matches_upstream(name: str, mx: MLX) -> None:
    """Compare sliding-window logits and batch shapes with recorded MONAI cases."""
    arrays, metadata = load_fixture(FIXTURES / f"predictor_{name}.npz")
    metadata = cast(PredictorMetadata, metadata)
    arrays = cast(NumericArrays, arrays)
    configuration = SlidingWindowConfig.from_bundle_config(metadata["inferer"])
    network: Callable[[Array], Array]
    if "spec" in metadata:
        network = SegResNet(segresnet_spec(metadata["spec"]), mlx_weights(arrays, mx), mx=mx)
    else:
        coefficients = arrays["coefficients"]

        def standin(x: Array) -> Array:
            """Reproduce the recorded elementwise or per-window reduction predictor."""
            origin = (
                mx.mean(x, axis=(2, 3, 4), keepdims=True)
                if metadata["kind"] == "standin_reduction"
                else x[:, :, :1, :1, :1]
            )
            return mx.concatenate(
                [
                    (
                        x[:, index % x.shape[1] : index % x.shape[1] + 1]
                        + origin[:, index % x.shape[1] : index % x.shape[1] + 1]
                    )
                    * scale
                    + bias
                    for index, (scale, bias) in enumerate(coefficients)
                ],
                axis=1,
            )

        network = standin

    batches: list[list[int]] = []

    def measured(values: Array) -> Array:
        """Record each window batch shape before evaluating its network."""
        batches.append(list(values.shape))
        return network(values)

    result = predict_logits(arrays["input"], configuration, network=measured)
    assert result.dtype == np.float32
    assert result.shape == arrays["output"].shape
    # Torch CPU and Metal may fuse/reduce float32 operations differently, including
    # elementwise stand-ins. Seeded raw logits must still agree within 1e-6.
    np.testing.assert_allclose(result, arrays["output"], rtol=0, atol=1e-6)
    assert np.array_equal(np.array(batches), arrays["window_batches"])
