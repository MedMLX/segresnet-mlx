"""Full-volume parity against the original MONAI SlidingWindowInferer."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Literal, cast

from fixture_contracts import NumericArrays, PredictorMetadata

if TYPE_CHECKING:
    from pathlib import Path

    from medmlx_core.typing import MlxRuntime

    from segresnet_mlx._types import Array

import numpy as np
import pytest
from medmlx_core.errors import InvalidInputError
from reference_data import FIXTURES, load_fixture, mlx_weights, segresnet_spec

from segresnet_mlx import SegResNet, SlidingWindowConfig, predict_logits
from segresnet_mlx.predictor import BundlePredictor
from segresnet_mlx.telemetry import WindowTelemetry

pytestmark = pytest.mark.mlx


@pytest.mark.parametrize("device_input", [False, True])
def test_forward_retains_network_logits(device_input: bool, mx: MlxRuntime) -> None:
    """The MLX callback returns the network array itself, with no host round trip."""
    host = np.full((1, 1, 2, 3, 4), 0.5, dtype=np.float32)
    logits = mx.array(host * 2)

    def network(values: Array) -> Array:
        assert values.dtype == mx.float32
        return logits

    result = BundlePredictor(network, mx=mx).forward(mx.array(host) if device_input else host)
    assert result is logits
    np.testing.assert_array_equal(np.array(result, dtype=np.float32), host * 2)


@pytest.mark.parametrize("failure", ["input_dtype", "input_nonfinite", "output_nonfinite"])
@pytest.mark.parametrize("device_input", [False, True])
def test_forward_validation_records_failure(
    failure: str, device_input: bool, mx: MlxRuntime, tmp_path: Path,
) -> None:
    """Both input boundaries reject bad data and output failures close telemetry."""
    host = np.ones((1, 1, 2, 3, 4), dtype=np.float32)
    if failure == "input_dtype":
        host = host.astype(np.float16)
    elif failure == "input_nonfinite":
        host.flat[0] = np.nan
    telemetry = WindowTelemetry(tmp_path / "windows.json", {})

    def network(values: Array) -> Array:
        return values * float("inf")

    predictor = BundlePredictor(network, mx=mx, telemetry=telemetry)
    expected = RuntimeError if failure == "output_nonfinite" else InvalidInputError
    with pytest.raises(expected):
        predictor.forward(mx.array(host) if device_input else host)
    assert telemetry.record["status"] == "failed"
    assert telemetry.record["window_count"] == 0


@pytest.mark.parametrize("accumulation", ["numpy", "mlx"])
def test_volume_telemetry_counts_completed_windows(
    mx: MlxRuntime, tmp_path: Path, accumulation: Literal["numpy", "mlx"],
) -> None:
    """Validated device patches retain the caller-owned telemetry lifecycle."""
    host = np.ones((1, 1, 5, 3, 4), dtype=np.float32)
    telemetry = WindowTelemetry(tmp_path / "windows.json", {})
    predictor = BundlePredictor(lambda values: values * 2, mx=mx, telemetry=telemetry)
    result = predictor.predict_logits(
        host, SlidingWindowConfig((3, 3, 4)), accumulation=accumulation
    )
    np.testing.assert_array_equal(result, host * 2)
    assert telemetry.record["window_count"] == 2
    assert telemetry.record["status"] == "running"
    telemetry.finish()
    assert telemetry.record["status"] == "completed"


@pytest.mark.parametrize(
    "name",
    [
        "renal_standin",
        "renal_actual_roi",
        "renal_strided_reduction",
        "renal_single_window_batch",
        "renal_no_decoder_network",
        "renal_decoder_network",
    ],
)
@pytest.mark.parametrize("accumulation", ["numpy", "mlx"])
def test_predictor_matches_upstream(
    name: str, mx: MlxRuntime, accumulation: Literal["numpy", "mlx"],
) -> None:
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

    result = predict_logits(arrays["input"], configuration, network=measured,
                            accumulation=accumulation)
    assert result.dtype == np.float32
    assert result.shape == arrays["output"].shape
    # Torch CPU and Metal may fuse/reduce float32 operations differently, including
    # elementwise stand-ins. Seeded raw logits must still agree within 1e-6.
    np.testing.assert_allclose(result, arrays["output"], rtol=0, atol=1e-6)
    assert np.array_equal(np.array(batches), arrays["window_batches"])
