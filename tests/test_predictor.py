"""Full-volume parity against the original MONAI SlidingWindowInferer."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, cast
from unittest.mock import patch

from fixture_contracts import ConstructorSpec, ModelFixture, NumericArrays, PredictorMetadata

from segresnet_mlx._arrays import Float32Array, HostArray

if TYPE_CHECKING:
    from segresnet_mlx._types import MLX, Array

import numpy as np
import pytest
from medmlx_core.errors import InvalidInputError
from medmlx_core.typing import ArrayFactory
from reference_data import FIXTURES, load_fixture

from segresnet_mlx import (
    BundlePredictor,
    SegResNet,
    SegResNetSpec,
    SlidingWindowConfig,
    WindowTelemetry,
    predict_logits,
)
from segresnet_mlx.predictor import predict_wide_logits


@pytest.mark.parametrize("padded", [False, True])
def test_single_window_inference_avoids_discarded_copies(padded: bool, mx: MLX) -> None:
    """Check single-window padding and concatenation while preserving values."""
    side = 3 if padded else 6
    values = np.arange(side**3, dtype=np.float32).reshape(1, 1, side, side, side)
    predictor = BundlePredictor(lambda value: value, mx=mx)
    with (
        patch("numpy.concatenate", wraps=np.concatenate) as concatenate,
        patch("numpy.pad", wraps=np.pad) as pad,
    ):
        result = predictor.predict_logits(values, SlidingWindowConfig((4, 4, 4), overlap=0.5))
    assert concatenate.call_count == 0
    assert pad.call_count == int(padded)
    np.testing.assert_array_equal(result, values)


@pytest.mark.parametrize("failure", [None, "exception", "nonfinite"])
def test_wide_predictor_keeps_cache_until_completion_or_failure(
    failure: str | None, mx: MLX, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Release the MLX cache after the last window or an inference failure."""
    windows = 0
    releases: list[int] = []
    clear_cache = mx.clear_cache

    def release() -> None:
        """Record when the predictor releases the MLX cache."""
        releases.append(windows)
        clear_cache()

    class PairedNetwork:
        def paired_forward(self, values: Array) -> Array:
            """Return paired test values or trigger the selected second-window failure."""
            nonlocal windows
            windows += 1
            assert not releases
            if windows == 2 and failure == "exception":
                raise RuntimeError("paired forward failed")
            low = float("nan") if windows == 2 and failure == "nonfinite" else 0.25
            return mx.concatenate([values, mx.full(values.shape, low, dtype=mx.float32)], axis=0)

    monkeypatch.setattr(mx, "clear_cache", release)
    values = np.arange(6**3, dtype=np.float32).reshape(1, 1, 6, 6, 6)

    def predict() -> Float32Array:
        """Run paired-precision stitching over the synthetic input."""
        return predict_wide_logits(
            cast(SegResNet, PairedNetwork()),
            values,
            SlidingWindowConfig((4, 4, 4), overlap=0.5),
            mx=mx,
        )

    if failure is not None:
        with pytest.raises(RuntimeError, match=r"paired forward failed|non-finite paired logits"):
            predict()
        assert releases == [2]
    else:
        np.testing.assert_array_equal(predict(), values + np.float32(0.25))
        assert releases == [8]


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
def test_predictor_matches_upstream(
    name: str,
    tmp_path: Path,
    mx: MLX,
) -> None:
    """Compare sliding-window logits and batch shapes with recorded MONAI cases."""
    path = FIXTURES / f"predictor_{name}.npz"
    arrays, metadata = load_fixture(path)
    metadata = cast(PredictorMetadata, metadata)
    arrays = cast(NumericArrays, arrays)
    configuration = SlidingWindowConfig.from_bundle_config(metadata["inferer"])
    network: Callable[[Array], Array]
    if "spec" in metadata:
        spec = metadata["spec"]
        spec["blocks_down"] = tuple(spec["blocks_down"])
        spec["blocks_up"] = tuple(spec["blocks_up"])
        network = SegResNet(
            SegResNetSpec(**cast(ConstructorSpec, spec)),
            {
                key.removeprefix("weight::"): cast(ArrayFactory, mx.array)(value)
                for key, value in arrays.items()
                if key.startswith("weight::")
            },
            mx=mx,
        )
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

    telemetry = WindowTelemetry(tmp_path / "runtime.json", {"model_id": metadata["model_id"]})
    result = predict_logits(arrays["input"], configuration, network=measured, telemetry=telemetry)
    assert result.dtype == np.float32
    assert result.shape == arrays["output"].shape
    # Torch CPU and Metal may fuse/reduce float32 operations differently, including
    # elementwise stand-ins. Seeded raw logits must still agree within 1e-6.
    np.testing.assert_allclose(result, arrays["output"], rtol=0, atol=1e-6)
    assert np.array_equal(np.array(batches), arrays["window_batches"])
    record = json.loads(telemetry.path.read_text())
    for key in ("status", "window_count", "window_samples_complete", "dtype", "model_id"):
        expected = {
            "status": "completed",
            "window_count": len(batches),
            "window_samples_complete": True,
            "dtype": "float32",
            "model_id": metadata["model_id"],
        }
        assert record[key] == expected[key], key
    assert len(record["window_seconds"]) == len(batches)
    assert record["schema_version"] == "medmlx.segresnet.mlx.v1"
    assert record["sw_batch_size"] == configuration.sw_batch_size
    assert record["synchronization"] == "mx.eval and mx.synchronize before host export"


def test_caller_owned_telemetry(tmp_path: Path) -> None:
    # The engine callback must stay running until postprocessing completes.
    """Keep caller-owned telemetry open until the caller finishes it."""
    telemetry = WindowTelemetry(tmp_path / "runtime.json", {})
    predictor = BundlePredictor(lambda x: x[:, :1], telemetry=telemetry)
    host = np.ones((2, 3, 3, 3, 3), dtype=np.float32)
    result = predictor.predict_logits(host, SlidingWindowConfig((2, 2, 2), 4))
    assert np.array_equal(result, host[:, :1])
    assert telemetry.record["status"] == "running"
    assert telemetry.record["window_count"] == 4
    telemetry.finish()
    assert telemetry.record["status"] == "completed"


@pytest.mark.parametrize(
    "invalid",
    ["dtype", "nonfinite", "channels", "rank", "output", "output_shape"],
)
def test_predictor_failures_finalize_telemetry(
    invalid: str,
    no_decoder_reference: ModelFixture,
    tmp_path: Path,
    mx: MLX,
) -> None:
    """Record a failed session for invalid inputs and invalid network outputs."""
    spec, arrays, _ = no_decoder_reference
    network: Callable[[Array], Array] = SegResNet(
        spec,
        {
            key.removeprefix("weight::"): cast(ArrayFactory, mx.array)(value)
            for key, value in arrays.items()
            if key.startswith("weight::")
        },
        mx=mx,
    )
    host: HostArray = arrays["input"].copy()
    if invalid == "dtype":
        host = host.astype(np.float16)
        error, message = InvalidInputError, "input must be finite float32"
    elif invalid == "nonfinite":
        host.flat[-1] = np.inf
        error, message = InvalidInputError, "input must be finite float32"
    elif invalid == "channels":
        host = host[:, :1]
        error, message = ValueError, "has 1 channels; expected 4"
    elif invalid == "rank":
        host = host[0]
        error, message = ValueError, "must be NCDHW rank-5"
    elif invalid == "output":

        def nonfinite(x: Array) -> Array:
            """Produce non-finite logits to exercise output admission."""
            return mx.full(x.shape, float("nan"), dtype=mx.float32)

        network = nonfinite
        error, message = RuntimeError, "produced non-finite logits"
    else:

        def clipped(x: Array) -> Array:
            """Produce a spatially truncated window to exercise shape admission."""
            return x[:, :, :-1]

        network = clipped
        error, message = ValueError, "output must match the window batch and ROI spatial shape"
    telemetry = WindowTelemetry(tmp_path / "runtime.json", {})
    with pytest.raises(error, match=message):
        predict_logits(host, SlidingWindowConfig((4, 6, 8)), network=network, telemetry=telemetry)
    record = json.loads(telemetry.path.read_text())
    assert record["status"] == "failed"
    assert record["error_type"] == error.__name__
    assert record["window_count"] == (1 if invalid == "output_shape" else 0)
    assert record["window_samples_complete"] is True
