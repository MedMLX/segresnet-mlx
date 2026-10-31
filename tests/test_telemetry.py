"""Progress snapshots, summary statistics and terminal lifecycle for package telemetry."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, cast

from fixture_contracts import IgniteRunner

if TYPE_CHECKING:
    from ignite.engine import Engine

import numpy as np
import pytest

from segresnet_mlx import WindowTelemetry


def test_telemetry_lifecycle_and_statistics(tmp_path: Path) -> None:
    """Check progress snapshots, final quantiles, failures, and session reuse."""
    timings = np.linspace(0.01, 2.0, 65)
    peaks = np.arange(65, dtype=np.int64) * 128
    telemetry = WindowTelemetry(tmp_path / "runtime.json", {"model_id": "synthetic"})
    telemetry.set_metadata({"dtype": "float32"})
    assert json.loads(telemetry.path.read_text())["status"] == "running"
    for seconds, peak in zip(timings, peaks, strict=True):
        telemetry.observe(float(seconds), int(peak))
    snapshot = json.loads(telemetry.path.read_text())
    assert snapshot["window_count"] == 64  # Bounded snapshot cadence.
    assert not snapshot["window_samples_complete"]
    assert "window_seconds" not in snapshot
    telemetry.finish()
    final = json.loads(telemetry.path.read_text())
    assert final["schema_version"] == "medmlx.segresnet.mlx.v1"
    assert final["window_count"] == 65
    assert final["window_seconds"] == list(timings)
    assert final["window_samples_complete"]
    assert final["window_total_seconds"] == pytest.approx(timings.sum(), abs=1e-14)
    assert final["window_max_seconds"] == 2.0
    assert final["peak_mlx_bytes"] == int(peaks.max())
    assert final["window_p50_seconds"] == np.percentile(timings, 50)
    assert final["window_p95_seconds"] == np.percentile(timings, 95)
    telemetry.begin()
    telemetry.observe(0.25, None)
    telemetry.fail(ValueError("synthetic failure"))
    telemetry.finish()
    failed = json.loads(telemetry.path.read_text())
    assert failed["status"] == "failed"
    assert failed["error_type"] == "ValueError"
    assert failed["window_seconds"] == [0.25]
    assert failed["peak_mlx_bytes"] is None
    with telemetry.session():
        telemetry.observe(0.5, 1024)
    completed = json.loads(telemetry.path.read_text())
    assert completed["status"] == "completed"
    assert completed["window_seconds"] == [0.5]
    assert "error_type" not in completed


def test_record_write_failure_preserves_inference_error(tmp_path: Path) -> None:
    """Preserve the inference exception if final telemetry cannot be written."""
    path = tmp_path / "runtime.json"
    telemetry = WindowTelemetry(path, {})
    failure = RuntimeError("non-finite logits")
    with pytest.raises(RuntimeError, match="non-finite logits") as raised, telemetry.session():
        path.unlink()
        path.mkdir()
        raise failure
    assert raised.value is failure
    assert any("Could not finalize SegResNet telemetry" in note for note in failure.__notes__)
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize("duration", [float("nan"), float("inf"), -1.0])
def test_invalid_timings_finalize_as_failed(tmp_path: Path, duration: float) -> None:
    """Reject invalid durations and finalize the session as failed."""
    path = tmp_path / "runtime.json"
    telemetry = WindowTelemetry(path, {})
    with pytest.raises(ValueError, match="finite and nonnegative"), telemetry.session():
        telemetry.observe(duration, None)
    final = json.loads(path.read_text())
    assert final["status"] == "failed"
    assert final["window_seconds"] == []
    assert final["window_p50_seconds"] is None
    assert final["window_p95_seconds"] is None


@pytest.mark.parametrize("outcome", ["completed", "failed", "interrupted"])
def test_real_ignite_engine_lifecycle(tmp_path: Path, outcome: str) -> None:
    """Follow completed, failed, and interrupted Ignite engine events."""
    pytest.importorskip("ignite.engine")
    import ignite.engine as ignite

    path = tmp_path / "runtime.json"
    telemetry = WindowTelemetry(path, {})
    telemetry.set_metadata({"checkpoint": "loaded"})
    telemetry.observe(100.0, 8192)  # Warmup is excluded when the evaluator starts.
    failure = RuntimeError("postprocessing failed")

    def step(engine: Engine, batch: object) -> None:
        """Record one window and trigger the requested engine outcome."""
        telemetry.observe(0.5, 1024)
        if outcome == "failed":
            raise failure
        if outcome == "interrupted":
            engine.terminate()

    engine = ignite.Engine(step)
    telemetry.attach(engine, events=ignite.Events)
    if outcome == "failed":
        with pytest.raises(RuntimeError) as raised:
            cast(IgniteRunner, engine).run([0, 1])
        assert raised.value is failure
    else:
        cast(IgniteRunner, engine).run([0, 1])
    record = json.loads(path.read_text())
    assert record["checkpoint"] == "loaded"
    assert record["window_seconds"] == [0.5] * (2 if outcome == "completed" else 1)
    assert record["status"] == outcome
    assert record["window_count"] == (2 if outcome == "completed" else 1)
    assert record["window_samples_complete"]
