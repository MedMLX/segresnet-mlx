"""Bounded progress records and explicit SegResNet inference lifecycle."""

from __future__ import annotations

import json
import math
import os
import tempfile
from collections.abc import Callable, Generator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol, cast

from segresnet_mlx._records import Json

if TYPE_CHECKING:
    from ignite.engine import Events

import numpy as np


class EventEngine(Protocol):
    """Structural type for the Ignite engine that receives telemetry handlers."""

    def add_event_handler(self, event_name: Events, handler: Callable[..., None]) -> object:
        """Register a handler for an Ignite lifecycle event."""
        ...


SCHEMA_VERSION = "medmlx.segresnet.mlx.v1"


class WindowTelemetry:
    """Atomic JSON progress record for sliding-window inference, times in seconds.

    ``record`` is updated in place and written to ``path`` (parent directories are
    created). While running it holds window counts and totals, refreshed after the
    first window and then every 32nd. The raw ``window_seconds`` samples and their
    p50/p95 are written only when the session finishes: samples stay in memory so the
    growing list is not rewritten per window.
    """

    def __init__(self, path: Path, record: dict[str, Json]) -> None:
        """Create the output directory and initialize the caller-owned progress record."""
        self.path = path
        self.record = record
        self._metadata = {"schema_version": SCHEMA_VERSION, **record}
        self.timings: list[float] = []
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.begin()

    def set_metadata(self, values: dict[str, Json]) -> None:
        """Merge run-level metadata that is kept across ``begin`` calls."""
        self._metadata.update(values)
        self.record.update(values)
        self._write()

    def begin(self) -> None:
        """Discard samples and start a new running record (``__init__`` calls this)."""
        self.timings = []
        self.record.clear()
        self.record.update(self._metadata)
        self.record.update(
            status="running",
            window_count=0,
            window_total_seconds=0.0,
            window_max_seconds=None,
            peak_mlx_bytes=None,
            window_samples_complete=False,
        )
        self._write()

    def observe(self, seconds: float, peak_bytes: int | None) -> None:
        """Add one window's duration (seconds) and the MLX peak memory so far (bytes).

        Raises:
            RuntimeError: If the session is closed.
            ValueError: If ``seconds`` is negative or not finite.
        """
        if self.record["status"] != "running":
            raise RuntimeError("SegResNet telemetry is closed; begin a new session")
        if not math.isfinite(seconds) or seconds < 0:
            raise ValueError("Window duration must be finite and nonnegative")
        self.timings.append(seconds)
        self.record["window_count"] = len(self.timings)
        self.record["window_total_seconds"] = (
            cast(float, self.record["window_total_seconds"]) + seconds
        )
        self.record["window_max_seconds"] = max(
            cast(float | None, self.record["window_max_seconds"]) or 0, seconds
        )
        if peak_bytes is not None:
            self.record["peak_mlx_bytes"] = max(
                cast(int | None, self.record["peak_mlx_bytes"]) or 0, peak_bytes
            )
        # Constant-size snapshots. Quantiles and full raw samples are final-only.
        if len(self.timings) == 1 or len(self.timings) % 32 == 0:
            self._write()

    def finish(self, status: Literal["completed", "failed", "interrupted"] = "completed") -> None:
        """Close the record with ``status`` and write the raw samples and quantiles.

        Does nothing if the record already has that status or has failed, or if a
        completed finish follows an interruption.
        """
        previous = self.record["status"]
        if previous in {status, "failed"} or (previous == "interrupted" and status == "completed"):
            return
        p50, p95 = np.percentile(self.timings, [50, 95]) if self.timings else (None, None)
        self.record.update(
            status=status,
            window_seconds=list(self.timings),
            window_samples_complete=True,
            window_p50_seconds=None if p50 is None else float(p50),
            window_p95_seconds=None if p95 is None else float(p95),
        )
        self._write()

    def fail(self, error: BaseException) -> None:
        """Record ``error``'s type and finish as failed, without masking ``error``."""
        self.record["error_type"] = type(error).__name__
        try:
            self.finish("failed")
        except Exception as record_error:
            # An unavailable output directory must not replace the inference error.
            error.add_note(f"Could not finalize SegResNet telemetry: {record_error!r}")

    @contextmanager
    def session(self) -> Generator[None]:
        """Use around direct inferer calls that are not owned by a MONAI engine."""
        self.begin()
        try:
            yield
            self.finish()
        except BaseException as error:
            self.fail(error)
            raise

    def attach(self, engine: object, *, events: type[Events]) -> None:
        """Attach caller-provided engine events before handlers that re-raise failures."""
        attached = cast(EventEngine, engine)
        attached.add_event_handler(events.STARTED, self._started)
        attached.add_event_handler(events.COMPLETED, self._completed)
        attached.add_event_handler(events.EXCEPTION_RAISED, self._failed)
        for event in (events.TERMINATE, events.TERMINATE_SINGLE_EPOCH, events.INTERRUPT):
            attached.add_event_handler(event, self._interrupted)

    def _started(self, engine: object) -> None:
        """Start a fresh record when the attached engine starts."""
        self.begin()

    def _completed(self, engine: object) -> None:
        """Finalize a completed engine run."""
        self.finish()

    def _failed(self, engine: object, error: BaseException) -> None:
        """Record and re-raise an engine failure."""
        self.fail(error)
        # Registering EXCEPTION_RAISED otherwise makes Ignite swallow the error.
        raise error

    def _interrupted(self, engine: object, **kwargs: object) -> None:
        """Finalize an interrupted engine run."""
        self.finish("interrupted")

    def _write(self) -> None:
        """Replace the progress file atomically with the current JSON record."""
        # Atomic visibility for progress readers, not a crash-durability promise.
        descriptor, name = tempfile.mkstemp(prefix=f".{self.path.name}.", dir=self.path.parent)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(self.record, handle, indent=2, allow_nan=False)
                handle.write("\n")
            os.replace(name, self.path)
        finally:
            Path(name).unlink(missing_ok=True)
