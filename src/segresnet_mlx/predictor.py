"""Measured MLX window execution and NumPy full-volume raw-logit inference."""

from __future__ import annotations

import time
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

from segresnet_mlx._records import BundleConfig, InfererConfig
from segresnet_mlx._runtime import import_mlx

if TYPE_CHECKING:
    from ignite.engine import Engine, Events
    from medmlx_core.volume_workspace import VolumeWorkspace

    from segresnet_mlx._types import MLX, Array, CoreSlidingWindow, WorkspaceOptions
    from segresnet_mlx.network import SegResNet

import numpy as np
from medmlx_core.errors import InvalidInputError
from medmlx_core.layout import require_ncdhw
from medmlx_core.typing import ArrayFactory

from segresnet_mlx._arrays import Float32Array, HostArray
from segresnet_mlx.medmlx import BUNDLES
from segresnet_mlx.telemetry import SCHEMA_VERSION, WindowTelemetry


@dataclass(frozen=True, slots=True)
class SlidingWindowConfig:
    """The bundle's ``SlidingWindowInferer`` settings; other options use MONAI defaults.

    Attributes:
        roi_size: Window size in voxels, ``(D, H, W)`` of the preprocessed volume.
        sw_batch_size: Windows sent to the network per call.
        overlap: Fraction of each window shared with its neighbour, in ``[0, 1)``.
        mode: Window blending, ``"constant"`` or ``"gaussian"`` (sigma scale 0.125).
        padding_mode: How a volume smaller than ``roi_size`` is padded: ``"constant"``
            (zeros), ``"reflect"``, ``"replicate"`` or ``"circular"``.
    """

    roi_size: tuple[int, int, int]
    sw_batch_size: int = 1
    overlap: float = 0.25
    mode: str = "constant"
    padding_mode: str = "constant"

    @classmethod
    def from_bundle_config(cls, config: InfererConfig | BundleConfig) -> SlidingWindowConfig:
        """Read the staged JSON's inferer section without evaluating bundle expressions.

        ``roi_size`` is required; the other listed keys default to MONAI's
        values. Values are taken literally, so a reference such as
        ``@displayable_configs#sw_overlap`` is not resolved.
        """
        inferer = cast(InfererConfig, config.get("inferer", config))
        return cls(
            cast(tuple[int, int, int], tuple(inferer["roi_size"])),
            inferer.get("sw_batch_size", 1),
            inferer.get("overlap", 0.25),
            inferer.get("mode", "constant"),
            inferer.get("padding_mode", "constant"),
        )


# Window settings of each bundle in ``segresnet_mlx.medmlx.BUNDLES``.
BUNDLE_CONFIGS = {
    name: SlidingWindowConfig(
        bundle.roi_size, bundle.sw_batch_size, bundle.overlap, bundle.mode, bundle.padding_mode
    )
    for name, bundle in BUNDLES.items()
}


class BundlePredictor:
    """Synchronized MLX window callback with an injected network.

    ``forward`` retains the MLX-array callback API; ``predict_logits`` passes
    synchronized NumPy logits directly to the host stitcher. Telemetry
    lifecycle remains caller-owned here, allowing an engine to finish only after
    postprocessing. The module-level ``predict_logits`` owns a direct-call session.
    """

    schema_version = SCHEMA_VERSION
    network_label = "SegResNet"

    def __init__(
        self,
        network: Callable[[Array], Array],
        *,
        telemetry: WindowTelemetry | None = None,
        mx: MLX | None = None,
    ) -> None:
        """Initialize the network runtime and optional per-window telemetry."""
        self.network = network
        self.telemetry = telemetry
        try:
            self.mx = import_mlx() if mx is None else mx
            self.mx.reset_peak_memory()
            if telemetry is not None:
                telemetry.set_metadata(
                    {
                        "schema_version": self.schema_version,
                        "dtype": "float32",
                        "network_device": str(self.mx.default_device()),
                        "transforms_and_stitching_device": "cpu",
                        "compile": False,
                        "synchronization": "mx.eval and mx.synchronize before host export",
                        "window_timing_scope": (
                            "synchronized network, MLX input creation and host export; "
                            "excludes validation and telemetry"
                        ),
                        "peak_memory_scope": "predictor lifetime, including warmup",
                    }
                )
        except BaseException as error:
            if telemetry is not None:
                telemetry.fail(error)
            raise

    def attach(self, engine: Engine, *, events: type[Events]) -> None:
        """Attach the telemetry (if any) to an Ignite ``engine`` using the caller's ``events``."""
        if self.telemetry is not None:
            self.telemetry.attach(engine, events=events)

    def __call__(self, values: HostArray | Array) -> Array:
        """Run one patch through the same validated path as ``forward``."""
        return self.forward(values)

    def forward(self, values: HostArray | Array) -> Array:
        """Run one finite float32 window and return its logits as an MLX array.

        The logits are exported to host NumPy for the finite check and timing, then
        wrapped again as an MLX array for callers that expect one.
        """
        try:
            return cast(ArrayFactory, self.mx.array)(self._forward_window(values))
        except BaseException as error:
            if self.telemetry is not None:
                self.telemetry.fail(error)
            raise

    def _forward_window(self, values: HostArray | Array) -> Float32Array:
        """Run and synchronize one window, returning finite host logits."""
        host = self._host_input(values)
        self.mx.synchronize()
        started = time.perf_counter()
        result = self.network(cast(ArrayFactory, self.mx.array)(host))
        self.mx.eval(result)
        self.mx.synchronize()
        output = cast(Float32Array, np.array(result))
        elapsed = time.perf_counter() - started
        if not np.isfinite(output).all():
            raise RuntimeError(f"{self.network_label} produced non-finite logits")
        if self.telemetry is not None:
            self.telemetry.observe(elapsed, self.mx.get_peak_memory())
        # The allocator cache is kept across windows: every window has the same
        # shapes, so later windows reuse the buffers instead of reallocating
        # them. ``predict_logits`` releases the cache once the volume is done.
        del result
        return output

    def _host_input(self, values: HostArray | Array) -> Float32Array:
        """Admit finite float32 input as a NumPy array or raise an input error."""
        host = np.asarray(values)
        if host.dtype != np.float32 or not np.isfinite(host).all():
            raise InvalidInputError(f"{self.network_label} input must be finite float32")
        return cast(Float32Array, host)

    def predict_logits(
        self,
        values: HostArray | Array,
        configuration: SlidingWindowConfig,
        *,
        workspace: VolumeWorkspace | None = None,
    ) -> Float32Array:
        """Stitch preprocessed ``(N, C, D, H, W)`` float32 data into full-volume logits.

        Returns float32 NumPy ``(N, out_channels, D, H, W)`` at the input spatial
        size. Volumes smaller than ``roi_size`` are padded and cropped back. Window
        placement and blending follow MONAI's ``SlidingWindowInferer`` via the
        shared core. Callers own telemetry completion.

        Raises:
            InvalidInputError: If ``values`` is not finite float32.
            ValueError: If ``values`` is not rank 5.
        """
        import medmlx_core.sliding_window as core_sliding_window

        sliding_window_inference = cast(
            "CoreSlidingWindow", core_sliding_window
        ).sliding_window_inference

        try:
            host = self._host_input(values)
            require_ncdhw(host, name="SegResNet input")
            if self.telemetry is not None:
                self.telemetry.set_metadata({"sw_batch_size": configuration.sw_batch_size})
            return sliding_window_inference(
                host,
                configuration.roi_size,
                configuration.sw_batch_size,
                self._forward_window,
                overlap=configuration.overlap,
                mode=configuration.mode,
                padding_mode=configuration.padding_mode,
                predictor_input="numpy",
                # MLX reduction kernels depend on layout. Match MONAI's sliced
                # batch-one input without a second padded volume or patch copy.
                preserve_window_strides=True,
                workspace=workspace,
            )
        except BaseException as error:
            if self.telemetry is not None:
                self.telemetry.fail(error)
            raise
        finally:
            self.mx.clear_cache()


def predict_logits(
    values: HostArray | Array,
    configuration: SlidingWindowConfig,
    *,
    network: Callable[[Array], Array],
    telemetry: WindowTelemetry | None = None,
) -> Float32Array:
    """Return float32 NumPy NCDHW logits, completing optional window telemetry.

    ``values`` is preprocessed, finite float32 ``(N, C, D, H, W)``; the result has the
    same spatial size. Input preparation, activation, label mapping, image I/O and QC
    are external.
    """
    with telemetry.session() if telemetry is not None else nullcontext():
        predictor = BundlePredictor(network, telemetry=telemetry)
        return predictor.predict_logits(values, configuration)


def predict_wide_logits(
    network: SegResNet,
    values: Float32Array,
    configuration: SlidingWindowConfig,
    *,
    mx: MLX,
    workspace: VolumeWorkspace | None = None,
) -> Float32Array:
    """Carry paired Metal logits into FP64 host blending, rounding only at export.

    Preprocessed values and checkpoint parameters remain the same FP32 inputs.
    The original MONAI float64 network/inferer is the independent oracle for this
    explicit profile. The learned graph runs on Metal; host work is stitching.
    """
    from medmlx_core.sliding_window import sliding_window_inference

    def window(
        patch: np.ndarray[tuple[int, ...], np.dtype[np.float64]],
    ) -> np.ndarray[tuple[int, ...], np.dtype[np.float64]]:
        """Combine paired Metal output words into float64 host logits."""
        result = network.paired_forward(cast(ArrayFactory, mx.array)(patch.astype(np.float32)))
        mx.eval(result)
        mx.synchronize()
        packed = np.array(result, dtype=np.float32)
        batch = packed.shape[0] // 2
        output = packed[:batch].astype(np.float64) + packed[batch:].astype(np.float64)
        if not np.isfinite(output).all():
            raise RuntimeError("SegResNet produced non-finite paired logits")
        del result
        return output

    workspace_options: WorkspaceOptions = {} if workspace is None else {"workspace": workspace}
    try:
        logits = sliding_window_inference(
            values.astype(np.float64),
            configuration.roi_size,
            configuration.sw_batch_size,
            window,
            overlap=configuration.overlap,
            mode=configuration.mode,
            padding_mode=configuration.padding_mode,
            predictor_input="numpy",
            **workspace_options,
        )
        return logits.astype(np.float32) if workspace is None else workspace.float32(logits)
    finally:
        # Reuse same-shaped Metal allocations across windows, including when
        # paired arithmetic requires a wider host accumulator.
        mx.clear_cache()
