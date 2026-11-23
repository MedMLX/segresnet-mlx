"""Measured MLX window execution and NumPy full-volume raw-logit inference."""

from __future__ import annotations

import time
from collections.abc import Callable, Generator
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, cast

from segresnet_mlx._records import BundleConfig, InfererConfig
from segresnet_mlx._runtime import import_mlx

if TYPE_CHECKING:
    from ignite.engine import Engine, Events
    from medmlx_core.volume_workspace import VolumeWorkspace

    from segresnet_mlx._types import (
        MLX,
        AccumulationOptions,
        Array,
        WorkspaceOptions,
    )
    from segresnet_mlx.network import SegResNet

import numpy as np
from medmlx_core.errors import InvalidInputError
from medmlx_core.layout import require_ncdhw
from medmlx_core.typing import ArrayFactory, HostArray

from segresnet_mlx._arrays import Float32Array
from segresnet_mlx.medmlx import BUNDLES
from segresnet_mlx.telemetry import SCHEMA_VERSION, WindowTelemetry


@dataclass(frozen=True, slots=True)
class SlidingWindowConfig:
    """The bundle's ``SlidingWindowInferer`` settings; other options use MONAI defaults.

    Attributes:
        roi_size: Window size in voxels, ``(D, H, W)`` of the preprocessed volume.
        sw_batch_size: Windows sent to the network per call.
        overlap: Fraction of each window shared with its neighbor, in ``[0, 1)``.
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
    """Native MLX window callback with an injected network.

    ``forward`` retains the MLX-array callback API; ``predict_logits`` exports
    host logits or opts into FP32 device slab stitching. Telemetry
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
                        "synchronization": "mx.eval; global synchronization only for telemetry",
                        "window_timing_scope": (
                            "synchronized network and MLX input creation; host export only "
                            "for host-output callbacks; excludes validation and telemetry"
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

        Device inputs and logits stay in MLX; finite checks read one Boolean.
        """
        try:
            if isinstance(values, self.mx.array):
                if values.dtype != self.mx.float32 or not bool(
                    self.mx.all(self.mx.isfinite(values)).item()
                ):
                    raise InvalidInputError(f"{self.network_label} input must be finite float32")
                validated = values
            else:
                validated = self._host_input(values)
            with self._measure_window():
                device_input = (
                    validated if isinstance(validated, self.mx.array)
                    else cast(ArrayFactory, self.mx.array)(validated)
                )
                return self._forward_validated_mlx(device_input)
        except BaseException as error:
            if self.telemetry is not None:
                self.telemetry.fail(error)
            raise

    def _forward_window(self, values: HostArray | Array) -> Float32Array:
        """Validate one standalone window and explicitly export finite host logits."""
        return self._forward_validated_window(self._host_input(values))

    def _forward_validated_mlx(self, values: Array) -> Array:
        """Run validated float32 MLX input and check logits on device."""
        logits = self.network(values)
        finite = self.mx.all(self.mx.isfinite(logits))
        self.mx.eval(logits, finite)
        if not bool(finite.item()):
            raise RuntimeError(f"{self.network_label} produced non-finite logits")
        return logits

    @contextmanager
    def _measure_window(self) -> Generator[None]:
        """Synchronize and time completed window work only when telemetry is active."""
        if self.telemetry is None:
            yield
            return
        self.mx.synchronize()
        started = time.perf_counter()
        yield
        self.mx.synchronize()
        elapsed = time.perf_counter() - started
        self.telemetry.observe(elapsed, self.mx.get_peak_memory())

    def _forward_validated_window(self, values: Float32Array) -> Float32Array:
        """Export one already-validated host patch without rescanning its input."""
        with self._measure_window():
            logits = self._forward_validated_mlx(cast(ArrayFactory, self.mx.array)(values))
            return cast(Float32Array, np.array(logits))

    def _forward_device_window(self, values: Array) -> Array:
        """Run a patch of a validated volume, retaining its logits on device."""
        with self._measure_window():
            return self._forward_validated_mlx(values)

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
        accumulation: Literal["numpy", "mlx"] = "numpy",
    ) -> Float32Array:
        """Stitch preprocessed ``(N, C, D, H, W)`` float32 data into full-volume logits.

        Returns float32 NumPy ``(N, out_channels, D, H, W)`` at the input spatial
        size. Volumes smaller than ``roi_size`` are padded and cropped back. Window
        placement and blending follow MONAI's ``SlidingWindowInferer`` via the
        shared core. Callers own telemetry completion.

        ``accumulation='mlx'`` opts into FP32 Metal slabs and completed-slab
        export. The default uses host blending; the wide-precision API remains
        a separate float64 host route.

        Raises:
            InvalidInputError: If ``values`` is not finite float32.
            ValueError: If ``values`` is not rank 5.
        """
        from medmlx_core.sliding_window import sliding_window_inference

        try:
            if accumulation not in {"numpy", "mlx"}:
                raise ValueError("accumulation must be 'numpy' or 'mlx'")
            host = self._host_input(values)
            require_ncdhw(host, name="SegResNet input")
            if self.telemetry is not None:
                self.telemetry.set_metadata({
                    "sw_batch_size": configuration.sw_batch_size,
                    "transforms_and_stitching_device": (
                        "cpu transforms, Metal stitching" if accumulation == "mlx" else "cpu"
                    ),
                })
            if (
                workspace is None
                and host.shape[0] == 1
                and host.shape[1] > 0
                and configuration.mode == "constant"
                and configuration.padding_mode == "constant"
                and configuration.sw_batch_size >= 1
                and 0 <= configuration.overlap < 1
                and all(
                    roi > 0 and 0 < size <= roi
                    for size, roi in zip(host.shape[2:], configuration.roi_size, strict=True)
                )
            ):
                padding = tuple(
                    ((roi - size) // 2, (roi - size + 1) // 2)
                    for size, roi in zip(host.shape[2:], configuration.roi_size, strict=True)
                )
                if any(before or after for before, after in padding):
                    window = np.ascontiguousarray(
                        np.pad(host, ((0, 0), (0, 0), *padding), mode="constant")
                    )
                else:
                    window = host
                crop = tuple(
                    slice(before, before + size)
                    for (before, _), size in zip(padding, host.shape[2:], strict=True)
                )
                # One constant-weight window contributes its logits exactly once;
                # blending would only copy and divide the same full volume.
                return self._forward_validated_window(window)[(slice(None), slice(None), *crop)]
            accumulation_options: AccumulationOptions = (
                {"accumulation": "mlx"} if accumulation == "mlx" else {}
            )
            return sliding_window_inference(
                host,
                configuration.roi_size,
                configuration.sw_batch_size,
                self._forward_device_window if accumulation == "mlx"
                else self._forward_validated_window,
                overlap=configuration.overlap,
                mode=configuration.mode,
                padding_mode=configuration.padding_mode,
                predictor_input="mlx" if accumulation == "mlx" else "numpy",
                # MLX reduction kernels depend on layout. Match MONAI's sliced
                # batch-one input without a second padded volume or patch copy.
                preserve_window_strides=True,
                workspace=workspace,
                **accumulation_options,
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
    accumulation: Literal["numpy", "mlx"] = "numpy",
) -> Float32Array:
    """Return float32 NumPy NCDHW logits, completing optional window telemetry.

    ``values`` is preprocessed, finite float32 ``(N, C, D, H, W)``; the result has the
    same spatial size. Input preparation, activation, label mapping, image I/O and QC
    are external. ``accumulation='mlx'`` opts into FP32 Metal slab blending and
    requires a shared core with that accumulation option.
    """
    with telemetry.session() if telemetry is not None else nullcontext():
        predictor = BundlePredictor(network, telemetry=telemetry)
        return predictor.predict_logits(values, configuration, accumulation=accumulation)


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
    explicit precision. The learned graph runs on Metal; host work is stitching.
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
