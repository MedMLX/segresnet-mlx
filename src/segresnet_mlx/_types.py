"""Static native MLX contracts; runtime imports remain lazy."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence
    from typing import Literal, Protocol, TypedDict

    import mlx.core as mx
    from medmlx_core.typing import MlxRuntime
    from medmlx_core.volume_workspace import VolumeWorkspace

    from segresnet_mlx._arrays import Float32Array, HostArray

    type Array = mx.array

    class WorkspaceOptions(TypedDict, total=False):
        workspace: VolumeWorkspace

    type ArrayValue = bool | int | float | complex | list[ArrayValue] | tuple[ArrayValue, ...]

    class ArrayConstructor(Protocol):
        def __call__(
            self, value: HostArray | Array | ArrayValue, dtype: mx.Dtype | None = None
        ) -> Array:
            """Construct an MLX array with an optional dtype."""
            ...

    class MLX(MlxRuntime, Protocol):
        all = staticmethod(mx.all)
        isfinite = staticmethod(mx.isfinite)
        full = staticmethod(mx.full)
        sum = staticmethod(mx.sum)
        ones = staticmethod(mx.ones)
        synchronize = staticmethod(mx.synchronize)
        clear_cache = staticmethod(mx.clear_cache)
        load = staticmethod(mx.load)

        def eval(self, *args: Array | Mapping[str, Array] | Sequence[Array]) -> None: ...
        def save_safetensors(self, file: str, arrays: dict[str, Array]) -> None: ...

    class SlidingWindowInference(Protocol):
        def __call__(
            self,
            inputs: Float32Array,
            roi_size: Sequence[int],
            sw_batch_size: int,
            predictor: Callable[[Float32Array | Array], Float32Array]
            | Callable[[Float32Array | Array, list[list[slice]]], Float32Array],
            *,
            overlap: float,
            mode: str,
            padding_mode: str,
            with_coord: bool = False,
            predictor_input: Literal["numpy"],
            preserve_window_strides: bool = False,
            workspace: VolumeWorkspace | None = None,
        ) -> Float32Array:
            """Blend predictor outputs from overlapping windows into host logits."""
            ...

    class CoreSlidingWindow(Protocol):
        sliding_window_inference: SlidingWindowInference

    class Metal(Protocol):
        def is_available(self) -> bool:
            """Report whether the Metal backend is available."""
            ...
