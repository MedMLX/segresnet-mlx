"""Static native MLX contracts; runtime imports remain lazy."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from typing import Literal, Protocol, TypedDict

    import mlx.core as mx
    from medmlx_core.typing import MlxRuntime
    from medmlx_core.volume_workspace import VolumeWorkspace

    type Array = mx.array

    class WorkspaceOptions(TypedDict, total=False):
        workspace: VolumeWorkspace

    class AccumulationOptions(TypedDict, total=False):
        accumulation: Literal["mlx"]

    class MLX(MlxRuntime, Protocol):
        all = staticmethod(mx.all)
        isfinite = staticmethod(mx.isfinite)
        synchronize = staticmethod(mx.synchronize)
        clear_cache = staticmethod(mx.clear_cache)
        load = staticmethod(mx.load)

        def eval(self, *args: Array | Mapping[str, Array] | Sequence[Array]) -> None: ...
        def save_safetensors(self, file: str, arrays: dict[str, Array]) -> None: ...
