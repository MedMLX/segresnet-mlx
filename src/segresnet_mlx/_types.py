"""Static native MLX contracts; runtime imports remain lazy."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Literal, TypedDict

    import mlx.core as mx
    from medmlx_core.volume_workspace import VolumeWorkspace

    type Array = mx.array

    class WorkspaceOptions(TypedDict, total=False):
        workspace: VolumeWorkspace

    class AccumulationOptions(TypedDict, total=False):
        accumulation: Literal["mlx"]
