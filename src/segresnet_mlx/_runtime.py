"""Typed native import after the shared runtime's existing admission checks."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from medmlx_core.runtime import import_mlx as _import_mlx

if TYPE_CHECKING:
    from segresnet_mlx._types import MLX


def import_mlx() -> MLX:
    """Return ``mlx.core`` with the Metal GPU as the default device.

    The host checks and device selection are done by ``medmlx_core``; on an
    unsupported host it raises ``MissingDependencyError`` and there is no CPU fallback.
    """
    _import_mlx()
    import mlx.core as mx

    return cast("MLX", mx)
