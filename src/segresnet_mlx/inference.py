"""Synchronized single-patch inference with NumPy inputs and raw-logit outputs."""

from __future__ import annotations

from typing import cast

import numpy as np
from medmlx_core.errors import InvalidInputError
from medmlx_core.typing import ArrayFactory

from segresnet_mlx._arrays import Float32Array, HostArray
from segresnet_mlx._runtime import import_mlx
from segresnet_mlx.network import SegResNet


def forward_patch(network: SegResNet, values: HostArray) -> Float32Array:
    """Run one finite float32 ``(N, C, D, H, W)`` patch without transforms or stitching.

    Returns float32 NumPy logits ``(N, out_channels, D, H, W)``, synchronized and
    checked for finiteness.

    Raises:
        InvalidInputError: If ``values`` is not finite float32.
        RuntimeError: If the network produced non-finite logits.
    """
    host = np.asarray(values)
    if host.dtype != np.float32 or not np.isfinite(host).all():
        raise InvalidInputError("SegResNet input must be finite float32")
    mx = import_mlx()
    mx.synchronize()
    result = network(cast(ArrayFactory, mx.array)(host))
    mx.eval(result)
    mx.synchronize()
    output = cast(Float32Array, np.array(result))
    if not np.isfinite(output).all():
        raise RuntimeError("SegResNet produced non-finite logits")
    return output
