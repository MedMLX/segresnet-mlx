"""Host array dtypes at model, checkpoint and image boundaries."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, cast

import numpy as np

if TYPE_CHECKING:
    from segresnet_mlx._types import Array

type HostArray = np.ndarray[tuple[int, ...], np.dtype[np.generic]]
type Float32Array = np.ndarray[tuple[int, ...], np.dtype[np.float32]]
type Float64Array = np.ndarray[tuple[int, ...], np.dtype[np.float64]]
type Int64Array = np.ndarray[tuple[int, ...], np.dtype[np.int64]]
type MaskArray = Float32Array | Int64Array

type NumericScalar = (
    np.bool_
    | np.int8
    | np.int16
    | np.int32
    | np.int64
    | np.uint8
    | np.uint16
    | np.uint32
    | np.uint64
    | np.float16
    | np.float32
    | np.float64
    | np.complex64
    | np.complex128
)
type NumericArray = np.ndarray[tuple[int, ...], np.dtype[NumericScalar]]
as_numpy = cast("Callable[[Array], NumericArray]", np.array)
