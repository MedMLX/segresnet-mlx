"""NumPy ports of the MONAI 1.6 transforms of the SegResNet renal CECT bundle that
``medmlx_core.transforms`` does not provide bit-identically here.

``medmlx_core`` supplies header correction, ``Orientation``, forward ``Spacing``
and ``ScaleIntensityRange``. This module keeps the inverse of ``Spacing``, which
the bundle's ``Invertd`` replays, ``ConcatImages``, ``Activations(sigmoid=True)``,
``AsDiscrete(threshold=0.5)`` and ``MergeClassesd``. Spatial resampling follows
MONAI's ``AffineTransform`` path, that is ``torch.nn.functional.affine_grid`` and
trilinear ``grid_sample`` with border padding and ``align_corners=False``, computed
in float64 as ``Spacing`` does and returned as float32. Core's resampler rounds
some inverse-grid voxels one float32 unit apart from this one, so the inverse
stays local.

Image arrays are channel-first float32 ``(C, *spatial)`` and affines are 4x4 float64
voxel-to-world matrices in mm.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import cast

import numpy as np
import numpy.typing as npt
from medmlx_core.geometry import (
    _bmm as _bmm,
)
from medmlx_core.geometry import (
    _linspace_from_neg_one as _linspace_from_neg_one,
)
from medmlx_core.geometry import (
    _mm as _mm,
)
from medmlx_core.geometry import (
    to_affine_nd as to_affine_nd,
)

from segresnet_mlx._arrays import Int64Array

type Float32Array = npt.NDArray[np.float32]
type Float64Array = npt.NDArray[np.float64]

# monai.data.utils.AFFINE_TOL
AFFINE_TOL = 1e-3
# Bound the float64 corner arrays of all resampling blocks in flight together.
_CHUNK_VOXELS = 1 << 20
_THREADS = min(8, os.cpu_count() or 1)


@dataclass(frozen=True, slots=True)
class Resampling:
    """What ``SpatialResample`` records for its inverse."""

    src_affine: Float64Array
    orig_size: tuple[int, ...]


def invert_spacing(
    data: Float32Array,
    affine: Float64Array,
    record: Resampling,
) -> tuple[Float32Array, Float64Array]:
    """``SpatialResample.inverse``: resample back to the recorded affine and size."""
    result, new_affine, _ = spatial_resample(data, affine, record.src_affine, record.orig_size)
    return result, new_affine


def spatial_resample(
    data: Float32Array,
    src_affine: Float64Array,
    dst_affine: Float64Array,
    spatial_size: Sequence[int],
) -> tuple[Float32Array, Float64Array, Resampling]:
    """``monai.transforms.spatial.functional.spatial_resample``: trilinear, border, float64."""
    rank = min(data.ndim - 1, len(src_affine) - 1, 3, len(spatial_size))
    source = to_affine_nd(rank, src_affine)
    target = to_affine_nd(rank, dst_affine)
    in_size = tuple(int(size) for size in data.shape[1 : rank + 1])
    size = tuple(int(value) for value in spatial_size[:rank])
    record = Resampling(source, in_size)
    xform = np.linalg.solve(source, target)
    same_size = np.allclose(size, in_size)
    unchanged = (np.allclose(source, target, atol=AFFINE_TOL) and same_size) or (
        np.allclose(xform, np.eye(len(xform)), atol=AFFINE_TOL) and same_size
    )
    if unchanged:
        return data.astype(np.float32), np.array(src_affine, dtype=np.float64), record
    values = _grid_sample(data, xform, in_size, size)
    affine = np.array(src_affine, dtype=np.float64)
    new_affine = _mm(affine, to_affine_nd(len(affine) - 1, xform))
    return values, new_affine, record


def concatenate(channels: Sequence[Float32Array]) -> Float32Array:
    """The renal bundle's ``ConcatImages`` (``np.concatenate`` along channels)."""
    return np.concatenate(list(channels))


def sigmoid(data: Float32Array) -> Float32Array:
    """Float32 sigmoid with a stable negative tail and source threshold rounding."""
    one = np.float32(1.0)
    result = np.empty_like(data)
    negative_tail = data < np.float32(-1.0)
    tail = np.exp(data[negative_tail])
    result[negative_tail] = tail / (one + tail)
    # Keep the original expression near zero: equivalent sigmoid formulas can
    # round to opposite sides of the bundle's >= 0.5 threshold there.
    result[~negative_tail] = one / (one + np.exp(-data[~negative_tail]))
    return result


def threshold(data: Float32Array, value: float = 0.5) -> Float32Array:
    """``AsDiscrete(threshold=value)``: ``x >= value`` as float32."""
    return (data >= value).astype(np.float32)


def sigmoid_threshold_mask(data: Float32Array) -> npt.NDArray[np.bool_]:
    """Threshold float32 sigmoid at 0.5 without exponentiating distant logits."""
    sample = data.flat[::1024]
    near_sample = (sample < np.float32(0.0)) & (sample >= np.float32(-0.001))
    # Dense near-zero data costs more to index than to evaluate in one pass.
    if np.count_nonzero(near_sample) > sample.size // 4:
        return sigmoid(data) >= np.float32(0.5)
    active = data >= np.float32(0.0)
    near_negative = (data < np.float32(0.0)) & (data >= np.float32(-0.001))
    # Negative values close to zero can round to sigmoid == 0.5 in float32.
    # At or below -0.001 the result is safely below the threshold.
    active[near_negative] = sigmoid(data[near_negative]) >= np.float32(0.5)
    return active


def merge_classes(binary: Float32Array) -> Float32Array:
    """The renal bundle's ``MergeClassesd``: the lowest nonzero channel index wins.

    Returns a float32 ``(1, *spatial)`` label map; channel 0 contributes nothing.
    """
    weighted = binary * np.arange(binary.shape[0], dtype=np.float32).reshape(
        -1, *([1] * (binary.ndim - 1))
    )
    merged = weighted[0]
    for channel in weighted[1:]:
        merged = merged + channel * ~((merged != 0) & (channel != 0))
    return merged[None].astype(np.float32, copy=False)


def _normalize_transform(shape: Sequence[int]) -> Float64Array:
    """``monai.networks.utils.normalize_transform`` (``align_corners=False``)."""
    size = np.asarray(shape, dtype=np.float64)
    norm = size.copy()
    norm[norm <= 0.0] = 2.0
    norm = 2.0 / norm
    result = np.diag(np.concatenate((norm, np.ones(1, dtype=np.float64))))
    result[:-1, -1] = 1.0 / size - 1.0
    return result


def _theta(
    xform: Float64Array,
    in_size: Sequence[int],
    out_size: Sequence[int],
) -> Float64Array:
    """``AffineTransform`` normalization and reverse indexing of a voxel transform."""
    theta = _bmm(
        _bmm(_normalize_transform(in_size), xform),
        np.linalg.inv(_normalize_transform(out_size)),
    )
    rank = len(in_size)
    reverse = list(range(rank - 1, -1, -1))
    theta[:rank] = theta[reverse]
    theta[:, :rank] = theta[:, reverse]
    return theta[:rank]


def _grid_sample(
    data: Float32Array,
    xform: Float64Array,
    in_size: Sequence[int],
    out_size: Sequence[int],
) -> Float32Array:
    """``affine_grid`` then ``grid_sample`` (border, ``align_corners=False``).

    MONAI samples a float64 copy of the image and rounds the result to float32; here
    each output chunk is computed in float64 from the gathered float32 voxels, which
    gives the same values without a float64 copy of the whole image.
    The grid's four-term dot products follow the reference Torch build's MKL order,
    ``fma(z, t2, x * t0) + fma(1, t3, y * t1)``; with zeros off the diagonal every
    order gives the same coordinates.
    """
    if len(in_size) != 3:
        raise ValueError("SegResNet bundle resampling is three-dimensional")
    theta = _theta(xform, in_size, out_size)
    depth, height, width = (int(size) for size in out_size)
    # Base grid: x follows the last data axis, z the first.
    x = np.broadcast_to(_linspace_from_neg_one(width), (width,))
    y = np.broadcast_to(_linspace_from_neg_one(height), (height,))
    z = np.broadcast_to(_linspace_from_neg_one(depth), (depth,))
    # Per output row: the (z, x) part of the dot product and the y part.
    zx = [_fma_array(z[:, None], theta[row, 2], x[None, :] * theta[row, 0]) for row in range(3)]
    ys = [y * theta[row, 1] + theta[row, 3] for row in range(3)]
    separable = all(
        theta[row, column] == 0 for row in range(3) for column in range(3) if row != column
    )
    result = np.empty((data.shape[0], depth, height, width), dtype=np.float32)
    # Blocks of whole output rows, run on _THREADS threads (NumPy releases the
    # GIL). Each block holds _CHUNK_VOXELS // _THREADS voxels, so the blocks in
    # flight together hold about one serial chunk of float64 temporaries. Every
    # voxel uses the same arithmetic in any split, so the result is bitwise
    # identical to a single-threaded pass.
    budget = max(1, _CHUNK_VOXELS // _THREADS)
    row_voxels = max(1, width * data.shape[0])
    planes = max(1, budget // (height * row_voxels))
    rows = height if planes > 1 else min(height, max(1, budget // row_voxels))

    def fill(start: int, top: int) -> None:
        stop, bottom = min(depth, start + planes), min(height, top + rows)
        if separable:
            gx = zx[0][0] + ys[0][0]
            gy = zx[1][0, 0] + ys[1][top:bottom]
            gz = zx[2][start:stop, 0] + ys[2][0]
            result[:, start:stop, top:bottom] = _sample_separable(data, gx, gy, gz)
        else:
            coordinates = [
                zx[row][start:stop, None, :] + ys[row][None, top:bottom, None] for row in range(3)
            ]
            result[:, start:stop, top:bottom] = _sample(data, *coordinates)

    with ThreadPoolExecutor(max_workers=_THREADS) as pool:
        blocks = [
            pool.submit(fill, start, top)
            for start in range(0, depth, planes)
            for top in range(0, height, rows)
        ]
        for block in blocks:
            block.result()
    return result


def _fma_array(
    left: Float64Array, right: float | Float64Array, addend: Float64Array
) -> Float64Array:
    """Elementwise ``left * right + addend`` with one rounding (error-free transforms)."""
    left, right_value, addend = np.broadcast_arrays(
        np.asarray(left, np.float64), np.asarray(right, np.float64), np.asarray(addend, np.float64)
    )
    product = left * right_value
    if not np.any(right_value) or not np.any(left):
        return product + addend
    error = _product_error(left, right_value, product)
    total = product + addend
    virtual = total - product
    sum_error = (product - (total - virtual)) + (addend - virtual)
    return total + (sum_error + error)


def _split[T: (Float64Array, np.float64)](value: T) -> tuple[T, T]:
    """Split a float64 value into high and low words for Dekker multiplication."""
    scaled = 134217729.0 * value
    high = scaled - (scaled - value)
    return cast(T, high), cast(T, value - high)


def _product_error(left: Float64Array, right: Float64Array, product: Float64Array) -> Float64Array:
    """Dekker's exact remainder ``left * right - product``."""
    left_high, left_low = _split(left)
    right_high, right_low = _split(right)
    return (
        ((left_high * right_high - product) + left_high * right_low) + left_low * right_high
    ) + left_low * right_low


def _accumulate_corner(
    values: Float64Array, weight: Float64Array, addend: Float64Array
) -> Float64Array:
    """``_fma_array(values, weight, addend)`` for one trilinear corner, bitwise equal.

    ``values`` are float32 voxels widened to float64 and ``weight`` is a product of
    non-negative corner weights, possibly broadcast over the channel axis. A float32
    value has at most 24 significant bits, so Dekker's split returns it unchanged as
    its high word with a +0 low word. The two low-word products of ``_product_error``
    are then zeros, the first of them +0, and adding them only turns a -0 remainder
    into +0, which the ``+ 0.0`` below reproduces. The weight is split once at its own
    shape instead of per channel. Every other operation and its order are those of
    ``_fma_array``.
    """
    product = values * weight
    if not np.any(weight) or not np.any(values):
        return product + addend
    weight_high, weight_low = _split(weight)
    error = values * weight_high
    error -= product
    error += values * weight_low
    error += 0.0
    total = product + addend
    virtual = total - product
    sum_error = product - (total - virtual)
    sum_error += addend - virtual
    sum_error += error
    total += sum_error
    return total


def _source_index(coordinate: Float64Array, size: int) -> Float64Array:
    """``grid_sampler_compute_source_index`` for border padding, ``align_corners=False``."""
    unnormalized = ((coordinate + 1) * size - 1) / 2
    return np.minimum(np.float64(size - 1), np.maximum(unnormalized, 0.0))


def _corners(
    index: Float64Array,
) -> tuple[tuple[Int64Array, Int64Array], tuple[Float64Array, Float64Array]]:
    """Lower/upper voxel indices and their ATen weights ``(upper - i, i - lower)``."""
    low = np.floor(index)
    high = low + 1
    return (low.astype(np.int64), high.astype(np.int64)), (high - index, index - low)


# ATen's corner order (tnw, tne, tsw, tse, bnw, bne, bsw, bse) as (z, y, x) picks.
_CORNERS = (
    (0, 0, 0),
    (0, 0, 1),
    (0, 1, 0),
    (0, 1, 1),
    (1, 0, 0),
    (1, 0, 1),
    (1, 1, 0),
    (1, 1, 1),
)


def _depth_planes(data: Float32Array, index: Int64Array) -> Float32Array:
    """``data.take(index, axis=1)`` without copying a non-contiguous ``data`` in full.

    ``ndarray.take`` first makes its source C-contiguous. Logits cropped from a padded
    window (a volume smaller than the ROI on one axis) are a strided view, so every
    block would copy the whole volume. Gathering from the block's own range of planes
    copies only those planes.
    """
    if data.flags.c_contiguous:
        return data.take(index, axis=1)
    first = int(index.min())
    return data[:, first : int(index.max()) + 1].take(index - first, axis=1)


def _sample_separable(
    data: Float32Array,
    gx: Float64Array,
    gy: Float64Array,
    gz: Float64Array,
) -> Float64Array:
    """Trilinear corners in ATen's order, with its fused multiply-add accumulation.

    Separate product/sum rounding can move a float64 interpolation across a
    float32 midpoint. Preserve the source CPU kernel's fusion before the final
    float32 cast; the network still receives ordinary float32 values.
    """
    depth, height, width = data.shape[1:]
    x_index, x_weight = _corners(_source_index(gx, width))
    y_index, y_weight = _corners(_source_index(gy, height))
    z_index, z_weight = _corners(_source_index(gz, depth))
    along_d, along_h, along_w = (
        (slice(None), None, None),
        (None, slice(None), None),
        (None, None, slice(None)),
    )
    result = np.zeros((data.shape[0], len(gz), len(gy), len(gx)), dtype=np.float64)
    # Each axis has two corner sides: gather the depth planes and then their rows
    # once per side instead of once per corner.
    planes = [_depth_planes(data, np.minimum(z_index[side], depth - 1)) for side in (0, 1)]
    rows = {
        (cz, cy): planes[cz].take(np.minimum(y_index[cy], height - 1), axis=2)
        for cz in (0, 1)
        for cy in (0, 1)
    }
    for cz, cy, cx in _CORNERS:
        # (x * y) * z, left to right, as in grid_sampler_3d_cpu_impl.
        weight = (x_weight[cx][along_w] * y_weight[cy][along_h]) * z_weight[cz][along_d]
        zi, yi, xi = z_index[cz], y_index[cy], x_index[cx]
        values = rows[cz, cy].take(np.minimum(xi, width - 1), axis=3).astype(np.float64)
        inside = ((zi < depth)[along_d] & (yi < height)[along_h]) & (xi < width)[along_w]
        accumulated = _accumulate_corner(values, weight, result)
        result = accumulated if inside.all() else np.where(inside, accumulated, result)
    return result


def _sample(
    data: Float32Array,
    gx: Float64Array,
    gy: Float64Array,
    gz: Float64Array,
) -> Float64Array:
    """General-grid corners with the same fused accumulation as the separable path."""
    depth, height, width = data.shape[1:]
    x_index, x_weight = _corners(_source_index(gx, width))
    y_index, y_weight = _corners(_source_index(gy, height))
    z_index, z_weight = _corners(_source_index(gz, depth))
    corners = [
        (z_index[cz], y_index[cy], x_index[cx], (x_weight[cx] * y_weight[cy]) * z_weight[cz])
        for cz, cy, cx in _CORNERS
    ]
    shape = np.broadcast_shapes(gx.shape, gy.shape, gz.shape)
    result = np.zeros((data.shape[0], *shape), dtype=np.float64)
    for zi, yi, xi, weight in corners:
        inside = (zi < depth) & (yi < height) & (xi < width)
        values = data[
            :,
            np.minimum(zi, depth - 1),
            np.minimum(yi, height - 1),
            np.minimum(xi, width - 1),
        ].astype(np.float64)
        accumulated = _accumulate_corner(values, weight, result)
        result = accumulated if inside.all() else np.where(inside, accumulated, result)
    return result
