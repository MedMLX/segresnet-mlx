"""NumPy ports of the MONAI 1.6 transforms used by the SegResNet bundles.

These functions reproduce the renal CECT, BraTS MRI and whole-body CT bundles'
``preprocessing`` and ``postprocessing`` without Torch or MONAI: ``LoadImage``
(nibabel reader), ``Orientation``, ``Spacing`` and its inverse, ``ScaleIntensityRange``,
``ScaleIntensity``, ``NormalizeIntensity``, ``Activations``, ``AsDiscrete``, the renal
bundle's ``MergeClassesd`` and the NIfTI writer. Spatial resampling follows MONAI's
``AffineTransform`` path, that is ``torch.nn.functional.affine_grid`` and trilinear
``grid_sample`` with border padding and ``align_corners=False``, computed in float64
as ``Spacing`` does and returned as float32.

Image arrays are channel-first float32 ``(C, *spatial)`` and affines are 4x4 float64
voxel-to-world matrices as stored in the NIfTI header. Bundle ``pixdim`` targets
are in mm; the runner validates spatial units, and these functions assume mm.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import numpy.typing as npt
from medmlx_core.geometry import (
    _bmm as _bmm,
)
from medmlx_core.geometry import (
    _fma as _fma,
)
from medmlx_core.geometry import (
    _linspace_from_neg_one as _linspace_from_neg_one,
)
from medmlx_core.geometry import (
    _mm as _mm,
)
from medmlx_core.geometry import (
    affine_to_spacing as affine_to_spacing,
)
from medmlx_core.geometry import (
    compute_shape_offset as compute_shape_offset,
)
from medmlx_core.geometry import (
    to_affine_nd as to_affine_nd,
)
from medmlx_core.geometry import (
    zoom_affine as zoom_affine,
)

from segresnet_mlx._arrays import HostArray, Int64Array
from segresnet_mlx._nifti import NiftiImage, nib

type Float32Array = npt.NDArray[np.float32]
type Float64Array = npt.NDArray[np.float64]

# monai.data.utils.AFFINE_TOL
AFFINE_TOL = 1e-3
_LABELS = (("L", "R"), ("P", "A"), ("I", "S"))
# Bound the float64 corner arrays of one resampling chunk.
_CHUNK_VOXELS = 1 << 20


@dataclass(frozen=True, slots=True)
class Orientation:
    """What ``Orientation`` records for its inverse: the affine it started from."""

    original_affine: Float64Array


@dataclass(frozen=True, slots=True)
class Resampling:
    """What ``SpatialResample`` records for its inverse."""

    src_affine: Float64Array
    orig_size: tuple[int, ...]


def image_arrays(image: NiftiImage) -> tuple[Float32Array, Float64Array]:
    """MONAI's nibabel reader: correct the 3D header, then export float32 voxels."""
    correct_nifti_header(image)
    data = np.asanyarray(image.dataobj, order="C").astype(np.float32, copy=False)
    return data, np.array(image.affine, dtype=np.float64, copy=True)


def correct_nifti_header(image: NiftiImage) -> None:
    """Apply MONAI's spacing/form correction to a loaded 3D NIfTI, in memory."""
    zooms = np.asarray(image.header.get_zooms())[:3]
    if np.allclose(zooms, affine_to_spacing(image.affine, 3)):
        return
    sform, qform = image.get_sform(), image.get_qform()
    sform_matches = np.allclose(affine_to_spacing(sform, 3), zooms)
    qform_matches = np.allclose(affine_to_spacing(qform, 3), zooms)
    if image.header["sform_code"] != 0:
        if sform_matches:
            return
        if qform_matches:
            image.set_sform(qform)
            return
    if image.header["qform_code"] != 0:
        if qform_matches:
            return
        if sform_matches:
            image.set_qform(sform)
            return
    image.header.set_zooms(affine_to_spacing(image.affine, 3))










def orient(
    data: Float32Array,
    affine: Float64Array,
    axcodes: str,
) -> tuple[Float32Array, Float64Array, Orientation]:
    """``Orientation(axcodes)`` on channel-first data; returns data, new affine, inverse."""
    rank = data.ndim - 1
    affine_np = np.array(affine, dtype=np.float64, copy=True)
    affine_ = to_affine_nd(rank, affine_np)
    source = nib.io_orientation(affine_)
    target = nib.orientations.axcodes2ornt(axcodes[:rank], labels=_LABELS)
    spatial_ornt = nib.orientations.ornt_transform(source, target)
    xform = nib.orientations.inv_ornt_aff(spatial_ornt, data.shape[1:])
    spatial_ornt[:, 0] += 1
    spatial_ornt = np.concatenate([np.array([[0, 1]]), spatial_ornt])
    axes = [axis for axis, flip in enumerate(spatial_ornt[:, 1]) if flip == -1]
    full_transpose = np.arange(data.ndim)
    full_transpose[: len(spatial_ornt)] = np.argsort(spatial_ornt[:, 0])
    result = np.flip(data, tuple(axes)) if axes else data
    if not np.all(full_transpose == np.arange(data.ndim)):
        result = result.transpose(full_transpose)
    new_affine = _mm(affine_np, to_affine_nd(len(affine_np) - 1, xform))
    return np.ascontiguousarray(result), new_affine, Orientation(affine_np)


def invert_orientation(
    data: Float32Array,
    affine: Float64Array,
    record: Orientation,
) -> tuple[Float32Array, Float64Array]:
    """``Orientation.inverse``: reorient to the recorded original axis codes."""
    axcodes = "".join(
        cast(tuple[str, ...], nib.orientations.aff2axcodes(record.original_affine, labels=_LABELS))
    )
    result, new_affine, _ = orient(data, affine, axcodes)
    return result, new_affine


def spacing(
    data: Float32Array,
    affine: Float64Array,
    pixdim: Sequence[float],
    output_spatial_shape: Sequence[int] | None = None,
) -> tuple[Float32Array, Float64Array, Resampling]:
    """``Spacing(pixdim, mode="bilinear")`` with MONAI defaults (border, float64)."""
    rank = data.ndim - 1
    affine_ = to_affine_nd(rank, affine)
    out_d = np.array(tuple(pixdim), dtype=np.float64)[:rank].copy()
    if out_d.size < rank:
        out_d = np.append(out_d, [out_d[-1]] * (rank - out_d.size))
    original = affine_to_spacing(affine_, rank)
    for index, current in enumerate(original):
        target = out_d[index]
        keep = (target - AFFINE_TOL) <= current <= (target + AFFINE_TOL)
        out_d[index] = current if keep else target
    new_affine = zoom_affine(affine_, out_d)
    output_shape, offset = compute_shape_offset(data.shape[1:], affine_, new_affine)
    new_affine[:rank, -1] = offset[:rank]
    shape = list(output_shape) if output_spatial_shape is None else list(output_spatial_shape)
    return spatial_resample(data, affine, new_affine, shape)


def invert_spacing(
    data: Float32Array,
    affine: Float64Array,
    record: Resampling,
    mode: str = "bilinear",
) -> tuple[Float32Array, Float64Array]:
    """``SpatialResample.inverse``: resample back to the recorded affine and size.

    ``mode="nearest"`` matches ``Invertd(nearest_interp=True)``, used to invert a
    discrete label map (e.g. the whole-body bundle's argmax output).
    """
    result, new_affine, _ = spatial_resample(
        data, affine, record.src_affine, record.orig_size, mode
    )
    return result, new_affine


def spatial_resample(
    data: Float32Array,
    src_affine: Float64Array,
    dst_affine: Float64Array,
    spatial_size: Sequence[int],
    mode: str = "bilinear",
) -> tuple[Float32Array, Float64Array, Resampling]:
    """``monai.transforms.spatial.functional.spatial_resample`` (border, float64).

    ``mode`` is ``"bilinear"`` (trilinear in 3D) or ``"nearest"``.
    """
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
    values = _grid_sample(data, xform, in_size, size, mode)
    affine = np.array(src_affine, dtype=np.float64)
    new_affine = _mm(affine, to_affine_nd(len(affine) - 1, xform))
    return values, new_affine, record


def concatenate(channels: Sequence[Float32Array]) -> Float32Array:
    """The renal bundle's ``ConcatImages`` (``np.concatenate`` along channels)."""
    return np.concatenate(list(channels))


def scale_intensity_range(
    data: Float32Array,
    a_min: float,
    a_max: float,
    b_min: float,
    b_max: float,
) -> Float32Array:
    """``ScaleIntensityRange(..., clip=True)`` in float32."""
    result = (data - np.float32(a_min)) / np.float32(a_max - a_min)
    result = result * np.float32(b_max - b_min) + np.float32(b_min)
    return np.clip(result, np.float32(b_min), np.float32(b_max)).astype(np.float32, copy=False)


def normalize_intensity_nonzero(data: Float32Array) -> Float32Array:
    """``NormalizeIntensity(nonzero=True, channel_wise=True)`` on channel-first data.

    Torch takes the float32 sum of the nonzero voxels divided in float32, and the
    population standard deviation around that mean accumulated in float64. The
    sums here use NumPy's float64 order, so the statistics can differ from Torch's
    in the last float32 bit.
    """
    result = np.array(data, dtype=np.float32, copy=True)
    for channel in result:
        selected = channel != 0
        if not selected.any():
            continue
        masked = channel[selected]
        total = np.float32(np.sum(masked, dtype=np.float64))
        mean = np.float32(total / np.float32(masked.size))
        deviation = masked.astype(np.float64) - np.float64(mean)
        std = np.float32(np.sqrt(cast(np.float64, np.sum(deviation * deviation)) / masked.size))
        divisor = np.float32(1.0) if std == 0.0 else std
        channel[selected] = (masked - mean) / divisor
    return result


def scale_intensity_minmax(data: Float32Array, minv: float, maxv: float) -> Float32Array:
    """``ScaleIntensity(minv, maxv)``: ``monai.transforms.utils.rescale_array`` in float32."""
    arr = np.asarray(data, dtype=np.float32)
    mina = arr.min()
    maxa = arr.max()
    if mina == maxa:
        return (arr * np.float32(minv)).astype(np.float32, copy=False)
    norm = (arr - mina) / (maxa - mina)
    return (norm * np.float32(maxv - minv) + np.float32(minv)).astype(np.float32, copy=False)


def softmax_channel(data: Float32Array) -> Float32Array:
    """``Activations(softmax=True)``: ``torch.softmax(img, dim=0)`` in float32."""
    shifted = data - data.max(axis=0, keepdims=True)
    exponentiated = np.exp(shifted)
    return (exponentiated / exponentiated.sum(axis=0, keepdims=True)).astype(
        np.float32, copy=False
    )


def argmax_channel(data: Float32Array) -> Float32Array:
    """``AsDiscrete(argmax=True)``: ``torch.argmax(img, dim=0, keepdim=True)`` as float32."""
    return np.argmax(data, axis=0)[None].astype(np.float32, copy=False)


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


def brats_labels(binary: Float32Array) -> npt.NDArray[np.int64]:
    """BraTS bundle ``Lambdad``: ET (channel 2) -> 4, TC (0) -> 1, WT (1) -> 2.

    ``binary`` is ``(3, *spatial)`` with channels (TC, WT, ET). Returns int64
    ``(1, *spatial)`` with labels 0, 1, 2 and 4.
    """
    return np.where(
        binary[[2]] > 0, 4, np.where(binary[[0]] > 0, 1, np.where(binary[[1]] > 0, 2, 0))
    ).astype(np.int64)


def write_nifti(path: Path, data: HostArray, affine: Float64Array, dtype: npt.DTypeLike) -> None:
    """Cast and write a millimetre NIfTI with sform and qform code 1.

    The sform stores the affine within NIfTI precision. The qform cannot
    represent shear, which NiBabel removes when encoding that form.
    """
    array = np.asarray(data).astype(dtype, copy=False)
    matrix = to_affine_nd(3, affine)
    image = nib.Nifti1Image(array, matrix)
    image.header.set_xyzt_units("mm")
    image.set_sform(matrix, code=1)
    image.set_qform(matrix, code=1)
    nib.save(image, str(path))






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
    mode: str = "bilinear",
) -> Float32Array:
    """``affine_grid`` then ``grid_sample`` (border, ``align_corners=False``).

    MONAI samples a float64 copy of the image and rounds the result to float32; here
    each output chunk is computed in float64 from the gathered float32 voxels, which
    gives the same values without a float64 copy of the whole image.
    The grid's four-term dot products follow the reference Torch build's MKL order,
    ``fma(z, t2, x * t0) + fma(1, t3, y * t1)``; with zeros off the diagonal every
    order gives the same coordinates. ``mode="nearest"`` matches
    ``grid_sample(mode="nearest")``: the single closest voxel, rounded to even.
    """
    if len(in_size) != 3:
        raise ValueError("SegResNet bundle resampling is three-dimensional")
    if mode not in ("bilinear", "nearest"):
        raise ValueError(f"Unsupported grid_sample mode: {mode}")
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
    step = max(1, _CHUNK_VOXELS // max(1, height * width * data.shape[0]))
    separable_fn = _sample_separable if mode == "bilinear" else _sample_separable_nearest
    general_fn = _sample if mode == "bilinear" else _sample_nearest
    for start in range(0, depth, step):
        stop = min(depth, start + step)
        if separable:
            gx = zx[0][0] + ys[0][0]
            gy = zx[1][0, 0] + ys[1]
            gz = zx[2][start:stop, 0] + ys[2][0]
            result[:, start:stop] = separable_fn(data, gx, gy, gz)
        else:
            coordinates = [
                zx[row][start:stop, None, :] + ys[row][None, :, None] for row in range(3)
            ]
            result[:, start:stop] = general_fn(data, *coordinates)
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


def _product_error(
    left: Float64Array, right: Float64Array, product: Float64Array
) -> Float64Array:
    """Dekker's exact remainder ``left * right - product``."""

    def split[T: (Float64Array, np.float64)](value: T) -> tuple[T, T]:
        """Split a float64 value into high and low words for Dekker multiplication."""
        scaled = 134217729.0 * value
        high = scaled - (scaled - value)
        return cast(T, high), cast(T, value - high)

    left_high, left_low = split(left)
    right_high, right_low = split(right)
    return (
        ((left_high * right_high - product) + left_high * right_low) + left_low * right_high
    ) + left_low * right_low


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
    for cz, cy, cx in _CORNERS:
        # (x * y) * z, left to right, as in grid_sampler_3d_cpu_impl.
        weight = (x_weight[cx][along_w] * y_weight[cy][along_h]) * z_weight[cz][along_d]
        zi, yi, xi = z_index[cz], y_index[cy], x_index[cx]
        values = (
            data.take(np.minimum(zi, depth - 1), axis=1)
            .take(np.minimum(yi, height - 1), axis=2)
            .take(np.minimum(xi, width - 1), axis=3)
            .astype(np.float64)
        )
        inside = ((zi < depth)[along_d] & (yi < height)[along_h]) & (xi < width)[along_w]
        result = np.where(inside, _fma_array(values, weight, result), result)
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
        result = np.where(inside, _fma_array(values, weight, result), result)
    return result


def _nearest_index(coordinate: Float64Array, size: int) -> Int64Array:
    """``nearbyint`` of the border-clamped source index: round half to even."""
    return np.rint(_source_index(coordinate, size)).astype(np.int64)


def _sample_separable_nearest(
    data: Float32Array,
    gx: Float64Array,
    gy: Float64Array,
    gz: Float64Array,
) -> Float64Array:
    """Nearest-voxel pick for an axis-aligned grid: one index per axis."""
    depth, height, width = data.shape[1:]
    x_index = _nearest_index(gx, width)
    y_index = _nearest_index(gy, height)
    z_index = _nearest_index(gz, depth)
    return data.take(z_index, axis=1).take(y_index, axis=2).take(x_index, axis=3).astype(
        np.float64
    )


def _sample_nearest(
    data: Float32Array,
    gx: Float64Array,
    gy: Float64Array,
    gz: Float64Array,
) -> Float64Array:
    """Nearest-voxel pick for a general grid: one (z, y, x) index per output site."""
    x_index = _nearest_index(gx, data.shape[3])
    y_index = _nearest_index(gy, data.shape[2])
    z_index = _nearest_index(gz, data.shape[1])
    return data[:, z_index, y_index, x_index].astype(np.float64)
