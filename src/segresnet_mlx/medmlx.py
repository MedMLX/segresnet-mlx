"""medmlx declaration and runner for the MONAI SegResNet renal CECT bundle."""

from __future__ import annotations

import os
import tempfile
import time
from collections.abc import Mapping
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

from segresnet_mlx._arrays import Float32Array, Float64Array, MaskArray
from segresnet_mlx._records import RunResult, Runtime

if TYPE_CHECKING:
    from medmlx_core.volume_workspace import VolumeWorkspace

    from segresnet_mlx.network import SegResNet

import numpy as np
from medmlx_core.errors import (
    AssetNotReadyError,
    InvalidInputError,
    ModelExecutionError,
)
from medmlx_core.weights import weights_dir, weights_revision

from segresnet_mlx import transforms
from segresnet_mlx.topology import ArithmeticProfile, SegResNetSpec


@dataclass(frozen=True, slots=True)
class Bundle:
    """One MONAI bundle: network topology, inferer settings and file contract.

    ``roi_size``, ``sw_batch_size``, ``overlap``, ``mode`` and ``padding_mode`` are the
    sliding-window settings described in ``SlidingWindowConfig``. ``roles`` names the
    ``run`` path arguments the bundle needs, in network channel order. ``labels`` maps
    each value written to the mask to its name; ``output_dtype`` is the NIfTI dtype.
    """

    spec: SegResNetSpec
    roi_size: tuple[int, int, int]
    sw_batch_size: int
    overlap: float
    roles: tuple[str, ...]
    labels: Mapping[int, str]
    output_dtype: str
    mode: str = "constant"
    padding_mode: str = "constant"


# The renal bundle uses the MONAI SegResNet defaults for blocks_down (1, 2, 2, 4),
# blocks_up (1, 1, 1), group norm with 8 groups and eps 1e-5, with deconv upsampling.
BUNDLES: dict[str, Bundle] = {
    "renalStructures_CECT_segmentation": Bundle(
        spec=SegResNetSpec(3, 6, 32, upsample_mode="deconv"),
        roi_size=(96, 96, 96),
        sw_batch_size=4,
        overlap=0.25,
        roles=("arterial_path", "venous_path", "excretory_path"),
        labels={
            0: "background",
            1: "artery",
            2: "vein",
            3: "ureter",
            4: "neoplasm",
            5: "parenchyma",
        },
        output_dtype="float32",
    ),
}

RENAL_CECT_MODEL = {
    "id": "segresnet-renal-cect",
    "label": "SegResNet renal CECT",
    "task": "segmentation",
    "run": "segresnet_mlx.medmlx:renal_cect",
    # Measured MLX peak for a 478x277x161 three-phase CT; larger inputs need more.
    "peak_memory_gb": 4.1,
    "summary": "Renal structures from arterial, venous and excretory phase CT.",
    "labels": {
        str(value): name
        for value, name in BUNDLES["renalStructures_CECT_segmentation"].labels.items()
    },
}

# The renal bundle's Spacingd pixdim, in mm.
_RENAL_SPACING = (0.8, 0.8, 0.8)


@dataclass(frozen=True, slots=True)
class Volume:
    """A finite 3D NIfTI input and its float32 voxels on the reader-selected grid.

    ``data`` has no channel axis and keeps the file's voxel axis order; ``affine`` is
    the voxel-to-world matrix chosen by the nibabel reader after MONAI's header
    correction (see ``transforms.image_arrays``), and ``shape`` is the stored shape.
    """

    path: Path
    shape: tuple[int, int, int]
    affine: Float64Array
    data: Float32Array


@dataclass(frozen=True, slots=True)
class SpatialInverse:
    """The reference key's applied transforms, which the bundle's ``Invertd`` reverses.

    ``affine`` is the voxel-to-world matrix of the preprocessed grid that the network
    sees (after reorientation and resampling).
    """

    orientation: transforms.Orientation
    resampling: transforms.Resampling
    affine: Float64Array


def run(
    *,
    bundle: str,
    weights_path: str,
    output_dir: str,
    arterial_path: str | None = None,
    venous_path: str | None = None,
    excretory_path: str | None = None,
    overwrite: bool = False,
    arithmetic_profile: ArithmeticProfile = "torch-order-fp32",
    streaming: bool = False,
    scratch_dir: str | None = None,
) -> RunResult:
    """Segment one case with a MONAI SegResNet bundle and write the label map.

    ``bundle`` is ``renalStructures_CECT_segmentation``, which takes
    ``arterial_path``, ``venous_path`` and ``excretory_path``: co-registered CT
    phases on one grid. ``weights_path`` is the native
    ``segresnet-renal-cect.safetensors`` file from ``MedMLX/segresnet-renal-cect-mlx``.

    Inputs must be finite 3D NIfTI volumes with identical shape and affine
    (atol 1e-5). Writes ``<bundle>_mask.nii.gz`` in ``output_dir`` on the first
    input's shape and reader-selected affine (see ``Volume``): float32 labels 0-5,
    as the bundle's ``SaveImaged`` wrote them. An existing mask is replaced only when
    ``overwrite`` is true; output aliases of input images are refused by a
    preflight check. Every argument is checked before weights or inputs are loaded.
    The NIfTI spatial-unit field must be mm or unspecified (then assumed mm); other
    units are refused, not converted. Paths are not locked against concurrent
    replacement after the alias check.

    ``arithmetic_profile="compensated-fp32"`` uses Metal float pairs for
    convolution and GroupNorm accumulation, with float32 layer boundaries.
    Its source comparison uses float64 internal accumulators and float32
    outputs. The default retains the existing Torch-order FP32 arithmetic.
    ``compensated-fp48`` carries both words through the hidden layers and uses
    float64 host stitching before one float32 logits export. It compares with the
    unchanged original graph evaluated in float64 throughout its network and inferer.

    Args:
        bundle: ``renalStructures_CECT_segmentation``.
        weights_path: Path to ``segresnet-renal-cect.safetensors``. Hugging Face
            cache file paths are accepted; directories and Torch ``.pt`` checkpoints
            are not.
        output_dir: Directory for ``<bundle>_mask.nii.gz``; created if missing.
        arterial_path: Arterial-phase 3D CT NIfTI; required.
        venous_path: Venous-phase 3D CT NIfTI on the arterial grid; required.
        excretory_path: Excretory-phase 3D CT NIfTI on the arterial grid; required.
        overwrite: Replace an existing output mask when true; defaults to false.
        arithmetic_profile: ``torch-order-fp32`` (default), ``compensated-fp32``,
            or ``compensated-fp48``.
        streaming: Use bounded blending intermediates and temporary mapped logits.
            The final label volume remains resident; defaults to False.
        scratch_dir: Parent directory for temporary maps when streaming is enabled;
            None uses the system temporary directory. Files are cleaned up on exit.

    Returns:
        ``{"outputs": {"mask": path}, "details": {...}}``; ``details`` is described
        by ``segresnet_mlx._records.Details`` and always has
        ``requires_human_review`` set to true.

    Raises:
        InvalidInputError: Unknown bundle or profile, missing or absent input paths,
            ``overwrite`` or ``streaming`` not a bool, ``output_dir`` not a directory,
            an existing output without ``overwrite``, or an unreadable, non-3D,
            non-finite, non-mm or mismatched-grid NIfTI input.
        AssetNotReadyError: The weights file is missing, not ``.safetensors``, or
            does not match the bundle's network.
        ModelExecutionError: Inference failed or produced a mask off the source grid
            or with labels outside the bundle's set.
    """
    values = {
        "arterial_path": arterial_path,
        "venous_path": venous_path,
        "excretory_path": excretory_path,
    }
    config, mask_path = _check_arguments(
        bundle,
        values,
        output_dir,
        overwrite=overwrite,
        arithmetic_profile=arithmetic_profile,
        streaming=streaming,
        scratch_dir=scratch_dir,
    )
    output_root = mask_path.parent
    weights = _weights_file(weights_path)
    started = time.perf_counter()
    volumes = load_inputs([(role, values[role]) for role in config.roles])
    network = _load_network(weights, config.spec, arithmetic_profile=arithmetic_profile)
    loaded = time.perf_counter()
    image, inverse = preprocess_renal(volumes)
    prepared = time.perf_counter()
    if streaming:
        from medmlx_core.volume_workspace import VolumeWorkspace

        context = VolumeWorkspace(Path(scratch_dir).expanduser() if scratch_dir else None)
    else:
        context = nullcontext(None)
    with context as workspace:
        logits, runtime = (
            predict(network, image, config)
            if workspace is None
            else predict(network, image, config, workspace=workspace)
        )
        predicted = time.perf_counter()
        mask = (
            postprocess_renal(logits[0], inverse)
            if workspace is None
            else postprocess_streaming(logits[0], inverse)
        )
    _validate_mask(mask, volumes[0], config)
    output_root.mkdir(parents=True, exist_ok=True)
    write_atomic(mask_path, mask, volumes[0].affine, config.output_dtype)
    finished = time.perf_counter()
    present = sorted(int(value) for value in np.unique(mask))
    return {
        "outputs": {"mask": mask_path.as_posix()},
        "details": {
            "weights_revision": weights_revision(weights.parent),
            "bundle": bundle,
            "arithmetic_profile": arithmetic_profile,
            "streaming": streaming,
            "labels": {str(value): name for value, name in config.labels.items()},
            "labels_present": present,
            "input_shape": list(volumes[0].shape),
            "network_input_shape": list(image.shape[2:]),
            "output_dtype": config.output_dtype,
            "sliding_window": {
                "roi_size": list(config.roi_size),
                "sw_batch_size": config.sw_batch_size,
                "overlap": config.overlap,
                "mode": config.mode,
                "padding_mode": config.padding_mode,
            },
            "mlx_device": runtime["device"],
            "peak_mlx_bytes": runtime["peak_bytes"],
            "seconds": {
                "load_inputs_and_weights": round(loaded - started, 4),
                "preprocess": round(prepared - loaded, 4),
                "inference": round(predicted - prepared, 4),
                "postprocess_and_write": round(finished - predicted, 4),
            },
            "requires_human_review": True,
        },
    }


def renal_cect(
    *,
    arterial_path: str,
    venous_path: str,
    excretory_path: str,
    output_dir: str,
    weights_path: str | None = None,
    overwrite: bool = False,
    arithmetic_profile: ArithmeticProfile = "torch-order-fp32",
    streaming: bool = False,
    scratch_dir: str | None = None,
) -> RunResult:
    """Segment renal CECT; omit weights_path to download its cached model folder.

    The arterial, venous and excretory CT volumes must share a voxel grid. An
    explicit folder contains ``segresnet-renal-cect.safetensors`` as published for
    this bundle; an explicit safetensors file is also accepted. Automatic downloads
    use ``MedMLX/segresnet-renal-cect-mlx``. Arithmetic and preprocessing are
    delegated unchanged to ``run``.

    Args:
        arterial_path: Arterial-phase CT NIfTI on the shared input grid.
        venous_path: Venous-phase CT NIfTI on the arterial grid.
        excretory_path: Excretory-phase CT NIfTI on the arterial grid.
        output_dir: Directory for the renal label map.
        weights_path: Optional folder containing ``segresnet-renal-cect.safetensors``;
            omit to download and reuse the renal bundle. Explicit files remain accepted.
        overwrite: Replace an existing output mask when true.
        arithmetic_profile: ``torch-order-fp32`` (default), ``compensated-fp32``
            or ``compensated-fp48``; see ``run`` for their arithmetic contracts.
        streaming: Use bounded blending and mapped logits, as in ``run``.
        scratch_dir: Existing parent directory for temporary streaming maps.
    """
    _check_arguments(
        "renalStructures_CECT_segmentation",
        {
            "arterial_path": arterial_path,
            "venous_path": venous_path,
            "excretory_path": excretory_path,
        },
        output_dir,
        overwrite=overwrite,
        arithmetic_profile=arithmetic_profile,
        streaming=streaming,
        scratch_dir=scratch_dir,
    )
    folder = (
        weights_dir("segresnet-renal-cect")
        if weights_path is None
        else Path(weights_path).expanduser()
    )
    checkpoint = folder / "segresnet-renal-cect.safetensors" if folder.is_dir() else folder
    return run(
        bundle="renalStructures_CECT_segmentation",
        weights_path=str(checkpoint),
        output_dir=output_dir,
        arterial_path=arterial_path,
        venous_path=venous_path,
        excretory_path=excretory_path,
        overwrite=overwrite,
        arithmetic_profile=arithmetic_profile,
        streaming=streaming,
        scratch_dir=scratch_dir,
    )


def _check_arguments(
    bundle: str,
    values: Mapping[str, str | None],
    output_dir: str,
    *,
    overwrite: bool,
    arithmetic_profile: ArithmeticProfile,
    streaming: bool = False,
    scratch_dir: str | None = None,
) -> tuple[Bundle, Path]:
    """Check every ``run`` argument before weights, inputs or MLX are loaded.

    Returns the bundle settings and the absolute mask path.
    """
    if bundle not in BUNDLES:
        raise InvalidInputError(
            f"Unknown SegResNet bundle {bundle!r}; expected one of: {', '.join(BUNDLES)}"
        )
    if arithmetic_profile not in {"torch-order-fp32", "compensated-fp32", "compensated-fp48"}:
        raise InvalidInputError(f"Unknown SegResNet arithmetic profile {arithmetic_profile!r}")
    config = BUNDLES[bundle]
    missing = [role for role in config.roles if values.get(role) is None]
    if missing:
        raise InvalidInputError(
            f"{bundle} requires {', '.join(config.roles)}; missing {', '.join(missing)}"
        )
    sources = [Path(str(values[role])).expanduser() for role in config.roles]
    for role, source in zip(config.roles, sources, strict=True):
        if not source.is_file():
            raise InvalidInputError(f"{role} not found: {source}")
    for name, flag in (("overwrite", overwrite), ("streaming", streaming)):
        if not isinstance(cast(object, flag), bool):
            raise InvalidInputError(f"{name} must be true or false")
    if streaming and scratch_dir is not None and not Path(scratch_dir).expanduser().is_dir():
        raise InvalidInputError("scratch_dir must name an existing directory")
    output_root = Path(output_dir).expanduser().resolve()
    if output_root.exists() and not output_root.is_dir():
        raise InvalidInputError(f"output_dir is not a directory: {output_root}")
    mask_path = output_root / f"{bundle}_mask.nii.gz"
    for source in sources:
        if mask_path.resolve() == source.resolve() or (
            mask_path.exists() and mask_path.samefile(source)
        ):
            raise InvalidInputError("SegResNet output must differ from every input image")
    if mask_path.exists() and not overwrite:
        raise InvalidInputError(f"Output exists: {mask_path}; pass overwrite=true to replace it")
    return config, mask_path


def preprocess_renal(volumes: list[Volume]) -> tuple[Float32Array, SpatialInverse]:
    """The renal bundle's ``preprocessing`` from ``EnsureChannelFirstd`` onward.

    Each phase is reoriented to LPS and resampled to 0.8 mm. A later phase reuses
    the first phase's output shape when its oriented grid matches (as ``Spacingd``
    does across keys). The phases are concatenated as channels, then intensities
    are clipped and scaled from [-1000, 1000] HU to [0, 1]. Returns the
    ``(1, 3, *spatial)`` float32 network input, with spatial axes in (L, P, S)
    order, and the arterial key's inverse record.
    """
    channels: list[Float32Array] = []
    record: SpatialInverse | None = None
    first_shape: tuple[int, ...] | None = None
    first_pixdim: Float64Array | None = None
    output_shape: tuple[int, ...] | None = None
    for volume in volumes:
        oriented, affine, orientation = transforms.orient(volume.data[None], volume.affine, "LPS")
        pixdim = transforms.affine_to_spacing(affine, 3)
        if first_shape is None or first_pixdim is None:
            first_shape, first_pixdim = oriented.shape[1:], pixdim
            should_match = False
        else:
            should_match = np.allclose(first_shape, oriented.shape[1:]) and np.allclose(
                first_pixdim, pixdim, atol=1e-3
            )
        resampled, affine, resampling = transforms.spacing(
            oriented, affine, _RENAL_SPACING, output_shape if should_match else None
        )
        if output_shape is None:
            output_shape = resampled.shape[1:]
        if record is None:
            record = SpatialInverse(orientation, resampling, affine)
        channels.append(resampled)
    assert record is not None
    image = transforms.scale_intensity_range(
        transforms.concatenate(channels), -1000, 1000, 0.0, 1.0
    )
    return image[None], record


def postprocess_renal(logits: Float32Array, inverse: SpatialInverse) -> Float32Array:
    """``Invertd`` (0.8 mm grid back to the source grid), sigmoid, 0.5, ``MergeClassesd``.

    ``logits`` is ``(6, *spatial)`` on the resampled grid. Returns the float32 label
    map (values 0-5) on the source grid. The inversion is applied to the logits
    before the sigmoid, in the bundle's order.
    """
    values, affine = transforms.invert_spacing(logits, inverse.affine, inverse.resampling)
    values, _ = transforms.invert_orientation(values, affine, inverse.orientation)
    binary = transforms.sigmoid_threshold_mask(values).astype(np.float32)
    return transforms.merge_classes(binary)[0]


def postprocess_streaming(logits: Float32Array, inverse: SpatialInverse) -> Float32Array:
    """``postprocess_renal`` one class channel at a time, bounding class storage.

    Each foreground channel is inverted, thresholded and merged in turn; a voxel keeps
    the first channel that claims it, as ``MergeClassesd`` does.
    """
    labels: Float32Array | None = None
    for channel in range(1, logits.shape[0]):
        values, affine = transforms.invert_spacing(
            logits[channel : channel + 1], inverse.affine, inverse.resampling
        )
        values, _ = transforms.invert_orientation(values, affine, inverse.orientation)
        active = transforms.sigmoid_threshold_mask(values)[0]
        if labels is None:
            labels = np.zeros(active.shape, dtype=np.float32)
        labels[(labels == 0) & active] = channel
    assert labels is not None
    return labels


def _weights_file(weights_path: str) -> Path:
    """Validate a safetensors file path while retaining any Hub-cache symlink."""
    # MLX identifies the file format from the supplied name. Hub cache paths are
    # named .safetensors symlinks whose underlying content-addressed blobs have no suffix.
    path = Path(weights_path).expanduser().absolute()
    hint = "Download the matching safetensors file from the MedMLX weight repositories."
    if not path.is_file():
        raise AssetNotReadyError(
            f"SegResNet weights not found: {path}", reason=f"missing {path}", hint=hint
        )
    if path.suffix != ".safetensors":
        raise AssetNotReadyError(
            f"SegResNet weights must be converted MLX safetensors, got {path.name}",
            reason=f"unconverted {path.name}",
            hint=hint,
        )
    return path


def load_inputs(roles: list[tuple[str, str | None]]) -> list[Volume]:
    """Load each ``(role, path)`` and require one shared voxel grid.

    Every volume must match the first one's shape and affine (``atol=1e-5``).
    This check does not align mismatched inputs; bundle preprocessing may still
    resample the admitted grid afterward.
    """
    volumes = [load_volume(Path(str(path)).expanduser().resolve(), role) for role, path in roles]
    reference = volumes[0]
    for (role, _), volume in zip(roles[1:], volumes[1:], strict=True):
        if volume.shape != reference.shape or not np.allclose(
            volume.affine, reference.affine, atol=1e-5, rtol=0
        ):
            raise InvalidInputError(
                f"{role} must share {roles[0][0]}'s voxel grid (shape and affine); "
                "resample it explicitly first"
            )
    return volumes


def load_volume(path: Path, role: str) -> Volume:
    """Read 3D voxels using the affine selected by the upstream nibabel reader.

    Voxels must be real, finite and representable as float32. Spatial units must be
    mm or unspecified (assumed mm), and the corrected affine must be invertible.
    """
    from segresnet_mlx._nifti import ImageFileError, nib

    if not path.is_file():
        raise InvalidInputError(f"{role} not found: {path}")
    try:
        image = nib.load(path.as_posix())
    except (OSError, ImageFileError) as error:
        raise InvalidInputError(f"Invalid {role} NIfTI: {path}") from error
    try:
        spatial_units, _ = image.header.get_xyzt_units()
    except (AttributeError, KeyError, ValueError) as error:
        raise InvalidInputError(f"{role} has invalid NIfTI spatial units") from error
    if spatial_units not in ("mm", "unknown"):
        raise InvalidInputError(f"{role} spatial units must be mm or unspecified")
    raw = np.asanyarray(image.dataobj)
    if raw.ndim != 3:
        raise InvalidInputError(f"{role} must be a 3D NIfTI; got shape {tuple(raw.shape)}")
    if np.iscomplexobj(raw):
        raise InvalidInputError(f"{role} must contain real-valued voxels")
    if not np.all(np.isfinite(raw)):
        raise InvalidInputError(f"{role} contains non-finite voxels")
    limit = np.finfo(np.float32).max
    if np.any(raw > limit) or np.any(raw < -limit):
        raise InvalidInputError(f"{role} contains voxels outside the finite float32 range")
    affine = np.asarray(image.affine, dtype=np.float64)
    if affine.shape != (4, 4) or not np.all(np.isfinite(affine)):
        raise InvalidInputError(f"{role} affine must be a finite 4x4 matrix")
    data, reader_affine = transforms.image_arrays(image)
    if not np.all(np.isfinite(reader_affine)) or np.linalg.matrix_rank(reader_affine[:3, :3]) < 3:
        raise InvalidInputError(f"{role} affine must define an invertible 3D grid")
    shape = (int(raw.shape[0]), int(raw.shape[1]), int(raw.shape[2]))
    return Volume(path=path, shape=shape, affine=reader_affine, data=data)


def _load_network(
    weights: Path,
    spec: SegResNetSpec,
    *,
    arithmetic_profile: ArithmeticProfile = "torch-order-fp32",
) -> SegResNet:
    """Strictly load converted weights for the bundle topology."""
    from segresnet_mlx.weights import load_model

    try:
        return load_model(weights, spec=spec, arithmetic_profile=arithmetic_profile)
    except (InvalidInputError, ValueError, RuntimeError) as error:
        raise AssetNotReadyError(
            f"SegResNet weights {weights} do not match {spec}: {error}",
            reason=f"mismatched {weights.name}",
            hint="Download the matching safetensors file from the MedMLX weight repositories.",
        ) from error


def predict(
    network: SegResNet,
    image: Float32Array,
    config: Bundle,
    *,
    workspace: VolumeWorkspace | None = None,
) -> tuple[Float32Array, Runtime]:
    """Run the bundle's ``SlidingWindowInferer`` settings around the MLX network.

    ``image`` is the preprocessed ``(1, C, *spatial)`` array. Returns float32 logits
    ``(1, out_channels, *spatial)`` and the MLX device and peak memory in bytes.
    Raises ``ModelExecutionError`` if inference fails.
    """
    from segresnet_mlx._runtime import import_mlx
    from segresnet_mlx.predictor import BundlePredictor, SlidingWindowConfig, predict_wide_logits

    mx = import_mlx()
    window = SlidingWindowConfig(
        config.roi_size, config.sw_batch_size, config.overlap, config.mode, config.padding_mode
    )
    try:
        host = np.ascontiguousarray(image, dtype=np.float32)
        logits = (
            predict_wide_logits(network, host, window, mx=mx, workspace=workspace)
            if network.arithmetic_profile == "compensated-fp48"
            else BundlePredictor(network, mx=mx).predict_logits(host, window, workspace=workspace)
        )
    except (RuntimeError, ValueError) as error:
        raise ModelExecutionError(f"SegResNet MLX inference failed: {error}") from error
    return logits, {"device": str(mx.default_device()), "peak_bytes": mx.get_peak_memory()}


def _validate_mask(mask: MaskArray, reference: Volume, config: Bundle) -> None:
    """Require the source grid and only the bundle's labels."""
    if tuple(mask.shape) != reference.shape:
        raise ModelExecutionError(
            f"SegResNet mask shape {tuple(mask.shape)} is not the input shape {reference.shape}"
        )
    if not set(np.unique(mask).tolist()) <= set(config.labels):
        raise ModelExecutionError("SegResNet exported labels outside the bundle contract")


def write_atomic(path: Path, mask: MaskArray, affine: Float64Array, dtype: str) -> None:
    """Write the mask NIfTI via a temporary file in ``path.parent``, then rename it.

    A failure therefore never leaves a partial file at ``path``.
    """
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".nii.gz", dir=path.parent)
    os.close(descriptor)
    try:
        transforms.write_nifti(Path(name), mask, affine, dtype)
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


__all__ = ["BUNDLES", "RENAL_CECT_MODEL", "renal_cect", "run"]
