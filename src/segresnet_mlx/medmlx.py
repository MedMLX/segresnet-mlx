"""medmlx declaration and runner for the original MONAI SegResNet bundles."""

from __future__ import annotations

import os
import tempfile
import time
from collections.abc import Mapping
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

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

MODEL = {
    "id": "segresnet",
    "label": "SegResNet",
    "task": "segmentation",
    "run": "segresnet_mlx.medmlx:run",
    "summary": (
        "MONAI SegResNet bundles: renal structures in three-phase contrast CT "
        "(renalStructures_CECT_segmentation), brain tumor in BraTS MRI "
        "(brats_mri_segmentation), and 104 whole-body structures in CT at 1.5 mm or "
        "3 mm (wholeBody_ct_segmentation, wholeBody_ct_segmentation_lowres)."
    ),
}

BRATS_MODEL = {
    "id": "segresnet-brats",
    "label": "SegResNet BraTS MRI",
    "task": "segmentation",
    "run": "segresnet_mlx.medmlx:brats",
    "summary": "BraTS brain tumor segmentation from co-registered MRI sequences.",
}
RENAL_CECT_MODEL = {
    "id": "segresnet-renal-cect",
    "label": "SegResNet renal CECT",
    "task": "segmentation",
    "run": "segresnet_mlx.medmlx:renal_cect",
    "summary": "Renal structures from arterial, venous and excretory phase CT.",
}

# Label values of the 104 TotalSegmentator structures plus background, as listed in the
# whole-body bundle's metadata.json ``channel_def``.
_WHOLE_BODY_LABELS: Mapping[int, str] = {
    0: "background",
    1: "spleen",
    2: "kidney_right",
    3: "kidney_left",
    4: "gallbladder",
    5: "liver",
    6: "stomach",
    7: "aorta",
    8: "inferior_vena_cava",
    9: "portal_vein_and_splenic_vein",
    10: "pancreas",
    11: "adrenal_gland_right",
    12: "adrenal_gland_left",
    13: "lung_upper_lobe_left",
    14: "lung_lower_lobe_left",
    15: "lung_upper_lobe_right",
    16: "lung_middle_lobe_right",
    17: "lung_lower_lobe_right",
    18: "vertebrae_L5",
    19: "vertebrae_L4",
    20: "vertebrae_L3",
    21: "vertebrae_L2",
    22: "vertebrae_L1",
    23: "vertebrae_T12",
    24: "vertebrae_T11",
    25: "vertebrae_T10",
    26: "vertebrae_T9",
    27: "vertebrae_T8",
    28: "vertebrae_T7",
    29: "vertebrae_T6",
    30: "vertebrae_T5",
    31: "vertebrae_T4",
    32: "vertebrae_T3",
    33: "vertebrae_T2",
    34: "vertebrae_T1",
    35: "vertebrae_C7",
    36: "vertebrae_C6",
    37: "vertebrae_C5",
    38: "vertebrae_C4",
    39: "vertebrae_C3",
    40: "vertebrae_C2",
    41: "vertebrae_C1",
    42: "esophagus",
    43: "trachea",
    44: "heart_myocardium",
    45: "heart_atrium_left",
    46: "heart_ventricle_left",
    47: "heart_atrium_right",
    48: "heart_ventricle_right",
    49: "pulmonary_artery",
    50: "brain",
    51: "iliac_artery_left",
    52: "iliac_artery_right",
    53: "iliac_vena_left",
    54: "iliac_vena_right",
    55: "small_bowel",
    56: "duodenum",
    57: "colon",
    58: "rib_left_1",
    59: "rib_left_2",
    60: "rib_left_3",
    61: "rib_left_4",
    62: "rib_left_5",
    63: "rib_left_6",
    64: "rib_left_7",
    65: "rib_left_8",
    66: "rib_left_9",
    67: "rib_left_10",
    68: "rib_left_11",
    69: "rib_left_12",
    70: "rib_right_1",
    71: "rib_right_2",
    72: "rib_right_3",
    73: "rib_right_4",
    74: "rib_right_5",
    75: "rib_right_6",
    76: "rib_right_7",
    77: "rib_right_8",
    78: "rib_right_9",
    79: "rib_right_10",
    80: "rib_right_11",
    81: "rib_right_12",
    82: "humerus_left",
    83: "humerus_right",
    84: "scapula_left",
    85: "scapula_right",
    86: "clavicula_left",
    87: "clavicula_right",
    88: "femur_left",
    89: "femur_right",
    90: "hip_left",
    91: "hip_right",
    92: "sacrum",
    93: "face",
    94: "gluteus_maximus_left",
    95: "gluteus_maximus_right",
    96: "gluteus_medius_left",
    97: "gluteus_medius_right",
    98: "gluteus_minimus_left",
    99: "gluteus_minimus_right",
    100: "autochthon_left",
    101: "autochthon_right",
    102: "iliopsoas_left",
    103: "iliopsoas_right",
    104: "urinary_bladder",
}


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


# Every bundle uses the MONAI SegResNet defaults for blocks_down (1, 2, 2, 4),
# blocks_up (1, 1, 1), group norm with 8 groups and eps 1e-5. They differ in channel
# counts, init_filters and (renal only) deconv instead of nontrainable upsampling.
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
    "brats_mri_segmentation": Bundle(
        spec=SegResNetSpec(4, 3, 16),
        roi_size=(240, 240, 160),
        sw_batch_size=1,
        overlap=0.5,
        roles=("t1c_path", "t1_path", "t2_path", "flair_path"),
        labels={
            0: "background",
            1: "tumor core, not enhancing",
            2: "whole tumor outside the core",
            4: "enhancing tumor",
        },
        output_dtype="uint8",
    ),
    "wholeBody_ct_segmentation": Bundle(
        spec=SegResNetSpec(1, 105, 32),
        roi_size=(96, 96, 96),
        sw_batch_size=1,
        overlap=0.25,
        roles=("image_path",),
        labels=_WHOLE_BODY_LABELS,
        output_dtype="float32",
        mode="gaussian",
        padding_mode="replicate",
    ),
    "wholeBody_ct_segmentation_lowres": Bundle(
        spec=SegResNetSpec(1, 105, 32),
        roi_size=(96, 96, 96),
        sw_batch_size=1,
        overlap=0.25,
        roles=("image_path",),
        labels=_WHOLE_BODY_LABELS,
        output_dtype="float32",
        mode="gaussian",
        padding_mode="replicate",
    ),
}

_ROLES = tuple(dict.fromkeys(role for bundle in BUNDLES.values() for role in bundle.roles))
# The renal bundle's Spacingd pixdim, in mm.
_RENAL_SPACING = (0.8, 0.8, 0.8)
# wholeBody_ct_segmentation's `displayable_configs#highres` toggle: true selects
# 1.5 mm (models/model.pt), false selects 3 mm (models/model_lowres.pt). Both share
# one network and inferer; only the resampling target and checkpoint differ.
_WHOLE_BODY_PIXDIM = {
    "wholeBody_ct_segmentation": (1.5, 1.5, 1.5),
    "wholeBody_ct_segmentation_lowres": (3.0, 3.0, 3.0),
}


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
    t1c_path: str | None = None,
    t1_path: str | None = None,
    t2_path: str | None = None,
    flair_path: str | None = None,
    image_path: str | None = None,
    overwrite: bool = False,
    arithmetic_profile: ArithmeticProfile = "torch-order-fp32",
    streaming: bool = False,
    scratch_dir: str | None = None,
) -> RunResult:
    """Segment one case with a MONAI SegResNet bundle and write the label map.

    ``bundle`` is ``renalStructures_CECT_segmentation`` (``arterial_path``,
    ``venous_path``, ``excretory_path``: co-registered CT phases on one grid),
    ``brats_mri_segmentation`` (``t1c_path``, ``t1_path``, ``t2_path``,
    ``flair_path``: co-registered MRI on one grid), or ``wholeBody_ct_segmentation``
    / ``wholeBody_ct_segmentation_lowres`` (``image_path``: one CT volume, segmented
    at 1.5 mm or 3 mm respectively). ``weights_path`` is a native ``.safetensors``
    file: ``segresnet-brats.safetensors`` from ``MedMLX/segresnet-brats-mlx`` for
    BraTS, or ``segresnet-renal-cect.safetensors`` from
    ``MedMLX/segresnet-renal-cect-mlx`` for renal CECT, or the corresponding
    high- or low-resolution file from ``MedMLX/segresnet-wholebody-mlx``.

    Inputs must be finite 3D NIfTI volumes; multi-input bundles need identical shape
    and affine (atol 1e-5). Writes ``<bundle>_mask.nii.gz`` in ``output_dir`` on the
    first input's shape and reader-selected affine (see ``Volume``): renal labels 0-5
    and whole-body labels 0-104 as float32, BraTS labels 0, 1, 2, 4 as uint8, as the
    bundles' ``SaveImaged`` wrote them. An existing mask is replaced only when
    ``overwrite`` is true; output aliases of input images are refused by a
    preflight check. Spatial affine values are assumed to be in millimeters; the
    NIfTI spatial-unit field is not checked or converted. Paths are not locked
    against concurrent replacement after the alias check.

    ``arithmetic_profile="compensated-fp32"`` uses Metal float pairs for
    convolution and GroupNorm accumulation, with float32 layer boundaries.
    Its source comparison uses float64 internal accumulators and float32
    outputs. The default retains the existing Torch-order FP32 arithmetic.
    ``compensated-fp48`` carries both words through the hidden layers and uses
    float64 host stitching before one float32 logits export. This profile
    requires transpose-convolution upsampling and compares with the unchanged
    original graph evaluated in float64 throughout its network and inferer.

    Args:
        bundle: ``brats_mri_segmentation``, ``renalStructures_CECT_segmentation``,
            ``wholeBody_ct_segmentation`` or ``wholeBody_ct_segmentation_lowres``.
        weights_path: Path to the matching native ``.safetensors`` file, such as
            ``segresnet-brats.safetensors`` or ``segresnet-renal-cect.safetensors``.
            Hugging Face cache file paths are accepted; directories and Torch ``.pt``
            checkpoints are not.
        output_dir: Directory for ``<bundle>_mask.nii.gz``; created if missing.
        arterial_path: Renal arterial-phase 3D NIfTI; required for the renal bundle.
        venous_path: Renal venous-phase 3D NIfTI on the arterial grid; required for renal.
        excretory_path: Renal excretory-phase 3D NIfTI on the arterial grid; required
            for renal.
        t1c_path: BraTS contrast-enhanced T1 3D NIfTI; required for BraTS.
        t1_path: BraTS T1 3D NIfTI on the T1c grid; required for BraTS.
        t2_path: BraTS T2 3D NIfTI on the T1c grid; required for BraTS.
        flair_path: BraTS FLAIR 3D NIfTI on the T1c grid; required for BraTS.
        image_path: Single 3D CT NIfTI; required for either whole-body bundle.
        overwrite: Replace an existing output mask when true; defaults to false.
        arithmetic_profile: ``torch-order-fp32`` (default), ``compensated-fp32``,
            or ``compensated-fp48``; the last supports only the renal bundle.
        streaming: Use bounded blending intermediates and temporary mapped logits.
            The final label volume remains resident; defaults to False.
        scratch_dir: Parent directory for temporary maps when streaming is enabled;
            None uses the system temporary directory. Files are cleaned up on exit.

    Returns:
        ``{"outputs": {"mask": path}, "details": {...}}``; ``details`` is described
        by ``segresnet_mlx._records.Details`` and always has
        ``requires_human_review`` set to true.

    Raises:
        InvalidInputError: Unknown bundle or profile, missing or inapplicable input
            paths, an existing output without ``overwrite``, or an unreadable,
            non-3D, non-finite or mismatched-grid NIfTI input.
        AssetNotReadyError: The weights file is missing, not ``.safetensors``, or
            does not match the bundle's network.
        ModelExecutionError: Inference failed or produced a mask off the source grid
            or with labels outside the bundle's set.
    """
    if type(streaming) is not bool:
        raise InvalidInputError("streaming must be boolean")
    values = {
        "arterial_path": arterial_path,
        "venous_path": venous_path,
        "excretory_path": excretory_path,
        "t1c_path": t1c_path,
        "t1_path": t1_path,
        "t2_path": t2_path,
        "flair_path": flair_path,
        "image_path": image_path,
    }
    if bundle not in BUNDLES:
        raise InvalidInputError(
            f"Unknown SegResNet bundle {bundle!r}; expected one of: {', '.join(BUNDLES)}"
        )
    if arithmetic_profile not in {"torch-order-fp32", "compensated-fp32", "compensated-fp48"}:
        raise InvalidInputError(f"Unknown SegResNet arithmetic profile {arithmetic_profile!r}")
    config = BUNDLES[bundle]
    if arithmetic_profile == "compensated-fp48" and config.spec.upsample_mode != "deconv":
        raise InvalidInputError("Paired precision requires transpose-convolution upsampling")
    missing = [role for role in config.roles if values[role] is None]
    extra = [role for role in _ROLES if role not in config.roles and values[role] is not None]
    if missing or extra:
        raise InvalidInputError(
            f"{bundle} requires {', '.join(config.roles)}"
            + (f"; missing {', '.join(missing)}" if missing else "")
            + (f"; not applicable: {', '.join(extra)}" if extra else "")
        )
    output_root = Path(output_dir).expanduser().resolve()
    mask_path = output_root / f"{bundle}_mask.nii.gz"
    for role in config.roles:
        value = values[role]
        assert value is not None
        source = Path(value).expanduser().resolve()
        if mask_path.resolve() == source or (
            mask_path.exists() and source.exists() and mask_path.samefile(source)
        ):
            raise InvalidInputError("SegResNet output must differ from every input image")
    if mask_path.exists() and not overwrite:
        raise InvalidInputError(f"Output exists: {mask_path}; pass overwrite=true to replace it")
    weights = _weights_file(weights_path)
    started = time.perf_counter()
    volumes = load_inputs([(role, values[role]) for role in config.roles])
    network = _load_network(weights, config.spec, arithmetic_profile=arithmetic_profile)
    loaded = time.perf_counter()
    if bundle == "renalStructures_CECT_segmentation":
        image, inverse = preprocess_renal(volumes)
    elif bundle in _WHOLE_BODY_PIXDIM:
        image, inverse = preprocess_wholebody(volumes, _WHOLE_BODY_PIXDIM[bundle])
    else:
        image, inverse = preprocess_brats(volumes), None
    prepared = time.perf_counter()
    if streaming:
        from medmlx_core.volume_workspace import VolumeWorkspace

        context = VolumeWorkspace(scratch_dir)
    else:
        context = nullcontext(None)
    with context as workspace:
        logits, runtime = (
            predict(network, image, config)
            if workspace is None
            else predict(network, image, config, workspace=workspace)
        )
        predicted = time.perf_counter()
        if workspace is not None:
            mask = postprocess_streaming(logits[0], inverse, bundle, workspace)
        elif bundle == "renalStructures_CECT_segmentation":
            assert inverse is not None
            mask = postprocess_renal(logits[0], inverse)
        elif bundle in _WHOLE_BODY_PIXDIM:
            assert inverse is not None
            mask = postprocess_wholebody(logits[0], inverse)
        else:
            mask = postprocess_brats(logits[0])
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


def brats(
    *,
    t1c_path: str,
    t1_path: str,
    t2_path: str,
    flair_path: str,
    output_dir: str,
    weights_path: str | None = None,
    overwrite: bool = False,
    arithmetic_profile: ArithmeticProfile = "torch-order-fp32",
) -> RunResult:
    """Segment BraTS MRI; omit weights_path to download its cached model folder.

    The four input volumes must share a voxel grid. An explicit folder contains
    ``segresnet-brats.safetensors``; an explicit safetensors file is also accepted.
    Automatic downloads use ``MedMLX/segresnet-brats-mlx``. All arithmetic and
    preprocessing are delegated unchanged to ``run``.

    Args:
        t1c_path: Contrast-enhanced T1 MRI NIfTI on the shared input grid.
        t1_path: T1 MRI NIfTI on the T1c grid.
        t2_path: T2 MRI NIfTI on the T1c grid.
        flair_path: FLAIR MRI NIfTI on the T1c grid.
        output_dir: Directory for the brain-tumor label map.
        weights_path: Optional folder containing ``segresnet-brats.safetensors``;
            omit to download and reuse the BraTS bundle. Explicit files remain accepted.
        overwrite: Replace an existing output mask when true.
        arithmetic_profile: ``torch-order-fp32`` (default) or ``compensated-fp32``;
            ``compensated-fp48`` requires the renal bundle.
    """
    folder = (
        weights_dir("segresnet-brats")
        if weights_path is None else Path(weights_path).expanduser()
    )
    checkpoint = folder / "segresnet-brats.safetensors" if folder.is_dir() else folder
    return run(
        bundle="brats_mri_segmentation", weights_path=str(checkpoint), output_dir=output_dir,
        t1c_path=t1c_path, t1_path=t1_path, t2_path=t2_path, flair_path=flair_path,
        overwrite=overwrite, arithmetic_profile=arithmetic_profile,
    )


def renal_cect(
    *,
    arterial_path: str,
    venous_path: str,
    excretory_path: str,
    output_dir: str,
    weights_path: str | None = None,
    overwrite: bool = False,
    arithmetic_profile: ArithmeticProfile = "torch-order-fp32",
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
    """
    folder = (
        weights_dir("segresnet-renal-cect")
        if weights_path is None else Path(weights_path).expanduser()
    )
    checkpoint = folder / "segresnet-renal-cect.safetensors" if folder.is_dir() else folder
    return run(
        bundle="renalStructures_CECT_segmentation", weights_path=str(checkpoint),
        output_dir=output_dir, arterial_path=arterial_path, venous_path=venous_path,
        excretory_path=excretory_path, overwrite=overwrite, arithmetic_profile=arithmetic_profile,
    )


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
    binary = transforms.threshold(transforms.sigmoid(values), 0.5)
    return transforms.merge_classes(binary)[0]


def preprocess_brats(volumes: list[Volume]) -> Float32Array:
    """``LoadImaged`` stacks the four modalities; ``NormalizeIntensityd`` per channel.

    The bundle has no reorientation or resampling, so the volumes stay on the file
    grid. Each modality is z-scored over its nonzero voxels only (zeros stay zero).
    Returns ``(1, 4, *spatial)`` float32 in the order of ``roles``.
    """
    image = np.stack([volume.data for volume in volumes], axis=0)
    return transforms.normalize_intensity_nonzero(image)[None]


def postprocess_brats(logits: Float32Array) -> transforms.npt.NDArray[np.int64]:
    """Sigmoid, ``Invertd`` (no spatial transform to undo), 0.5, BraTS labels 1/2/4.

    ``logits`` is ``(3, *spatial)`` with channels (tumor core, whole tumor, enhancing
    tumor). Returns an int64 label volume with values 0, 1, 2 and 4.
    """
    binary = transforms.threshold(transforms.sigmoid(logits), 0.5)
    return transforms.brats_labels(binary)[0]


def preprocess_wholebody(
    volumes: list[Volume], pixdim: tuple[float, float, float]
) -> tuple[Float32Array, SpatialInverse]:
    """The whole-body bundle's ``preprocessing`` from ``EnsureChannelFirstd`` onward.

    The single CT channel is reoriented to RAS, resampled to ``pixdim`` (1.5 mm for
    ``wholeBody_ct_segmentation``, 3 mm for the ``_lowres`` variant), normalized over
    its nonzero voxels (no HU window is applied), then rescaled to [-1, 1] by its own
    min and max. Returns the ``(1, 1, *spatial)`` float32 network input, with spatial
    axes in (R, A, S) order, and the image key's inverse record.
    """
    (volume,) = volumes
    oriented, affine, orientation = transforms.orient(volume.data[None], volume.affine, "RAS")
    resampled, affine, resampling = transforms.spacing(oriented, affine, pixdim)
    record = SpatialInverse(orientation, resampling, affine)
    normalized = transforms.normalize_intensity_nonzero(resampled)
    image = transforms.scale_intensity_minmax(normalized, -1.0, 1.0)
    return image[None], record


def postprocess_wholebody(logits: Float32Array, inverse: SpatialInverse) -> Float32Array:
    """Softmax, argmax, ``Invertd`` (nearest-neighbor, back to the source grid).

    ``logits`` is ``(105, *spatial)`` on the resampled grid. Returns the float32 label
    map (values 0-104) on the source grid.
    """
    # Kept to mirror the bundle's Activationsd(softmax=True) step; in exact
    # arithmetic it does not change the argmax.
    probabilities = transforms.softmax_channel(logits)
    labels = transforms.argmax_channel(probabilities)
    values, affine = transforms.invert_spacing(
        labels, inverse.affine, inverse.resampling, "nearest"
    )
    values, _ = transforms.invert_orientation(values, affine, inverse.orientation)
    return values[0]


def postprocess_streaming(
    logits: Float32Array,
    inverse: SpatialInverse | None,
    bundle: str,
    workspace: VolumeWorkspace,
) -> MaskArray:
    """Retain each bundle's activation/restoration/label order with bounded class storage."""
    from medmlx_core.volume_workspace import restored_argmax

    if bundle in _WHOLE_BODY_PIXDIM:
        assert inverse is not None
        probabilities = workspace.map_channels(logits, transforms.softmax_channel)
        labels = restored_argmax(probabilities, lambda channel: channel)[None]
        values, affine = transforms.invert_spacing(
            labels, inverse.affine, inverse.resampling, "nearest"
        )
        return transforms.invert_orientation(values, affine, inverse.orientation)[0][0]
    if bundle == "renalStructures_CECT_segmentation":
        assert inverse is not None
        labels: Float32Array | None = None
        for channel in range(1, logits.shape[0]):
            values, affine = transforms.invert_spacing(
                logits[channel : channel + 1], inverse.affine, inverse.resampling
            )
            values, _ = transforms.invert_orientation(values, affine, inverse.orientation)
            active = transforms.threshold(transforms.sigmoid(values), 0.5)[0] != 0
            if labels is None:
                labels = np.zeros(active.shape, dtype=np.float32)
            labels[(labels == 0) & active] = channel
        assert labels is not None
        return labels
    if bundle != "brats_mri_segmentation":
        raise InvalidInputError(f"Unsupported streaming SegResNet bundle: {bundle}")
    flat = logits.reshape(logits.shape[0], -1)
    output = np.empty(flat.shape[1], dtype=np.int64)
    for start in range(0, flat.shape[1], 65536):
        stop = min(start + 65536, flat.shape[1])
        binary = transforms.threshold(transforms.sigmoid(flat[:, start:stop]), 0.5)
        output[start:stop] = transforms.brats_labels(binary)[0]
    return output.reshape(logits.shape[1:])


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


__all__ = ["BRATS_MODEL", "BUNDLES", "MODEL", "RENAL_CECT_MODEL", "brats", "renal_cect", "run"]
