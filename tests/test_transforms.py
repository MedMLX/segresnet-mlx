"""The original MONAI bundle transforms and NIfTI writer on seeded inputs."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import cast

import numpy as np
import pytest
from fixture_contracts import from_numpy
from hypothesis import given, settings
from hypothesis import strategies as st
from medmlx_core.volume_workspace import VolumeWorkspace
from reference_data import (
    BUNDLE_INPUT_KEYS,
    FIXTURES,
    bundle_fixture,
    load_fixture,
    write_inputs,
)

from segresnet_mlx import medmlx, transforms
from segresnet_mlx._arrays import MaskArray
from segresnet_mlx._nifti import nib
from segresnet_mlx.medmlx import BUNDLES

RENAL = "renalStructures_CECT_segmentation"
BRATS = "brats_mri_segmentation"
WHOLE_BODY = "wholeBody_ct_segmentation"
WHOLE_BODY_LOWRES = "wholeBody_ct_segmentation_lowres"
CASES = ("axis", "flipped", "oblique")
WHOLE_BODY_PIXDIM = {"highres": (1.5, 1.5, 1.5), "lowres": (3.0, 3.0, 3.0)}
RENAL_KEYS = BUNDLE_INPUT_KEYS["renal"]
BRATS_KEYS = BUNDLE_INPUT_KEYS["brats"]
WHOLE_BODY_KEYS = BUNDLE_INPUT_KEYS["wholebody"]


@pytest.mark.parametrize("separable", [False, True])
def test_trilinear_rounding_at_float32_midpoint(separable: bool) -> None:
    """Fusing corner accumulation prevents a one-ULP change in network input."""
    import torch.nn.functional as functional

    data = np.broadcast_to(
        np.array([-528.2138061523438, -508.7498474121094], dtype=np.float32),
        (1, 2, 2, 2),
    ).copy()
    # A two-voxel intensity ramp and subvoxel coordinates put the unfused sum
    # exactly at a float32 midpoint. The fused sum lies one float64 ULP above it.
    grid = np.array(
        [127.2 - 127 - 0.5, 272.7935916750003 - 272 - 0.5, 464.4863858250004 - 464 - 0.5],
        dtype=np.float64,
    )
    expected = functional.grid_sample(
        from_numpy(data[None].astype(np.float64)),
        from_numpy(grid[None, None, None, None]),
        mode="bilinear", padding_mode="border", align_corners=False,
    ).numpy()[0].astype(np.float32)
    affine = np.eye(4)
    affine[:3, 3] = grid[::-1] + 0.5
    if not separable:
        affine[0, 1] = 0.125
    actual, _, _ = transforms.spatial_resample(data, np.eye(4), affine, (1, 1, 1))
    np.testing.assert_array_equal(actual, expected)


def test_sigmoid_avoids_overflow_and_preserves_source_threshold_rounding() -> None:
    """Compare extreme sigmoid values and threshold decisions with Torch."""
    limit = np.finfo(np.float32).max
    values = np.array(
        [-limit, -1000, -100, -90, -80, -1, -1e-7, 0, 1e-7, 1, 80, 90, 100, limit],
        dtype=np.float32,
    )
    expected = from_numpy(values).sigmoid().numpy()
    with np.errstate(over="raise", invalid="raise"):
        actual = transforms.sigmoid(values)
    assert actual.dtype == np.float32
    assert np.isfinite(actual).all()
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-6)
    np.testing.assert_array_equal(transforms.threshold(actual), expected >= 0.5)


@given(seed=st.integers(0, 2**32 - 1))
@settings(max_examples=25, database=None, deadline=None)
def test_softmax_channel_matches_torch(seed: int) -> None:
    """Compare channel-wise float32 softmax with Torch across seeded inputs."""
    import torch

    data = np.random.default_rng(seed).normal(0, 5, (6, 4, 5, 3)).astype(np.float32)
    expected = torch.softmax(from_numpy(data), dim=0).numpy()
    actual = transforms.softmax_channel(data)
    assert actual.dtype == np.float32
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-6)


@given(seed=st.integers(0, 2**32 - 1))
@settings(max_examples=25, database=None, deadline=None)
def test_argmax_channel_matches_torch(seed: int) -> None:
    """Compare channel-wise argmax labels with Torch across seeded inputs."""
    import torch

    data = np.random.default_rng(seed).normal(0, 5, (6, 4, 5, 3)).astype(np.float32)
    expected = torch.argmax(from_numpy(data), dim=0, keepdim=True).to(torch.float32).numpy()
    actual = transforms.argmax_channel(data)
    assert actual.dtype == np.float32
    np.testing.assert_array_equal(actual, expected)


@given(
    seed=st.integers(0, 2**32 - 1),
    constant=st.booleans(),
)
@settings(max_examples=25, database=None, deadline=None)
def test_scale_intensity_minmax_matches_monai(seed: int, constant: bool) -> None:
    """Compare min-max scaling with MONAI for varying and constant inputs."""
    import torch
    from monai.transforms.utils import rescale_array

    rng = np.random.default_rng(seed)
    data = (
        np.full((4, 3, 2), 7.0, dtype=np.float32)
        if constant
        else rng.normal(0, 5, (4, 3, 2)).astype(np.float32)
    )
    expected = cast(
        "torch.Tensor", rescale_array(from_numpy(data), -1.0, 1.0, dtype=np.float32)
    ).numpy()
    actual = transforms.scale_intensity_minmax(data, -1.0, 1.0)
    assert actual.dtype == np.float32
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("oblique", [False, True])
@given(seed=st.integers(0, 2**32 - 1))
@settings(max_examples=10, database=None, deadline=None)
def test_nearest_resample_matches_monai_spatial_resample(seed: int, oblique: bool) -> None:
    """``Invertd(nearest_interp=True)``: the whole-body bundle's label-map inverse."""
    from monai.data.meta_tensor import MetaTensor
    from monai.transforms.spatial.array import SpatialResample

    rng = np.random.default_rng(seed)
    src_affine = np.diag([1.0, 1.0, 1.0, 1.0])
    if oblique:
        angle = 0.3
        dst_affine = np.eye(4)
        dst_affine[:3, :3] = np.array(
            [[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]]
        ) @ np.diag([1.2, 0.9, 1.4])
        dst_affine[:3, 3] = [1.0, -2.0, 0.5]
    else:
        dst_affine = np.diag([1.5, 1.5, 1.5, 1.0])
    data = rng.integers(0, 105, (1, 9, 8, 7)).astype(np.float32)

    expected = SpatialResample()(
        MetaTensor(from_numpy(data), affine=from_numpy(src_affine)),
        dst_affine=from_numpy(dst_affine),
        spatial_size=(6, 7, 5),
        mode="nearest",
        padding_mode="border",
        align_corners=False,
        dtype=np.float64,
    ).numpy()
    actual, _, _ = transforms.spatial_resample(data, src_affine, dst_affine, (6, 7, 5), "nearest")
    assert actual.dtype == np.float32
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("case", CASES)
def test_renal_transforms_match_the_bundle(tmp_path: Path, case: str) -> None:
    """Compare renal preprocessing and inverse labels with recorded bundle cases."""
    arrays, _ = bundle_fixture("renal")
    paths = write_inputs(tmp_path, "renal", case)
    volumes = medmlx.load_inputs([(role, paths[role]) for role in RENAL_KEYS])
    image, inverse = medmlx.preprocess_renal(volumes)
    assert image.dtype == np.float32
    # Torch and NumPy resampling/scaling need numerical parity, not identical
    # floating-point evaluation. Allow one float32 rounding unit on the unit range.
    assert np.allclose(image[0], arrays[f"{case}::image"], rtol=0, atol=np.finfo(np.float32).eps)
    mask = medmlx.postprocess_renal(arrays[f"{case}::logits"].astype(np.float32), inverse)
    with VolumeWorkspace(tmp_path) as workspace:
        streamed = medmlx.postprocess_streaming(
            arrays[f"{case}::logits"].astype(np.float32), inverse, RENAL, workspace
        )
    np.testing.assert_array_equal(streamed, mask)
    assert mask.dtype == np.float32
    assert np.array_equal(mask, arrays[f"{case}::mask"])


@pytest.mark.parametrize("case", CASES)
def test_brats_transforms_match_the_bundle(tmp_path: Path, case: str) -> None:
    """Compare BraTS normalization and labels with recorded bundle cases."""
    arrays, _ = bundle_fixture("brats")
    paths = write_inputs(tmp_path, "brats", case)
    volumes = medmlx.load_inputs([(role, paths[role]) for role in BRATS_KEYS])
    image = medmlx.preprocess_brats(volumes)
    expected = arrays[f"{case}::image"]
    assert image.dtype == np.float32
    # Torch reduces in float32 and the port in float64. The largest observed
    # difference on these seeded, single-thread CPU cases is exactly 2**-20.
    assert np.array_equal(image[0] == 0, expected == 0)
    assert np.allclose(image[0], expected, rtol=0, atol=2**-20)
    mask = medmlx.postprocess_brats(arrays[f"{case}::logits"].astype(np.float32))
    with VolumeWorkspace(tmp_path) as workspace:
        streamed = medmlx.postprocess_streaming(
            arrays[f"{case}::logits"].astype(np.float32), None, BRATS, workspace
        )
    np.testing.assert_array_equal(streamed, mask)
    assert np.array_equal(mask.astype(np.uint8), arrays[f"{case}::mask"])


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize(
    ("variant", "pixdim"),
    [
        ("wholebody", WHOLE_BODY_PIXDIM["highres"]),
        ("wholebody_lowres", WHOLE_BODY_PIXDIM["lowres"]),
    ],
)
def test_wholebody_transforms_match_the_bundle(
    tmp_path: Path, case: str, variant: str, pixdim: tuple[float, float, float]
) -> None:
    """Compare both whole-body resolutions with recorded bundle cases."""
    arrays, _ = bundle_fixture(variant)
    paths = write_inputs(tmp_path, variant, case)
    volumes = medmlx.load_inputs([(role, paths[role]) for role in WHOLE_BODY_KEYS])
    image, inverse = medmlx.preprocess_wholebody(volumes, pixdim)
    assert image.dtype == np.float32
    # Resampling then per-volume normalize/rescale needs numerical parity with the
    # Torch graph, not identical floating-point evaluation.
    assert np.allclose(image[0], arrays[f"{case}::image"], rtol=0, atol=2**-21)
    mask = medmlx.postprocess_wholebody(arrays[f"{case}::logits"].astype(np.float32), inverse)
    with VolumeWorkspace(tmp_path) as workspace:
        streamed = medmlx.postprocess_streaming(
            arrays[f"{case}::logits"].astype(np.float32),
            inverse,
            WHOLE_BODY if variant == "wholebody" else WHOLE_BODY_LOWRES,
            workspace,
        )
    np.testing.assert_array_equal(streamed, mask)
    assert mask.dtype == np.float32
    assert np.array_equal(mask, arrays[f"{case}::mask"])


@pytest.mark.parametrize(
    ("name", "bundle"),
    [
        ("renal", RENAL),
        ("brats", BRATS),
        ("wholebody", WHOLE_BODY),
        ("wholebody_lowres", WHOLE_BODY_LOWRES),
    ],
)
def test_mask_file_matches_the_bundle_writer(tmp_path: Path, name: str, bundle: str) -> None:
    """Compare saved label values, dtype, affine, and form codes with MONAI."""
    arrays, saved = bundle_fixture(name)
    for case in CASES:
        reference = nib.load(
            cast(str, write_inputs(tmp_path, name, case)[BUNDLES[bundle].roles[0]])
        )
        path = tmp_path / f"{case}.nii.gz"
        medmlx.write_atomic(
            path,
            cast(MaskArray, arrays[f"{case}::mask"]),
            reference.affine,
            BUNDLES[bundle].output_dtype,
        )
        written = nib.load(str(path))
        assert str(written.get_data_dtype()) == saved[case]["dtype"]
        assert list(written.shape) == saved[case]["shape"]
        assert int(written.header["qform_code"]) == saved[case]["qform_code"]
        assert int(written.header["sform_code"]) == saved[case]["sform_code"]
        assert np.allclose(written.affine, saved[case]["affine"], rtol=0, atol=1e-6)
        assert np.array_equal(np.asanyarray(written.dataobj), arrays[f"{case}::mask"])


@pytest.mark.parametrize("forms", ["disagree", "unset", "sform_spacing"])
def test_loader_uses_the_upstream_nibabel_affine(tmp_path: Path, forms: str) -> None:
    """Compare NIfTI data and selected affine with the recorded MONAI reader."""
    arrays, _ = load_fixture(FIXTURES / "nifti_reader.npz")
    path = tmp_path / f"{forms}.nii.gz"
    path.write_bytes(arrays[f"{forms}::nifti"].tobytes())
    original = path.read_bytes()
    volume = medmlx.load_volume(path, "arterial_path")
    assert np.array_equal(volume.affine, arrays[f"{forms}::affine"])
    assert np.array_equal(volume.data, arrays[f"{forms}::data"])
    assert path.read_bytes() == original


@given(
    seed=st.integers(0, 2**32 - 1),
    spacing=st.tuples(*(st.floats(0.25, 5, allow_nan=False) for _ in range(3))),
)
@settings(max_examples=25, database=None, deadline=None)
def test_nifti_reader_writer_round_trip(seed: int, spacing: tuple[float, float, float]) -> None:
    """Preserve voxel values and the saved affine through NIfTI I/O."""
    data = np.random.default_rng(seed).normal(size=(2, 3, 4)).astype(np.float32)
    with TemporaryDirectory() as directory:
        path = Path(directory) / "image.nii.gz"
        medmlx.write_atomic(path, data, np.diag([*spacing, 1.0]), "float32")
        source = nib.load(str(path))
        volume = medmlx.load_volume(path, "image_path")
        assert volume.shape == data.shape
        assert np.array_equal(volume.data, data)
        assert np.array_equal(volume.affine, source.affine)
