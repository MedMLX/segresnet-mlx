"""medmlx declaration, bundle transform parity with MONAI, and end-to-end runs.

The parity fixtures are recorded by scripts/make_medmlx_fixtures.py, which runs the
original bundles' pinned MONAI transforms on seeded inputs.
"""

from __future__ import annotations

import importlib
import json
import subprocess
import sys
import tomllib
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, cast

from fixture_contracts import ArgumentChange, ConstructorSpec, RenalInputPaths, RunArguments

from segresnet_mlx._arrays import Float32Array, NumericArray
from segresnet_mlx._records import Runtime

if TYPE_CHECKING:
    from segresnet_mlx.network import SegResNet

import numpy as np
import pytest
from medmlx_core.errors import AssetNotReadyError, InvalidInputError
from medmlx_core.typing import ArrayFactory
from reference_data import BUNDLE_INPUT_KEYS, bundle_fixture, write_inputs

from segresnet_mlx import medmlx
from segresnet_mlx._nifti import nib
from segresnet_mlx.medmlx import BUNDLES, MODEL, run
from segresnet_mlx.topology import SegResNetSpec
from segresnet_mlx.weights import required_shapes

ROOT = Path(__file__).resolve().parents[1]
RENAL = "renalStructures_CECT_segmentation"
BRATS = "brats_mri_segmentation"
WHOLE_BODY = "wholeBody_ct_segmentation"
WHOLE_BODY_LOWRES = "wholeBody_ct_segmentation_lowres"
CASES = ("axis", "flipped", "oblique")
RENAL_KEYS = BUNDLE_INPUT_KEYS["renal"]
BRATS_KEYS = BUNDLE_INPUT_KEYS["brats"]


def _weights(path: Path, spec: SegResNetSpec, seed: int) -> Path:
    """Write deterministic synthetic weights for a specified network topology."""
    from segresnet_mlx._runtime import import_mlx

    mx = import_mlx()

    rng = np.random.default_rng(seed)
    mx.save_safetensors(
        str(path),
        {
            key: cast(ArrayFactory, mx.array)(rng.normal(0.0, 0.2, shape).astype(np.float32))
            for key, shape in required_shapes(spec).items()
        },
    )
    return path


@pytest.fixture
def reduced(monkeypatch: pytest.MonkeyPatch) -> dict[str, SegResNetSpec]:
    """Use small seeded topologies to exercise the runner without real weights."""
    specs = {
        RENAL: SegResNetSpec(3, 6, 8, blocks_down=(1,), blocks_up=()),
        BRATS: SegResNetSpec(4, 3, 8, blocks_down=(1,), blocks_up=()),
        WHOLE_BODY: SegResNetSpec(1, 105, 8, blocks_down=(1,), blocks_up=()),
        WHOLE_BODY_LOWRES: SegResNetSpec(1, 105, 8, blocks_down=(1,), blocks_up=()),
    }
    for name, spec in specs.items():
        monkeypatch.setitem(BUNDLES, name, replace(BUNDLES[name], spec=spec, roi_size=(8, 8, 8)))
    return specs


def test_entry_point_declares_the_runner() -> None:
    """Check registered model entry points and the generic runner declaration."""
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert project["project"]["entry-points"]["medmlx.models"] == {
        "segresnet-brats": "segresnet_mlx.medmlx:BRATS_MODEL",
        "segresnet-renal-cect": "segresnet_mlx.medmlx:RENAL_CECT_MODEL",
    }
    assert {"id", "label", "task", "run"} <= MODEL.keys()
    assert MODEL["id"] == "segresnet"
    assert MODEL["task"] == "segmentation"
    assert MODEL["run"] == "segresnet_mlx.medmlx:run"
    module, _, attribute = MODEL["run"].partition(":")
    assert getattr(importlib.import_module(module), attribute) is run


@pytest.mark.parametrize(
    ("bundle", "name", "labels", "dtype"),
    [
        (RENAL, "renal", set(range(6)), np.float32),
        (BRATS, "brats", {0, 1, 2, 4}, np.uint8),
        (WHOLE_BODY, "wholebody", set(range(105)), np.float32),
        (WHOLE_BODY_LOWRES, "wholebody_lowres", set(range(105)), np.float32),
    ],
)
@pytest.mark.mlx
def test_run_writes_the_mask_on_the_input_grid(
    tmp_path: Path,
    reduced: dict[str, SegResNetSpec],
    bundle: str,
    name: str,
    labels: set[int],
    dtype: type,
) -> None:
    """Verify synthetic bundle outputs, labels, grid, and overwrite behavior."""
    inputs = write_inputs(tmp_path, name, "oblique")
    weights = _weights(tmp_path / "weights.safetensors", reduced[bundle], 5)
    if bundle == BRATS:
        # Hugging Face cache snapshots expose a named symlink to an extensionless blob.
        blob = tmp_path / "cached-blob"
        weights.rename(blob)
        weights.symlink_to(blob)
    output_dir = tmp_path / "out"
    result = run(bundle=bundle, weights_path=str(weights), output_dir=str(output_dir), **inputs)

    mask_path = output_dir.resolve() / f"{bundle}_mask.nii.gz"
    assert result["outputs"] == {"mask": mask_path.as_posix()}
    reference = nib.load(cast(str, next(iter(inputs.values()))))
    written = nib.load(str(mask_path))
    data = np.asanyarray(written.dataobj)
    assert written.shape == reference.shape
    assert np.array_equal(written.affine, reference.affine)
    assert written.header.get_xyzt_units()[0] == "mm"
    assert data.dtype == dtype
    assert set(np.unique(data).tolist()) <= labels
    arrays, _ = bundle_fixture(name)
    details = result["details"]
    assert details["bundle"] == bundle
    assert details["sliding_window"]["sw_batch_size"] == (4 if bundle == RENAL else 1)
    assert np.array_equal(data, arrays["oblique::runner_mask"]), "Mask differs from upstream"
    assert details["labels_present"] == sorted(int(value) for value in np.unique(data))
    assert details["output_dtype"] == np.dtype(dtype).name
    assert details["input_shape"] == list(reference.shape)
    assert details["network_input_shape"] == list(arrays["oblique::image"].shape[1:])
    json.dumps(result, allow_nan=False)

    with pytest.raises(InvalidInputError, match="overwrite"):
        run(bundle=bundle, weights_path=str(weights), output_dir=str(output_dir), **inputs)
    again = run(
        bundle=bundle,
        weights_path=str(weights),
        output_dir=str(output_dir),
        overwrite=True,
        **inputs,
    )
    assert again["outputs"] == result["outputs"]


@pytest.mark.parametrize(
    ("name", "bundle", "in_channels", "out_channels"),
    [
        ("renal", RENAL, 3, 6),
        ("wholebody", WHOLE_BODY, 1, 105),
    ],
)
@pytest.mark.mlx
def test_run_imports_no_reference_frameworks(
    tmp_path: Path, name: str, bundle: str, in_channels: int, out_channels: int
) -> None:
    """Run synthetic inference while Torch and MONAI imports are blocked."""
    inputs = write_inputs(tmp_path, name, "flipped")
    spec = SegResNetSpec(in_channels, out_channels, 8, blocks_down=(1,), blocks_up=())
    weights = _weights(tmp_path / "weights.safetensors", spec, 4)
    code = """
import importlib.abc, json, sys
from dataclasses import replace

class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"torch", "monai"}:
            raise ModuleNotFoundError(f"Forbidden runtime dependency: {fullname}")

sys.meta_path.insert(0, Block())
from segresnet_mlx import SegResNetSpec, medmlx

name = sys.argv[2]
spec = SegResNetSpec(*json.loads(sys.argv[3]), blocks_down=(1,), blocks_up=())
medmlx.BUNDLES[name] = replace(medmlx.BUNDLES[name], spec=spec, roi_size=(8, 8, 8))
arguments = json.loads(sys.argv[1])
result = medmlx.run(bundle=name, **arguments)
loaded = sorted(m for m in sys.modules if m.split(".")[0] in {"torch", "monai"})
print(json.dumps({"mask": result["outputs"]["mask"], "forbidden": loaded}))
"""
    arguments = {"weights_path": str(weights), "output_dir": str(tmp_path / "out"), **inputs}
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            code,
            json.dumps(arguments),
            bundle,
            json.dumps([in_channels, out_channels, 8]),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    report = json.loads(completed.stdout.strip().splitlines()[-1])
    assert report["forbidden"] == []
    assert Path(report["mask"]).is_file()


@pytest.mark.mlx
def test_run_executes_the_full_renal_topology(tmp_path: Path) -> None:
    """Exercise the full renal topology with synthetic weights and input."""
    inputs = write_inputs(tmp_path, "renal", "axis")
    weights = _weights(tmp_path / "weights.safetensors", BUNDLES[RENAL].spec, 9)
    result = run(bundle=RENAL, weights_path=str(weights), output_dir=str(tmp_path), **inputs)
    written = nib.load(result["outputs"]["mask"])
    assert written.shape == nib.load(cast(RenalInputPaths, inputs)["arterial_path"]).shape


def _valid_run(tmp_path: Path, reduced: dict[str, SegResNetSpec]) -> RunArguments:
    """Prepare valid renal arguments with placeholder weights for admission tests."""
    weights = tmp_path / "weights.safetensors"
    # Argument and volume validation must fail before the weights are read.
    weights.write_bytes(b"unused")
    return cast(
        RunArguments,
        {
            "bundle": RENAL,
            "weights_path": str(weights),
            "output_dir": str(tmp_path / "out"),
            **write_inputs(tmp_path, "renal", "axis"),
        },
    )


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"bundle": "spleen_ct_segmentation"}, "Unknown SegResNet bundle"),
        ({"excretory_path": None}, "missing excretory_path"),
        ({"flair_path": "flair.nii.gz"}, "not applicable: flair_path"),
        ({"venous_path": "absent.nii.gz"}, "venous_path not found"),
    ],
)
def test_run_rejects_invalid_arguments(
    tmp_path: Path,
    reduced: dict[str, SegResNetSpec],
    change: ArgumentChange,
    message: str,
) -> None:
    """Reject unknown bundles, missing inputs, and inapplicable inputs."""
    arguments = cast(
        RunArguments,
        cast(dict[str, object], _valid_run(tmp_path, reduced)) | cast(dict[str, object], change),
    )
    with pytest.raises(InvalidInputError, match=message):
        run(**arguments)


@pytest.mark.parametrize(
    ("corruption", "message"),
    [
        ("four_d", "must be a 3D NIfTI"),
        ("nonfinite", "non-finite voxels"),
        ("grid", "must share arterial_path's voxel grid"),
    ],
)
def test_run_rejects_invalid_volumes(
    tmp_path: Path,
    reduced: dict[str, SegResNetSpec],
    corruption: str,
    message: str,
) -> None:
    """Reject non-3D, non-finite, and mismatched-grid volumes before output."""
    arguments = _valid_run(tmp_path, reduced)
    source = nib.load(cast(str, arguments["venous_path"]))
    data = np.asanyarray(source.dataobj).copy()
    affine = source.affine.copy()
    if corruption == "four_d":
        data = data[..., None]
    elif corruption == "nonfinite":
        data[0, 0, 0] = np.nan
    elif corruption == "grid":
        affine[0, 3] += 1.0
    image = nib.Nifti1Image(data, affine)
    image.set_sform(affine, code=1)
    image.set_qform(affine, code=1)
    path = tmp_path / "venous_changed.nii.gz"
    nib.save(image, str(path))
    with pytest.raises(InvalidInputError, match=message):
        run(**cast(RunArguments, arguments | {"venous_path": str(path)}))
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize(
    ("corruption", "message"),
    [
        ("meter", "spatial units"),
        ("micron", "spatial units"),
        ("invalid_units", "spatial units"),
        ("overflow", "float32"),
        ("complex", "real"),
        ("singular", "affine"),
    ],
)
def test_run_rejects_unsupported_units_values_and_geometry(
    tmp_path: Path, reduced: dict[str, SegResNetSpec], corruption: str, message: str
) -> None:
    """Reject inputs that would misstate physical scale or corrupt float32 inference."""
    arguments = _valid_run(tmp_path, reduced)
    source = nib.load(cast(str, arguments["venous_path"]))
    data = np.asanyarray(source.dataobj).copy()
    if corruption == "overflow":
        data = data.astype(np.float64)
        data[0, 0, 0] = 1e100
    elif corruption == "complex":
        data = data.astype(np.complex64)
        data[0, 0, 0] += 1j
    initial_affine = np.eye(4) if corruption == "singular" else source.affine.copy()
    image = nib.Nifti1Image(data, initial_affine)
    if corruption in ("meter", "micron"):
        image.header.set_xyzt_units(corruption)
    elif corruption == "invalid_units":
        image.header["xyzt_units"] = 7
    elif corruption == "singular":
        affine = np.eye(4)
        affine[:3, :3] = [[1, 1, 0], [0, 0, 0], [0, 0, 1]]
        image.set_sform(affine, code=1)
    path = tmp_path / "venous_changed.nii.gz"
    nib.save(image, str(path))
    with pytest.raises(InvalidInputError, match=message):
        run(**cast(RunArguments, arguments | {"venous_path": str(path)}))
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize(
    ("weights", "message"),
    [
        ("absent", "weights not found"),
        ("torch", "must be converted MLX safetensors"),
        pytest.param("mismatched", "do not match", marks=pytest.mark.mlx),
    ],
)
def test_run_requires_matching_converted_weights(
    tmp_path: Path,
    reduced: dict[str, SegResNetSpec],
    weights: str,
    message: str,
) -> None:
    """Reject absent, non-safetensors, and mismatched weights."""
    arguments = _valid_run(tmp_path, reduced)
    if weights == "absent":
        path = tmp_path / "missing.safetensors"
    elif weights == "torch":
        path = tmp_path / "model.pt"
        path.write_bytes(b"not used")
    else:
        path = _weights(tmp_path / "brats.safetensors", reduced[BRATS], 2)
    with pytest.raises(AssetNotReadyError, match=message):
        run(**cast(RunArguments, arguments | {"weights_path": str(path)}))
    assert not (tmp_path / "out").exists()


def test_declaration_import_loads_no_mlx_or_reference_stack() -> None:
    """Check that the model declaration imports without MLX or reference frameworks."""
    code = (
        "import sys\n"
        "import segresnet_mlx.medmlx as m\n"
        "assert m.MODEL['run'] == 'segresnet_mlx.medmlx:run'\n"
        "loaded = {name.split('.')[0] for name in sys.modules}\n"
        "assert not loaded & {'mlx', 'torch', 'monai'}, loaded\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)


@pytest.mark.parametrize(
    ("name", "bundle"),
    [
        ("renal", RENAL),
        ("brats", BRATS),
        ("wholebody", WHOLE_BODY),
        ("wholebody_lowres", WHOLE_BODY_LOWRES),
    ],
)
@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("suffix", ["runner", "runner_decoder"])
@pytest.mark.mlx
def test_runner_matches_the_original_upstream_pipeline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    bundle: str,
    case: str,
    suffix: str,
) -> None:
    """Compare seeded runner logits and masks with recorded MONAI pipelines."""
    from segresnet_mlx._runtime import import_mlx

    mx = import_mlx()

    arrays, _ = bundle_fixture(name)
    definition = json.loads(str(arrays[f"{suffix}::spec"]))
    definition["blocks_down"] = tuple(definition["blocks_down"])
    definition["blocks_up"] = tuple(definition["blocks_up"])
    spec = SegResNetSpec(**cast(ConstructorSpec, definition))
    monkeypatch.setitem(BUNDLES, bundle, replace(BUNDLES[bundle], spec=spec, roi_size=(8, 8, 8)))
    predict = medmlx.predict

    def measured(
        network: SegResNet, image: Float32Array, config: medmlx.Bundle
    ) -> tuple[Float32Array, Runtime]:
        """Check raw logits against the recording before returning the runtime record."""
        output, runtime = predict(network, image, config)
        # Torch and MLX accumulate convolutions differently, and BraTS
        # normalization also rounds differently. Seeded raw logits allow 1e-6.
        np.testing.assert_allclose(
            output, cast(NumericArray, arrays[f"{case}::{suffix}_logits"]), rtol=0, atol=1e-6
        )
        return output, runtime

    monkeypatch.setattr(medmlx, "predict", measured)
    weights = tmp_path / "seeded.safetensors"
    prefix = f"{suffix}::weight::"
    mx.save_safetensors(
        str(weights),
        {
            key.removeprefix(prefix): cast(ArrayFactory, mx.array)(value)
            for key, value in arrays.items()
            if key.startswith(prefix)
        },
    )
    result = run(
        bundle=bundle,
        weights_path=str(weights),
        output_dir=str(tmp_path / "out"),
        **write_inputs(tmp_path, name, case),
    )
    written = nib.load(result["outputs"]["mask"])
    # The label map must agree exactly, including class priority and threshold ties.
    assert np.array_equal(np.asanyarray(written.dataobj), arrays[f"{case}::{suffix}_mask"])
    assert result["details"]["sliding_window"]["sw_batch_size"] == (4 if name == "renal" else 1)
