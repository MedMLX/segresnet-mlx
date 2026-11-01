"""End-to-end parity with the MONAI bundles and argument rejection before downloads.

The parity fixtures are recorded by scripts/make_medmlx_fixtures.py, which runs the
original bundles' pinned MONAI transforms on seeded inputs.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, NoReturn, cast

from segresnet_mlx._arrays import Float32Array, NumericArray
from segresnet_mlx._records import Runtime

if TYPE_CHECKING:
    from segresnet_mlx.network import SegResNet

import numpy as np
import pytest
from medmlx_core.errors import InvalidInputError
from reference_data import bundle_fixture, mlx_weights, segresnet_spec, write_inputs

from segresnet_mlx import medmlx
from segresnet_mlx._nifti import nib
from segresnet_mlx.medmlx import BUNDLES, run

RENAL = "renalStructures_CECT_segmentation"
BRATS = "brats_mri_segmentation"
WHOLE_BODY = "wholeBody_ct_segmentation"
WHOLE_BODY_LOWRES = "wholeBody_ct_segmentation_lowres"
REGISTERED = {
    "brats": (BRATS, ("t1c_path", "t1_path", "t2_path", "flair_path")),
    "renal_cect": (RENAL, ("arterial_path", "venous_path", "excretory_path")),
}


@pytest.mark.parametrize("variant", list(REGISTERED))
@pytest.mark.parametrize(
    "problem", ["absent-path", "unknown-profile", "non-bool", "existing-output"]
)
def test_registered_runners_reject_arguments_before_downloading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, variant: str, problem: str
) -> None:
    """Each category of bad argument fails before any weight download."""

    def download(model_id: str) -> NoReturn:
        """Fail the test if weights are fetched before the arguments are checked."""
        pytest.fail(f"downloaded {model_id} before checking arguments")

    monkeypatch.setattr(medmlx, "weights_dir", download)
    bundle, roles = REGISTERED[variant]
    output = tmp_path / "out"
    arguments: dict[str, object] = {"output_dir": str(output)}
    for role in roles:
        path = tmp_path / f"{role}.nii.gz"
        path.write_bytes(b"read only after the arguments are checked")
        arguments[role] = str(path)
    if problem == "absent-path":
        arguments[roles[0]] = str(tmp_path / "absent.nii.gz")
    elif problem == "unknown-profile":
        arguments["arithmetic_profile"] = "fp16"
    elif problem == "non-bool":
        arguments["overwrite"] = "false"
    else:
        output.mkdir()
        (output / f"{bundle}_mask.nii.gz").write_bytes(b"previous")
    with pytest.raises(InvalidInputError):
        getattr(medmlx, variant)(**arguments)


@pytest.mark.parametrize(
    ("name", "bundle"),
    [
        ("renal", RENAL),
        ("brats", BRATS),
        ("wholebody", WHOLE_BODY),
        ("wholebody_lowres", WHOLE_BODY_LOWRES),
    ],
)
@pytest.mark.parametrize("case", ["axis", "flipped", "oblique"])
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
    spec = segresnet_spec(json.loads(str(arrays[f"{suffix}::spec"])))
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
    mx.save_safetensors(str(weights), mlx_weights(arrays, mx, f"{suffix}::weight::"))
    result = run(
        bundle=bundle,
        weights_path=str(weights),
        output_dir=str(tmp_path / "out"),
        **write_inputs(tmp_path, name, case),
    )
    written = nib.load(result["outputs"]["mask"])
    # The label map must agree exactly, including class priority and threshold ties.
    assert np.array_equal(np.asanyarray(written.dataobj), arrays[f"{case}::{suffix}_mask"])
