"""Registered SegResNet variants select their own cached weights and fixed bundle."""

from pathlib import Path
from typing import NoReturn

import pytest

from segresnet_mlx import medmlx


class InferenceReached(Exception):
    """Stop at the unchanged lower-level bundle runner."""


@pytest.mark.parametrize("automatic", [True, False])
@pytest.mark.parametrize("variant", ["brats", "renal_cect"])
def test_registered_variants_choose_fixed_bundle_and_model_folder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, automatic: bool, variant: str,
) -> None:
    """Each runner passes its published checkpoint and bundle to inference."""
    folder = tmp_path / "model"
    folder.mkdir()
    filename = (
        "segresnet-brats.safetensors" if variant == "brats"
        else "segresnet-renal-cect.safetensors"
    )
    checkpoint = folder / filename
    checkpoint.write_bytes(b"fixture")
    downloads: list[str] = []
    selected: dict[str, object] = {}

    def download(model_id: str) -> Path:
        """Record the requested model ID and return its local folder."""
        downloads.append(model_id)
        return folder

    def run(**kwargs: object) -> NoReturn:
        """Capture the runner arguments without executing a network."""
        selected.update(kwargs)
        raise InferenceReached

    monkeypatch.setattr(medmlx, "weights_dir", download)
    monkeypatch.setattr(medmlx, "run", run)
    with pytest.raises(InferenceReached):
        if variant == "brats":
            medmlx.brats(
                t1c_path="t1c.nii.gz", t1_path="t1.nii.gz", t2_path="t2.nii.gz",
                flair_path="flair.nii.gz", output_dir=str(tmp_path / "out"),
                weights_path=None if automatic else str(folder),
            )
        else:
            medmlx.renal_cect(
                arterial_path="arterial.nii.gz", venous_path="venous.nii.gz",
                excretory_path="excretory.nii.gz", output_dir=str(tmp_path / "out"),
                weights_path=None if automatic else str(folder),
            )
    model_id = "segresnet-brats" if variant == "brats" else "segresnet-renal-cect"
    bundle = "brats_mri_segmentation" if variant == "brats" else "renalStructures_CECT_segmentation"
    assert downloads == ([model_id] if automatic else [])
    assert selected["weights_path"] == str(checkpoint)
    assert selected["bundle"] == bundle
