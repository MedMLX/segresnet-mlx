"""Record MONAI NIfTI reader behavior on HCC-TACE CT with controlled forms."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from monai import __version__ as monai_version
from monai.transforms import LoadImage

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    """Read real CT voxels under the pre-existing qform and sform test cases."""
    if monai_version != "1.6.0" or not torch.__version__.startswith("2.14."):
        raise ValueError("MONAI 1.6.0 and Torch 2.14 are required")
    ct = np.asarray(
        nib.load(Path.home() / ".cache/medmlx/test-data/hcc_063_ct.nii.gz").dataobj,
        dtype=np.float32,
    )[100:109, 210:218, 55:61]
    with np.load(
        ROOT / "tests/fixtures/darwin-arm64/nifti_reader.npz", allow_pickle=False
    ) as source:
        arrays = {key: source[key] for key in source.files}
    metadata = json.loads(arrays["metadata"].item())
    metadata["image_source"] = "HCC-TACE-Seg HCC_063 contrast CT"
    metadata.pop("seed", None)
    arrays["metadata"] = np.asarray(json.dumps(metadata))
    reader = LoadImage(reader="NibabelReader", image_only=False, dtype=np.float32)
    with tempfile.TemporaryDirectory(prefix="segresnet-reader-") as temp:
        for name in metadata["cases"]:
            header = nib.Nifti1Header(arrays[f"{name}::header"].tobytes())
            image = nib.Nifti1Image(ct, None, header)
            path = Path(temp) / f"{name}.nii.gz"
            nib.save(image, path)
            loaded, facts = reader(str(path))
            arrays[f"{name}::nifti"] = np.frombuffer(path.read_bytes(), dtype=np.uint8)
            arrays[f"{name}::data"] = loaded.numpy()
            arrays[f"{name}::affine"] = np.asarray(facts["affine"], dtype=np.float64)
    destination = ROOT / "work/real-reference/nifti_reader.npz"
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, **arrays)


if __name__ == "__main__":
    main()
