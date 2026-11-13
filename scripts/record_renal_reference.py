"""Record the pinned renal bundle on aligned HCC-TACE CT inputs."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from monai import __version__ as monai_version
from monai.data import MetaTensor
from monai.inferers import SlidingWindowInferer
from monai.transforms import (
    Activationsd,
    AsDiscreted,
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    Invertd,
    LoadImaged,
    Orientationd,
    ScaleIntensityRanged,
    Spacingd,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "src"))
from reference_contracts import seeded_network  # noqa: E402


def main() -> None:
    """Run source renal transforms and two seeded graph variants on real CT."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    args = parser.parse_args()
    if monai_version != "1.6.0" or not torch.__version__.startswith("2.14."):
        raise ValueError("MONAI 1.6.0 and Torch 2.14 are required")
    if args.bundle.name != "8ccbcd9573c004dc435d1913c0459ccd1a110ed1":
        raise ValueError("Supply the pinned renal bundle snapshot")
    spec = importlib.util.spec_from_file_location(
        "renal_transforms", args.bundle / "scripts/my_transforms.py"
    )
    if spec is None or spec.loader is None:
        raise ValueError("Cannot load renal bundle transforms")
    source = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(source)
    keys = ["artery", "vein", "excret"]
    pre = Compose(
        [
            LoadImaged(keys=keys, image_only=False),
            EnsureChannelFirstd(keys=keys),
            Orientationd(keys=keys, axcodes="LPS"),
            Spacingd(keys=keys, pixdim=(0.8, 0.8, 0.8), mode="bilinear"),
            source.ConcatImages(keys_merge=keys, keys_out="image"),
            ScaleIntensityRanged(
                keys="image", a_min=-1000, a_max=1000, b_min=0, b_max=1, clip=True
            ),
            EnsureTyped(keys="image"),
        ]
    )
    post = Compose(
        [
            Invertd(
                keys="pred", transform=pre, orig_keys="artery", nearest_interp=False, to_tensor=True
            ),
            Activationsd(keys="pred", sigmoid=True),
            AsDiscreted(keys="pred", threshold=0.5),
            source.MergeClassesd(keys="pred"),
        ]
    )
    ct = np.asarray(
        nib.load(Path.home() / ".cache/medmlx/test-data/hcc_063_ct.nii.gz").dataobj,
        dtype=np.float32,
    )[100:109, 210:218, 55:61]
    with np.load(
        ROOT / "tests/fixtures/darwin-arm64/medmlx_renal.npz", allow_pickle=False
    ) as fixture:
        arrays = {key: fixture[key] for key in fixture.files}
    metadata = json.loads(arrays["metadata"].item())
    metadata["image_source"] = "HCC-TACE-Seg HCC_063 contrast CT (same scan in all three channels)"
    torch.set_num_threads(1)
    models = {}
    for suffix, seed in (("runner", 5), ("runner_decoder", 6)):
        model_spec = json.loads(arrays[f"{suffix}::spec"].item())
        models[suffix] = seeded_network(model_spec, np.random.default_rng(seed), scale=0.2)[0]
    inferer = SlidingWindowInferer(roi_size=(8, 8, 8), sw_batch_size=4, overlap=0.25)
    with tempfile.TemporaryDirectory(prefix="segresnet-renal-") as temp:
        for case in metadata["cases"]:
            name = case["case"]
            affine = np.asarray(case["saved"]["affine"])
            paths = {}
            for role in keys:
                path = Path(temp) / f"{name}_{role}.nii.gz"
                nib.save(nib.Nifti1Image(ct, affine), path)
                paths[role] = str(path)
                arrays[f"{name}::{role}"] = np.frombuffer(path.read_bytes(), dtype=np.uint8)
            data = pre(paths)
            image = data["image"].as_tensor()
            arrays[f"{name}::image"] = image.numpy()
            for suffix, model in models.items():
                with torch.inference_mode():
                    logits = inferer(image[None], model)
                arrays[f"{name}::{suffix}_logits"] = logits.numpy()
                post_input = dict(data)
                post_input["pred"] = MetaTensor(logits[0])
                arrays[f"{name}::{suffix}_mask"] = (
                    post(post_input)["pred"].numpy().squeeze().astype(np.float32)
                )
    arrays["metadata"] = np.asarray(json.dumps(metadata))
    destination = ROOT / "work/real-reference/medmlx_renal.npz"
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, **arrays)


if __name__ == "__main__":
    main()
