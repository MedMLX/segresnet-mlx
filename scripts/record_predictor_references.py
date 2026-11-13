"""Record MONAI sliding-window outputs on HCC-TACE CT crops."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from monai import __version__ as monai_version
from monai.inferers import SlidingWindowInferer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "src"))
from reference_contracts import seeded_network  # noqa: E402


def main() -> None:
    """Record source window batches and logits for real multi-channel CT crops."""
    if monai_version != "1.6.0" or not torch.__version__.startswith("2.14."):
        raise ValueError("MONAI 1.6.0 and Torch 2.14 are required")
    ct = np.asarray(
        nib.load(Path.home() / ".cache/medmlx/test-data/hcc_063_ct.nii.gz").dataobj,
        dtype=np.float32,
    )
    out = ROOT / "work/real-reference"
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    for path in (ROOT / "tests/fixtures/darwin-arm64").glob("predictor_*.npz"):
        with np.load(path, allow_pickle=False) as source:
            arrays = {key: source[key] for key in source.files}
        metadata = json.loads(arrays["metadata"].item())
        shape = tuple(int(x) for x in arrays["input_shape"])
        batch, channels, width, height, depth = shape
        samples = []
        for index in range(batch):
            volumes = []
            for channel in range(channels):
                z = 55 + index * 8 + channel * 2
                crop = ct[100 : 100 + width, 210 : 210 + height, z : z + depth]
                volumes.append(np.clip((crop + np.float32(175)) / np.float32(425), 0, 1))
            samples.append(np.stack(volumes))
        image = np.stack(samples).astype(np.float32)
        if "spec" in metadata:
            rng = np.random.default_rng(metadata["seed"])
            rng.normal(size=shape)
            network, _ = seeded_network(metadata["spec"], rng)
        else:
            coefficients = arrays["coefficients"]

            def network(values, metadata=metadata, channels=channels, coefficients=coefficients):
                origin = (
                    values.mean(dim=(2, 3, 4), keepdim=True)
                    if metadata["kind"] == "standin_reduction"
                    else values[:, :, :1, :1, :1]
                )
                return torch.cat(
                    [
                        (
                            values[:, i % channels : i % channels + 1]
                            + origin[:, i % channels : i % channels + 1]
                        )
                        * float(scale)
                        + float(bias)
                        for i, (scale, bias) in enumerate(coefficients)
                    ],
                    dim=1,
                )

        batches = []

        def measured(values, batches=batches, network=network):
            batches.append(list(values.shape))
            return network(values)

        settings = metadata["inferer"]
        inferer = SlidingWindowInferer(
            roi_size=settings["roi_size"],
            sw_batch_size=settings["sw_batch_size"],
            overlap=settings["overlap"],
        )
        with torch.inference_mode():
            result = inferer(torch.from_numpy(image), measured).numpy()
        metadata["image_source"] = "HCC-TACE-Seg HCC_063 contrast CT"
        arrays.update(
            input=image,
            output=result,
            window_batches=np.asarray(batches),
            metadata=np.asarray(json.dumps(metadata)),
        )
        np.savez_compressed(out / path.name, **arrays)


if __name__ == "__main__":
    main()
