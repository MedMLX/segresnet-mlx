"""Record seeded MONAI SegResNet graphs on HCC-TACE CT crops."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from monai import __version__ as monai_version

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "src"))
from reference_contracts import seeded_network  # noqa: E402


def main() -> None:
    """Run the source model and recorded encoder boundaries with real CT channels."""
    if monai_version != "1.6.0" or not torch.__version__.startswith("2.14."):
        raise ValueError("MONAI 1.6.0 and Torch 2.14 are required")
    ct = np.asarray(
        nib.load(Path.home() / ".cache/medmlx/test-data/hcc_063_ct.nii.gz").dataobj,
        dtype=np.float32,
    )
    out = ROOT / "work/real-reference"
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    for name in ("segresnet_nontrainable", "segresnet_no_decoder", "segresnet_deconv"):
        with np.load(
            ROOT / "tests/fixtures/darwin-arm64" / f"{name}.npz", allow_pickle=False
        ) as data:
            arrays = {key: data[key] for key in data.files}
        metadata = json.loads(arrays["metadata"].item())
        spec = metadata["spec"]
        channels = spec["in_channels"]
        cases = []
        for batch in range(2):
            frames = []
            for channel in range(channels):
                z = 55 + batch * 8 + channel * 2
                volume = ct[100:104, 210:216, z : z + 8]
                frames.append(np.clip((volume + np.float32(175)) / np.float32(425), 0, 1))
            cases.append(np.stack(frames))
        image = np.stack(cases).astype(np.float32)
        network, _ = seeded_network(spec, np.random.default_rng(metadata["seed"]), dtype=np.float64)
        with torch.inference_mode():
            tensor = torch.from_numpy(image)
            result = network(tensor)
            initial = network.convInit(tensor)
            residual = network.down_layers[0][1](network.down_layers[0][0](initial))
            arrays.update(
                input=image,
                output=result.numpy(),
                encoder_initial=initial.numpy(),
                encoder_residual=residual.numpy(),
            )
            if len(spec["blocks_down"]) > 1:
                downsampled = network.down_layers[1][0](residual)
                arrays["encoder_downsampled"] = downsampled.numpy()
                arrays["encoder_output"] = network.down_layers[1][1](downsampled).numpy()
        metadata["image_source"] = "HCC-TACE-Seg HCC_063 contrast CT"
        arrays["metadata"] = np.asarray(json.dumps(metadata))
        np.savez_compressed(out / f"{name}.npz", **arrays)


if __name__ == "__main__":
    main()
