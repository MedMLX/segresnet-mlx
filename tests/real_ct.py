"""Source-aligned HCC-TACE inputs for released-network numerical tests."""

from pathlib import Path

import numpy as np
from medmlx_core.volumes import read_volume

from segresnet_mlx._arrays import Float32Array


def renal_input(seed: int, *, sparse: bool) -> Float32Array:
    """Build three co-registered channels from CT; sparse cases retain liver voxels."""
    root = Path.home() / ".cache/medmlx/test-data"
    ct = read_volume(root / "hcc_063_ct.nii.gz")
    liver = read_volume(root / "hcc_063_liver.nii.gz", dtype=None)
    if ct.data.shape != liver.data.shape or not np.allclose(ct.affine, liver.affine):
        raise ValueError("HCC-TACE CT and liver grids differ")
    size = 96 if sparse else 8
    start = 100 if seed == 23 else 108
    slices = (slice(start, start + size), slice(180, 180 + size), slice(2, 2 + size))
    voxels = ct.data[slices]
    image = np.clip((voxels + np.float32(1000)) / np.float32(2000), 0, 1)
    if sparse:
        mask = liver.data[slices] > 0
        if not np.any(mask):
            raise ValueError("HCC-TACE crop contains no liver")
        image = np.where(mask, image, np.float32(0))
    return np.broadcast_to(image, (1, 3, *image.shape)).copy()
