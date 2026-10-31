"""Inference and NumPy conversion work with framework imports actively forbidden."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import cast

import numpy as np
import pytest
from reference_data import FIXTURES, load_fixture

pytestmark = pytest.mark.mlx


def test_runtime_has_no_reference_framework_dependency(tmp_path: Path) -> None:
    """Run patch and window inference with reference frameworks blocked."""
    code = """
import importlib.abc
import json
import sys
from typing import cast

forbidden = {"torch", "monai", "ignite"}

class BlockFrameworks(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in forbidden:
            raise ModuleNotFoundError(f"Forbidden runtime dependency: {fullname}")

sys.meta_path.insert(0, BlockFrameworks())
import mlx.core as mx
import numpy as np
from medmlx_core.typing import ArrayFactory
from segresnet_mlx import (
    SegResNet, SegResNetSpec, SlidingWindowConfig,
    forward_patch, predict_logits,
)

with np.load(sys.argv[1], allow_pickle=False) as reference:
    spec = json.loads(str(reference["metadata"]))["spec"]
    spec["blocks_down"] = tuple(spec["blocks_down"])
    spec["blocks_up"] = tuple(spec["blocks_up"])
    spec = SegResNetSpec(**spec)
    state = {
        key.removeprefix("weight::"): reference[key]
        for key in reference.files
        if key.startswith("weight::")
    }
    weights = {
        key: cast(ArrayFactory, mx.array)(value)
        for key, value in state.items()
    }
    network = SegResNet(spec, weights, mx=mx)
    output = forward_patch(network, reference["input"])
    equal = np.allclose(output, reference["output"], rtol=0, atol=1e-6)
    stitched = predict_logits(
        reference["input"], SlidingWindowConfig((4, 6, 8)), network=network,
    )
    equal = equal and np.allclose(stitched, reference["output"], rtol=0, atol=1e-6)
    loaded = sorted(name for name in sys.modules if name.split(".")[0] in forbidden)
    print(json.dumps({"logits_equal": equal, "forbidden_modules": loaded}))
"""
    arrays, metadata = load_fixture(FIXTURES / "segresnet_no_decoder.npz")
    from reference_contracts import ArchiveWriter

    fixture = tmp_path / "reference.npz"
    cast(ArchiveWriter, np.savez_compressed)(
        fixture, metadata=np.array(json.dumps(metadata)), **arrays
    )
    result = subprocess.run(
        [sys.executable, "-c", code, str(fixture)],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {"logits_equal": True, "forbidden_modules": []}
