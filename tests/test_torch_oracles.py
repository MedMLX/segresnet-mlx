"""Released renal SegResNet parity against the independent MONAI graph.

Set MEDMLX_TEST_CHECKPOINT_ROOT to the staged MONAI model-zoo bundle directory
and MEDMLX_TEST_RENAL_WEIGHTS to the released native safetensors file.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import cast

import numpy as np
import pytest
from fixture_contracts import from_numpy
from medmlx_core.typing import ArrayFactory

from segresnet_mlx._arrays import as_numpy
from segresnet_mlx.topology import ArithmeticProfile, SegResNetSpec
from segresnet_mlx.weights import load_model

pytestmark = pytest.mark.mlx

_RAW_LOGIT_RTOL = 1e-4
_RAW_LOGIT_ATOL = 1e-4


@pytest.mark.parametrize("sparse_volume", [False, True], ids=["dense", "sparse-accumulation"])
@pytest.mark.parametrize("seed", [23, 47])
@pytest.mark.parametrize(
    "arithmetic_profile", ["torch-order-fp32", "compensated-fp32", "compensated-fp48"]
)
def test_renal_segresnet_checkpoint_matches_monai_raw_logits(
    sparse_volume: bool, seed: int, arithmetic_profile: ArithmeticProfile
) -> None:
    """Compare released renal logits and signs with the original MONAI graph."""
    checkpoint_root = os.environ.get("MEDMLX_TEST_CHECKPOINT_ROOT")
    weights_file = os.environ.get("MEDMLX_TEST_RENAL_WEIGHTS")
    if checkpoint_root is None or weights_file is None:
        pytest.skip("set MEDMLX_TEST_CHECKPOINT_ROOT and MEDMLX_TEST_RENAL_WEIGHTS")
    checkpoint = Path(checkpoint_root) / "renalStructures_CECT_segmentation/models/model.pt"
    assert checkpoint.is_file(), f"SegResNet checkpoint is missing: {checkpoint}"
    pytest.importorskip("mlx.core")
    from segresnet_mlx._runtime import import_mlx

    mx = import_mlx()
    if not mx.metal.is_available():
        pytest.skip("SegResNet parity requires Apple Silicon Metal")
    mx.set_default_device(mx.gpu)
    pytest.importorskip("torch")
    import torch

    pytest.importorskip("monai.networks.nets")
    from monai.networks.nets.segresnet import SegResNet

    spec = SegResNetSpec(3, 6, 32, upsample_mode="deconv")
    source = SegResNet(
        init_filters=spec.init_filters,
        in_channels=spec.in_channels,
        out_channels=spec.out_channels,
        blocks_down=spec.blocks_down,
        blocks_up=spec.blocks_up,
        dropout_prob=0.2,
        upsample_mode=spec.upsample_mode,
    ).eval()
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state = state.get("model", state)
    source.load_state_dict(state, strict=True)
    if arithmetic_profile == "compensated-fp48":
        source.double()
    if arithmetic_profile == "compensated-fp32":
        # Independent original Torch operators accumulate in float64, then
        # round at each convolution/GroupNorm boundary just as the profile states.
        def promote(
            module: torch.nn.Module, arguments: tuple[torch.Tensor, ...]
        ) -> tuple[torch.Tensor, ...]:
            """Promote layer inputs for the float64 accumulation reference."""
            return (arguments[0].double(),)

        def round_output(
            module: torch.nn.Module, arguments: tuple[torch.Tensor, ...], output: torch.Tensor
        ) -> torch.Tensor:
            """Round each reference layer output back to float32."""
            return output.float()

        for module in source.modules():
            if isinstance(module, (torch.nn.Conv3d, torch.nn.ConvTranspose3d, torch.nn.GroupNorm)):
                module.double()
                module.register_forward_pre_hook(promote)
                module.register_forward_hook(round_output)

    rng = np.random.default_rng(seed)
    if sparse_volume:
        # Long sparse groups exposed reduction drift missed by the small controls.
        size = 96
        values = np.zeros((1, spec.in_channels, size, size, size), dtype=np.float32)
        values[:, :, 24:40, 24:40, 48:80] = rng.normal(
            size=(1, spec.in_channels, 16, 16, 32)
        ).astype(np.float32)
    else:
        values = rng.normal(size=(1, spec.in_channels, 8, 8, 8)).astype(np.float32)
    with torch.inference_mode():
        source_input = from_numpy(values)
        if arithmetic_profile == "compensated-fp48":
            source_input = source_input.double()
        expected = source(source_input).float().numpy()

    candidate = load_model(Path(weights_file), spec=spec, arithmetic_profile=arithmetic_profile)
    actual = candidate(cast(ArrayFactory, mx.array)(values))
    mx.eval(actual)
    host = as_numpy(actual)

    assert host.shape == expected.shape
    assert host.dtype == np.float32
    assert np.isfinite(host).all()
    np.testing.assert_allclose(host, expected, rtol=_RAW_LOGIT_RTOL, atol=_RAW_LOGIT_ATOL)

    assert np.array_equal(host >= 0, expected >= 0)
