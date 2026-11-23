"""Strict safetensors loading for MONAI SegResNet MLX inference."""

from __future__ import annotations

from pathlib import Path

from medmlx_core.weights import load_tensors

from segresnet_mlx._runtime import import_mlx
from segresnet_mlx.medmlx import MODEL_ID
from segresnet_mlx.network import SegResNet
from segresnet_mlx.topology import Precision, SegResNetSpec


def required_shapes(spec: SegResNetSpec) -> dict[str, tuple[int, ...]]:
    """Return the exact ``{parameter name: shape}`` set a checkpoint must provide.

    Names follow MONAI's ``SegResNet`` state dict and shapes keep the Torch layout:
    convolution weights are ``(out, in, kD, kH, kW)``, transposed-convolution
    weights ``(in, out, 2, 2, 2)`` (``in == out`` here) and GroupNorm affine
    parameters ``(channels,)``. Dropout has no parameters.
    """
    filters = spec.init_filters
    shapes: dict[str, tuple[int, ...]] = {
        "convInit.conv.weight": (filters, spec.in_channels, 3, 3, 3),
    }
    for stage, block_count in enumerate(spec.blocks_down):
        channels = filters * 2**stage
        if stage > 0:
            shapes[f"down_layers.{stage}.0.conv.weight"] = (
                channels,
                channels // 2,
                3,
                3,
                3,
            )
        for block in range(block_count):
            prefix = f"down_layers.{stage}.{block + 1}"
            shapes.update(_residual_shapes(prefix, channels))
    for stage, block_count in enumerate(spec.blocks_up):
        in_channels = filters * 2 ** (len(spec.blocks_up) - stage)
        out_channels = in_channels // 2
        shapes[f"up_samples.{stage}.0.conv.weight"] = (
            out_channels,
            in_channels,
            1,
            1,
            1,
        )
        if spec.upsample_mode == "deconv":
            prefix = f"up_samples.{stage}.1.deconv"
            shapes[f"{prefix}.weight"] = (out_channels, out_channels, 2, 2, 2)
            shapes[f"{prefix}.bias"] = (out_channels,)
        for block in range(block_count):
            shapes.update(_residual_shapes(f"up_layers.{stage}.{block}", out_channels))
    shapes["conv_final.0.weight"] = (filters,)
    shapes["conv_final.0.bias"] = (filters,)
    shapes["conv_final.2.conv.weight"] = (spec.out_channels, filters, 1, 1, 1)
    shapes["conv_final.2.conv.bias"] = (spec.out_channels,)
    return shapes


def load_network(
    weights_path: Path,
    *,
    spec: SegResNetSpec,
    precision: Precision = "source-fp32",
) -> SegResNet:
    """Load converted float32 safetensors into a ``SegResNet`` for ``spec``.

    Raises:
        AssetNotReadyError: If the file cannot be read as safetensors, or its
            tensor names, shapes or dtypes differ from ``required_shapes(spec)``,
            or a parameter is not finite float32.
    """
    weights = load_tensors(
        weights_path,
        model_id=MODEL_ID,
        label="SegResNet",
        shapes=required_shapes(spec),
    )
    return SegResNet(spec, weights, mx=import_mlx(), precision=precision)


def _residual_shapes(prefix: str, channels: int) -> dict[str, tuple[int, ...]]:
    """Expected parameter shapes of one residual block (Torch layouts, no conv bias)."""
    return {
        f"{prefix}.norm1.weight": (channels,),
        f"{prefix}.norm1.bias": (channels,),
        f"{prefix}.norm2.weight": (channels,),
        f"{prefix}.norm2.bias": (channels,),
        f"{prefix}.conv1.conv.weight": (channels, channels, 3, 3, 3),
        f"{prefix}.conv2.conv.weight": (channels, channels, 3, 3, 3),
    }
