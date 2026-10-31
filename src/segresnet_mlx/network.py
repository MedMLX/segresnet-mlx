"""MONAI 1.6 SegResNet inference graph in MLX, using NCDHW tensors."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from medmlx_core.compensated import CompensatedFloat32Operators, CompensatedFloat48Operators

    from segresnet_mlx._types import MLX, Array

from medmlx_core.layout import require_ncdhw
from medmlx_core.ops import conv_transpose3d_ncdhw, upsample_trilinear_ncdhw
from medmlx_core.precision import Float32Operators

from segresnet_mlx.topology import ArithmeticProfile, SegResNetSpec


def group_mean(values: Array, mx: MLX) -> Array:
    """Mean over the last axis of ``(N, G, S)`` values, summed in 256-element tiles.

    Tiling bounds the float32 accumulation depth of a group that spans a whole
    volume. The last dimension must be nonempty. Returns ``(N, G, 1)``. ``SegResNet``
    does not call this (its GroupNorm comes from the ``medmlx_core`` operators); it is
    exercised by ``tests/test_equivalence.py``.
    """
    size = int(values.shape[-1])
    padded = mx.pad(values, [(0, 0), (0, 0), (0, (-size) % 256)])
    tiles = mx.reshape(padded, (*values.shape[:2], -1, 256))
    return mx.sum(mx.sum(tiles, axis=-1), axis=-1, keepdims=True) / size


class SegResNet:
    """Raw-logit forward pass for MONAI's 3-D, group-normalized SegResNet.

    Tensors are channels-first ``(N, C, D, H, W)``; the supported inference
    contract uses float32 inputs and returned logits. This class checks rank,
    channel count and spatial divisibility but does not itself check input dtype
    or finiteness; the public patch and sliding-window helpers do.
    ``weights`` uses MONAI's state-dict keys and Torch parameter layouts
    (convolution weights are ``(out, in, kD, kH, kW)``); ``required_shapes`` in
    ``segresnet_mlx.weights`` lists them. Inference only: dropout is omitted and
    no activation is applied to the output.

    ``arithmetic_profile`` selects the accumulation scheme (see ``docs/technical.md``):
    ``"torch-order-fp32"`` (default), ``"compensated-fp32"``, or
    ``"compensated-fp48"``, which needs ``upsample_mode="deconv"`` and raises
    ``ValueError`` otherwise. The first two profiles use float32 layer boundaries;
    ``compensated-fp48`` carries paired float32 values through the hidden layers.
    """

    def __init__(
        self,
        spec: SegResNetSpec,
        weights: Mapping[str, Array],
        *,
        mx: MLX,
        arithmetic_profile: ArithmeticProfile = "torch-order-fp32",
    ) -> None:
        """Bind the topology and weights and select their arithmetic operators."""
        self.spec = spec
        self._weights = weights
        self._mx = mx
        self.arithmetic_profile = arithmetic_profile
        # Both convolution and normalization accumulation order affect sparse
        # released-checkpoint logits; correcting only one leaves amplified drift.
        if arithmetic_profile == "torch-order-fp32":
            self._operators = Float32Operators(mx, spec.num_groups, spec.norm_eps)
        elif arithmetic_profile == "compensated-fp32":
            from medmlx_core.compensated import CompensatedFloat32Operators as AccuracyOperators

            self._operators = AccuracyOperators(mx, spec.num_groups, spec.norm_eps)
        elif arithmetic_profile == "compensated-fp48":
            if spec.upsample_mode != "deconv":
                raise ValueError("Paired precision requires transpose-convolution upsampling")
            from medmlx_core.compensated import CompensatedFloat48Operators as PairedOperators

            self._operators = PairedOperators(mx, spec.num_groups, spec.norm_eps)
        else:
            raise ValueError(f"Unknown SegResNet arithmetic profile {arithmetic_profile!r}")

    def __call__(self, values: Array) -> Array:
        """Return float32 logits ``(N, out_channels, D, H, W)`` for an ``(N, C, D, H, W)`` input.

        Raises:
            ValueError: If the input is not rank 5, has the wrong channel count, or a
                spatial size is not divisible by ``2 ** len(spec.blocks_up)``.
        """
        if self.arithmetic_profile == "compensated-fp48":
            operators = cast("CompensatedFloat48Operators", self._operators)
            return operators.round(self.paired_forward(values))
        return self._forward(values)

    def paired_forward(self, values: Array) -> Array:
        """Run the paired-precision graph without the final rounding.

        Returns ``(2N, out_channels, D, H, W)``: the high words of the ``N`` logit
        volumes followed by their low words. Their float64 sum is the logit; the
        caller (``predict_wide_logits``) does that sum on the host.
        """
        if self.arithmetic_profile != "compensated-fp48":
            raise ValueError("Paired logits require the compensated-fp48 profile")
        operators = cast("CompensatedFloat48Operators", self._operators)
        return self._forward(operators.pack(values))

    def _forward(self, values: Array) -> Array:
        """Run the encoder and decoder, retaining skip connections between stages."""
        require_ncdhw(values, name="SegResNet input")
        if int(values.shape[1]) != self.spec.in_channels:
            raise ValueError(
                f"SegResNet input has {values.shape[1]} channels; expected {self.spec.in_channels}"
            )

        divisor = 2 ** len(self.spec.blocks_up)
        if any(int(size) % divisor for size in values.shape[2:]):
            raise ValueError(f"SegResNet spatial sizes must be divisible by {divisor}")
        hidden = self.conv(values, "convInit.conv", padding=1)
        skips: list[Array] = []
        for stage, block_count in enumerate(self.spec.blocks_down):
            if stage > 0:
                hidden = self.conv(hidden, f"down_layers.{stage}.0.conv", padding=1, stride=2)
            block_offset = 1
            for block in range(block_count):
                hidden = self.residual(hidden, f"down_layers.{stage}.{block + block_offset}")
            skips.append(hidden)

        for stage, block_count in enumerate(self.spec.blocks_up):
            hidden = self.conv(hidden, f"up_samples.{stage}.0.conv", padding=0)
            if self.spec.upsample_mode == "deconv":
                prefix = f"up_samples.{stage}.1.deconv"
                if self.arithmetic_profile != "torch-order-fp32":
                    hidden = cast("CompensatedFloat32Operators", self._operators).conv_transpose3d(
                        hidden, self._weights[f"{prefix}.weight"], self._weights[f"{prefix}.bias"]
                    )
                else:
                    hidden = conv_transpose3d_ncdhw(
                        hidden,
                        self._weights[f"{prefix}.weight"],
                        self._weights[f"{prefix}.bias"],
                        padding=0,
                        output_padding=0,
                        stride=2,
                        mx=self._mx,
                    )
                hidden = self._add(hidden, skips[-(stage + 2)])
            else:
                hidden = upsample_trilinear_ncdhw(hidden, mx=self._mx) + skips[-(stage + 2)]
            for block in range(block_count):
                hidden = self.residual(hidden, f"up_layers.{stage}.{block}")

        hidden = self._norm(hidden, "conv_final.0")
        hidden = self._relu(hidden)
        return self.conv(hidden, "conv_final.2.conv", padding=0, bias=True)

    def residual(self, values: Array, prefix: str) -> Array:
        """Pre-activation residual block: (norm, ReLU, 3x3x3 conv) twice, plus the input."""
        hidden = self._norm(values, f"{prefix}.norm1")
        hidden = self._relu(hidden)
        hidden = self.conv(hidden, f"{prefix}.conv1.conv", padding=1)
        hidden = self._norm(hidden, f"{prefix}.norm2")
        hidden = self._relu(hidden)
        hidden = self.conv(hidden, f"{prefix}.conv2.conv", padding=1)
        return self._add(hidden, values)

    def _relu(self, values: Array) -> Array:
        """Apply ReLU using paired arithmetic when the fp48 profile is active."""
        # fp48 activations are paired float32 words, so ReLU must act on the pair.
        return (
            cast("CompensatedFloat48Operators", self._operators).relu(values)
            if self.arithmetic_profile == "compensated-fp48"
            else self._mx.maximum(values, 0.0)
        )

    def _add(self, values: Array, other: Array) -> Array:
        """Add residual or skip values using the selected arithmetic profile."""
        # Residual/skip add; for fp48 it is a paired-precision add, not a plain float32 sum.
        return (
            cast("CompensatedFloat48Operators", self._operators).add(values, other)
            if self.arithmetic_profile == "compensated-fp48"
            else values + other
        )

    def _norm(self, values: Array, prefix: str) -> Array:
        """Apply group normalization using the weights at the given prefix."""
        return self._operators.norm(
            values, self._weights[f"{prefix}.weight"], self._weights[f"{prefix}.bias"]
        )

    def conv(
        self,
        values: Array,
        prefix: str,
        *,
        padding: int,
        stride: int = 1,
        bias: bool = False,
    ) -> Array:
        """Convolve with ``{prefix}.weight`` (plus ``{prefix}.bias`` if ``bias``)."""
        return self._operators.conv(
            values,
            self._weights[f"{prefix}.weight"],
            self._weights.get(f"{prefix}.bias") if bias else None,
            padding=padding,
            stride=stride,
        )
