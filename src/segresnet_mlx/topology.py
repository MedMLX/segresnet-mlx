"""Frozen topology contract for the MONAI SegResNet MLX implementation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

# Accumulation scheme for convolution and GroupNorm; see docs/technical.md for the three
# profiles. ``compensated-fp48`` additionally requires transpose-convolution upsampling.
type ArithmeticProfile = Literal["torch-order-fp32", "compensated-fp32", "compensated-fp48"]


@dataclass(frozen=True, slots=True)
class SegResNetSpec:
    """The inference-relevant MONAI ``SegResNet`` constructor arguments.

    ``blocks_down`` and ``blocks_up`` count residual blocks per encoder and decoder
    stage (one more encoder stage than decoder stage). ``init_filters`` is the channel
    count of the first stage and doubles at each deeper stage. ``upsample_mode`` is
    ``"nontrainable"`` (1x1x1 convolution plus trilinear 2x interpolation) or
    ``"deconv"`` (1x1x1 convolution plus 2x transposed convolution). Dropout is not
    part of the spec because it is inactive at inference.
    """

    in_channels: int
    out_channels: int
    init_filters: int
    blocks_down: tuple[int, ...] = (1, 2, 2, 4)
    blocks_up: tuple[int, ...] = (1, 1, 1)
    num_groups: int = 8
    norm_eps: float = 1e-5
    upsample_mode: str = "nontrainable"

    def __post_init__(self) -> None:
        """Reject unsupported channel, stage, normalization, and upsampling settings."""
        if min(self.in_channels, self.out_channels, self.init_filters) < 1:
            raise ValueError("SegResNet channel counts must be positive")
        if len(self.blocks_down) != len(self.blocks_up) + 1:
            raise ValueError("SegResNet needs one more down stage than up stage")
        if not self.blocks_down or any(count < 1 for count in self.blocks_down):
            raise ValueError("SegResNet down stages must contain residual blocks")
        if any(count < 1 for count in self.blocks_up):
            raise ValueError("SegResNet up stages must contain residual blocks")
        if self.num_groups < 1 or not math.isfinite(self.norm_eps) or self.norm_eps <= 0:
            raise ValueError("SegResNet normalization parameters must be positive")
        if self.init_filters % self.num_groups != 0:
            raise ValueError("SegResNet init_filters must be divisible by num_groups")
        if self.upsample_mode not in {"nontrainable", "deconv"}:
            raise ValueError("SegResNet MLX supports nontrainable or deconv upsampling")
