"""JSON schemas for the public runner and window telemetry."""

from __future__ import annotations

from collections.abc import Sequence
from typing import NotRequired, TypedDict

from segresnet_mlx.topology import ArithmeticProfile

type Json = bool | int | float | str | list[Json] | dict[str, Json] | None


class InfererConfig(TypedDict):
    roi_size: Sequence[int]
    sw_batch_size: NotRequired[int]
    overlap: NotRequired[float]
    mode: NotRequired[str]
    padding_mode: NotRequired[str]


class BundleConfig(TypedDict):
    inferer: InfererConfig


class Runtime(TypedDict):
    """MLX device string and peak MLX memory in bytes (from ``mx.get_peak_memory``)."""

    device: str
    peak_bytes: int | None


class SlidingWindow(TypedDict):
    roi_size: list[int]
    sw_batch_size: int
    overlap: float
    mode: str
    padding_mode: str


class Seconds(TypedDict):
    """Wall-clock seconds per runner stage, rounded to four decimals."""

    load_inputs_and_weights: float
    preprocess: float
    inference: float
    postprocess_and_write: float


class Details(TypedDict):
    """The ``details`` block of ``run``'s result.

    ``labels`` maps each label value (as a string) to its name and ``labels_present``
    lists the values found in the written mask. ``input_shape`` is the first input's
    voxel shape (equal to the mask shape); ``network_input_shape`` is the spatial shape
    after preprocessing, which the sliding window sees. ``output_dtype`` is the NIfTI
    dtype of the mask. ``peak_mlx_bytes`` is the peak MLX memory in bytes.
    ``requires_human_review`` is always true: outputs are research results.
    """

    weights_revision: NotRequired[str | None]
    bundle: str
    arithmetic_profile: ArithmeticProfile
    streaming: NotRequired[bool]
    labels: dict[str, str]
    labels_present: list[int]
    input_shape: list[int]
    network_input_shape: list[int]
    output_dtype: str
    sliding_window: SlidingWindow
    mlx_device: str
    peak_mlx_bytes: int | None
    seconds: Seconds
    requires_human_review: bool


class RunResult(TypedDict):
    """Result of ``segresnet_mlx.medmlx.run``: ``outputs["mask"]`` is the mask path."""

    outputs: dict[str, str]
    details: Details
