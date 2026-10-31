"""Immutable reference archive schemas and typed external tensor boundaries."""

from __future__ import annotations

from typing import NotRequired, Protocol, TypedDict, cast

import torch

from segresnet_mlx._arrays import HostArray, NumericArray
from segresnet_mlx._records import InfererConfig
from segresnet_mlx.topology import SegResNetSpec

type Arrays = dict[str, HostArray]
type NumericArrays = dict[str, NumericArray]


class SpecRecord(TypedDict):
    in_channels: int
    out_channels: int
    init_filters: int
    blocks_down: list[int] | tuple[int, ...]
    blocks_up: list[int] | tuple[int, ...]
    num_groups: int
    norm_eps: float
    upsample_mode: str


class ConstructorSpec(TypedDict):
    in_channels: int
    out_channels: int
    init_filters: int
    blocks_down: tuple[int, ...]
    blocks_up: tuple[int, ...]
    num_groups: int
    norm_eps: float
    upsample_mode: str


class SavedFacts(TypedDict):
    dtype: str
    shape: list[int]
    qform_code: int
    sform_code: int
    affine: list[list[float]]


class Case(TypedDict):
    case: str
    saved: SavedFacts


class Metadata(TypedDict):
    schema_version: int
    reference: str
    recording_platform: str
    monai_version: str
    monai_revision: str
    model_zoo_revision: str
    reference_backend: str
    torch_threads: int
    seed: NotRequired[int]
    spec: NotRequired[SpecRecord]
    inferer: NotRequired[InfererConfig]
    kind: NotRequired[str]
    model_id: NotRequired[str]
    cases: NotRequired[list[Case]]


class Pins(TypedDict):
    fixture_schema_version: int
    monai_version: str
    monai_revision: str
    model_zoo_revision: str
    files: dict[str, str]


class ModelMetadata(TypedDict):
    spec: SpecRecord


class PredictorMetadata(TypedDict):
    inferer: InfererConfig
    model_id: str
    kind: str
    spec: NotRequired[SpecRecord]


class BundleMetadata(TypedDict):
    cases: list[Case]


type ModelFixture = tuple[SegResNetSpec, NumericArrays, ModelMetadata]


# Torch accepts NumPy's native buffer protocol; it does not ship this annotation.
class TorchArrays(Protocol):
    def from_numpy(self, values: HostArray) -> torch.Tensor:
        """Expose a NumPy array as a Torch tensor."""
        ...


from_numpy = cast(TorchArrays, torch).from_numpy


class InputPaths(TypedDict, total=False):
    arterial_path: str
    venous_path: str
    excretory_path: str
    t1c_path: str
    t1_path: str
    t2_path: str
    flair_path: str
    image_path: str


class RunArguments(TypedDict):
    bundle: str
    weights_path: str
    output_dir: str
    arterial_path: NotRequired[str | None]
    venous_path: str | None
    excretory_path: NotRequired[str | None]
    t1c_path: NotRequired[str | None]
    t1_path: NotRequired[str | None]
    t2_path: NotRequired[str | None]
    flair_path: NotRequired[str | None]
    image_path: NotRequired[str | None]


class ArgumentChange(TypedDict, total=False):
    bundle: str
    weights_path: str
    venous_path: str | None
    excretory_path: str | None
    flair_path: str | None


class RenalInputPaths(TypedDict):
    arterial_path: str
    venous_path: str
    excretory_path: str


class IgniteRunner(Protocol):
    def run(self, data: list[int]) -> object:
        """Run the Ignite engine over test input data."""
        ...
