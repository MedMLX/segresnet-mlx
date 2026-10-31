"""Pinned reference-only constructor, transform and archive schemas."""

from __future__ import annotations

from collections.abc import Mapping
from typing import NotRequired, Protocol, TypedDict, cast

import numpy as np
import torch

type HostArray = np.ndarray[tuple[int, ...], np.dtype[np.generic]]
type Float32Array = np.ndarray[tuple[int, ...], np.dtype[np.float32]]
type Float64Array = np.ndarray[tuple[int, ...], np.dtype[np.float64]]
type Json = bool | int | float | str | list[Json] | dict[str, Json] | None


class Spec(TypedDict):
    """SegResNet constructor topology for one seeded reference network.

    ``blocks_down`` and ``blocks_up`` hold one residual-block count per stage,
    ``init_filters`` the first convolution's channel count, and ``num_groups`` the
    group-normalization divisor paired with ``norm_eps``.
    """

    in_channels: int
    out_channels: int
    init_filters: int
    blocks_down: tuple[int, ...]
    blocks_up: tuple[int, ...]
    num_groups: int
    norm_eps: float
    upsample_mode: str


class NetworkDefinition(TypedDict):
    """``network_def`` section of a bundle inference config: channels and decoder mode."""

    in_channels: int
    out_channels: int
    upsample_mode: NotRequired[str]


class Inferer(TypedDict):
    """``inferer`` section of a bundle config, resolved into ``SlidingWindowInferer`` arguments.

    ``roi_size`` is the window shape in voxels and ``overlap`` the fractional
    overlap between neighbouring windows. ``mode`` and ``padding_mode`` select the
    blending and padding schemes and default to MONAI's ``constant``.
    """

    _target_: NotRequired[str]
    roi_size: list[int]
    sw_batch_size: int
    overlap: float
    mode: NotRequired[str]
    padding_mode: NotRequired[str]


class Config(TypedDict):
    """Inference sections pinned in ``tests/fixtures/bundle_configs`` for one bundle.

    ``network_def`` and ``inferer`` drive the recordings; ``preprocessing`` and
    ``postprocessing`` hold serialized MONAI transform configurations.
    """

    network_def: NetworkDefinition
    inferer: Inferer
    preprocessing: dict[str, Json]
    postprocessing: dict[str, Json]


class Pins(TypedDict):
    """Contents of ``scripts/upstream_reference.json``: revisions and required upstream files."""

    monai_version: str
    monai_revision: str
    model_zoo_revision: str
    fixture_schema_version: int
    files: list[str]


class Metadata(TypedDict, total=False):
    """Provenance recorded beside the arrays in each reference ``.npz``.

    ``spec`` and ``inferer`` belong to network and sliding-window cases, ``kind``
    names the predictor variant, and ``cases`` lists recorded affine cases or
    plain reader case names. The recorder merges pins and Torch details into it.
    """

    spec: Spec
    seed: int
    inferer: Inferer
    kind: str
    model_id: str
    cases: list[Case] | tuple[str, ...]


class SavedFacts(TypedDict):
    """Facts read back from a saved prediction NIfTI.

    ``affine`` is the 4x4 voxel-to-world matrix in millimetres and the form codes
    are the header's sform/qform codes. ``data`` carries the voxels until the
    recorder stores them under a separate array key.
    """

    dtype: str
    shape: list[int]
    qform_code: int
    sform_code: int
    affine: list[list[float]]
    data: NotRequired[HostArray]


class Case(TypedDict):
    """One recorded input-affine case and the facts of its saved prediction."""

    case: str
    saved: SavedFacts


class TensorModule(Protocol):
    """Structural type for a Torch module the recorder seeds or probes."""

    def __call__(self, values: torch.Tensor) -> torch.Tensor:
        """Apply a reference module to a tensor."""
        ...
    def state_dict(self) -> Mapping[str, torch.Tensor]:
        """Return the reference module parameters by name."""
        ...


class TensorSequence(TensorModule, Protocol):
    """Structural type for an indexable module container such as ``ModuleList``."""

    def __len__(self) -> int:
        """Return the number of modules in the sequence."""
        ...
    def __getitem__(self, index: int) -> TensorSequence:
        """Return the module sequence at an index."""
        ...


class Network(TensorModule, Protocol):
    """Structural type for MONAI SegResNet, walked in forward order to seed parameters."""

    convInit: TensorModule
    down_layers: TensorSequence
    up_samples: TensorSequence
    up_layers: TensorSequence
    conv_final: TensorModule

    def load_state_dict(self, state: Mapping[str, torch.Tensor], strict: bool) -> object:
        """Load reference parameters, optionally requiring exact keys."""
        ...


class Transform(Protocol):
    """Structural type for a MONAI transform applied to a data mapping."""

    def __call__(self, data: dict[str, object]) -> dict[str, object]:
        """Apply a reference transform to its data mapping."""
        ...


class TransformSequence(Transform, Protocol):
    """Structural type for a composed transform exposing its ``transforms`` list."""

    transforms: list[Transform]


class Parser(Protocol):
    """Structural type for the MONAI ``ConfigParser`` calls used by the recorders."""

    def get_parsed_content(self, key: str) -> TransformSequence:
        """Build a transform sequence from a named bundle section."""
        ...
    def __setitem__(self, key: str, value: object) -> None:
        """Set a bundle configuration value."""
        ...
    def read_config(self, path: str) -> None:
        """Read a bundle configuration from a path."""
        ...


class TorchArrays(Protocol):
    """Structural type for the ``torch`` entry point that wraps recorded arrays."""

    def from_numpy(self, values: HostArray) -> torch.Tensor:
        """Expose a NumPy array as a Torch tensor."""
        ...


from_numpy = cast(TorchArrays, torch).from_numpy


class ArchiveWriter(Protocol):
    """Structural type for the named-array signature of ``numpy.savez_compressed``."""

    def __call__(self, file: object, **arrays: HostArray) -> None:
        """Write named arrays to a reference archive."""
        ...


class Reader(Protocol):
    """Structural type for the MONAI ``NibabelReader`` calls used to record NIfTI behaviour."""

    def read(self, data: str) -> object:
        """Read an image through the reference reader."""
        ...
    def get_data(self, img: object) -> tuple[HostArray, dict[str, HostArray]]:
        """Return image voxels and metadata from the reference reader."""
        ...
