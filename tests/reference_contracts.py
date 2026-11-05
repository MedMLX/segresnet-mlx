"""The seeded MONAI SegResNet whose parameters the network fixtures were recorded with."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol, TypedDict, cast

import numpy as np
import torch
from fixture_contracts import from_numpy
from monai.networks.nets.segresnet import SegResNet

type HostArray = np.ndarray[tuple[int, ...], np.dtype[np.generic]]


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


class TensorModule(Protocol):
    """Structural type for a Torch module seeded for the reference network."""

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


def seeded_network(
    spec: Spec,
    rng: np.random.Generator,
    scale: float = 0.125,
    dtype: type[np.float32] | type[np.float64] = np.float32,
) -> tuple[Network, dict[str, HostArray]]:
    """Seed in forward-module order, deriving every key/shape from upstream modules."""
    network = cast(
        Network,
        SegResNet(
            in_channels=spec["in_channels"],
            out_channels=spec["out_channels"],
            init_filters=spec["init_filters"],
            blocks_down=spec["blocks_down"],
            blocks_up=spec["blocks_up"],
            upsample_mode=spec["upsample_mode"],
            norm=(
                "GROUP",
                {
                    "num_groups": spec["num_groups"],
                    "eps": spec["norm_eps"],
                },
            ),
        ).eval(),
    )
    modules = [("convInit", network.convInit), ("down_layers", network.down_layers)]
    for index in range(len(network.up_samples)):
        modules.extend(
            [
                (f"up_samples.{index}", network.up_samples[index]),
                (f"up_layers.{index}", network.up_layers[index]),
            ]
        )
    modules.append(("conv_final", network.conv_final))
    state: dict[str, HostArray] = {
        f"{prefix}.{key}": rng.normal(0, scale, tuple(value.shape)).astype(dtype)
        for prefix, module in modules
        for key, value in module.state_dict().items()
    }
    network.load_state_dict(
        {key: from_numpy(value.astype(np.float32)) for key, value in state.items()}, strict=True
    )
    return network, state
