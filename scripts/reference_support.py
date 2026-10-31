"""Pinned upstream reference inputs and recording boundary; never imported by tests."""

from __future__ import annotations

import json
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import monai
import numpy as np
import torch
from monai.networks.nets.segresnet import SegResNet
from reference_contracts import (
    ArchiveWriter,
    Config,
    HostArray,
    Metadata,
    Network,
    Pins,
    Spec,
    from_numpy,
)

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True, slots=True)
class Reference:
    """A verified MONAI installation and immutable model-zoo file set."""

    pins: Pins
    bundle_root: Path
    destination: Path

    @classmethod
    def load(cls, bundle_root: Path) -> Reference:
        """Validate reference pins and upstream files before recording on macOS arm64."""
        if (platform.system(), platform.machine()) != ("Darwin", "arm64"):
            raise RuntimeError("Record references on macOS arm64 with the pinned upstream stack")
        pins = cast(Pins, json.loads((ROOT / "scripts/upstream_reference.json").read_text()))
        if (monai.__version__, monai.__revision_id__) != (
            pins["monai_version"],
            pins["monai_revision"],
        ):
            raise RuntimeError(
                "Historical reference pins differ from the patched dev environment. "
                "Migrate the pins and regenerate fixtures together; do not downgrade MONAI."
            )
        for relative in pins["files"]:
            path = bundle_root / relative
            if not path.is_file():
                raise RuntimeError(f"Missing upstream file: {path}")
        destination = ROOT / "tests/fixtures/darwin-arm64"
        torch.set_num_threads(1)
        return cls(pins, bundle_root.resolve(), destination)

    def config(self, name: str) -> Config:
        """Load a bundle config and verify its recorded sections."""
        config = cast(
            Config, json.loads((self.bundle_root / name / "configs/inference.json").read_text())
        )
        sections = {
            key: config[key]
            for key in ("network_def", "inferer", "preprocessing", "postprocessing")
        }
        recorded = json.loads((ROOT / "tests/fixtures/bundle_configs" / f"{name}.json").read_text())
        if recorded != sections:
            raise RuntimeError(f"Committed bundle sections differ from pinned upstream: {name}")
        return config

    def save(self, name: str, arrays: dict[str, HostArray], metadata: Metadata) -> None:
        """Write a size-bounded reference archive with upstream provenance."""
        provenance: dict[str, object] = {
            key: self.pins[key] for key in ("monai_version", "monai_revision", "model_zoo_revision")
        }
        provenance.update(
            schema_version=self.pins["fixture_schema_version"],
            reference="MONAI",
            torch_version=torch.__version__,
            reference_backend="torch-cpu",
            torch_threads=1,
            recording_platform=self.destination.name,
        )
        self.destination.mkdir(parents=True, exist_ok=True)
        destination = self.destination / f"{name}.npz"
        cast(ArchiveWriter, np.savez_compressed)(
            destination,
            **arrays,
            metadata=np.array(json.dumps(metadata | provenance, sort_keys=True)),
        )
        if destination.stat().st_size >= 1_000_000:
            raise RuntimeError(f"Reference fixture is too large: {destination}")
        print(f"{destination.name}: {destination.stat().st_size} bytes")


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
