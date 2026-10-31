"""Read upstream outputs and regenerate seeded reference parameters when needed."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import cast

import numpy as np
from fixture_contracts import (
    Arrays,
    BundleMetadata,
    ConstructorSpec,
    InputPaths,
    Metadata,
    ModelFixture,
    ModelMetadata,
    NumericArrays,
    Pins,
    SavedFacts,
    SpecRecord,
)

from segresnet_mlx.topology import SegResNetSpec

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
PINS = cast(Pins, json.loads((ROOT / "scripts/upstream_reference.json").read_text()))
FIXTURES = ROOT / "tests/fixtures/darwin-arm64"


def load_fixture(path: Path) -> tuple[Arrays, Metadata]:
    """Read a safe NumPy archive and check its recorded provenance."""
    assert path.is_file(), f"Regenerate the macOS arm64 upstream fixtures: {path}"
    with np.load(path, allow_pickle=False) as fixture:
        arrays = {key: fixture[key] for key in fixture.files if key != "metadata"}
        metadata = cast(Metadata, json.loads(str(fixture["metadata"])))
    assert metadata["schema_version"] == PINS["fixture_schema_version"]
    assert metadata["reference"] == "MONAI"
    assert metadata["recording_platform"] == FIXTURES.name
    for key in ("monai_version", "monai_revision", "model_zoo_revision"):
        assert metadata[key] == PINS[key]
    _restore_parameters(path, arrays, metadata)
    return arrays, metadata


def _restore_parameters(path: Path, arrays: Arrays, metadata: Metadata) -> None:
    """Rebuild the recorded inputs to MONAI from their seeds and topology."""
    name = path.stem
    kind = metadata.get("kind")
    if not (name.startswith("segresnet_") or kind in ("decoder_network", "no_decoder_network")
            or (name.startswith("medmlx_") and "runner::spec" in arrays)):
        return
    from reference_contracts import Spec
    from reference_support import seeded_network

    if name.startswith("segresnet_"):
        seed, spec = metadata.get("seed"), metadata.get("spec")
        assert seed is not None and spec is not None
        rng = np.random.default_rng(seed)
        _, state = seeded_network(cast(Spec, spec), rng, dtype=np.float64)
        arrays.update({f"source::{key}": value for key, value in state.items()})
        arrays.update({f"weight::{key}": value.astype(np.float32) for key, value in state.items()})
    elif kind in ("decoder_network", "no_decoder_network"):
        seed, spec = metadata.get("seed"), metadata.get("spec")
        assert seed is not None and spec is not None
        rng = np.random.default_rng(seed)
        rng.normal(size=arrays["input"].shape)
        _, state = seeded_network(cast(Spec, spec), rng)
        arrays.update({f"weight::{key}": value for key, value in state.items()})
    else:
        for prefix, seed in (("runner", 5), ("runner_decoder", 6)):
            spec = cast(Spec, json.loads(str(arrays[f"{prefix}::spec"])))
            _, state = seeded_network(spec, np.random.default_rng(seed), scale=0.2)
            arrays.update({f"{prefix}::weight::{key}": value for key, value in state.items()})


def model_fixture(path: Path) -> ModelFixture:
    """Restore the topology and arrays from a network fixture."""
    arrays, metadata = load_fixture(path)
    metadata = cast(ModelMetadata, metadata)
    spec: SpecRecord = metadata["spec"].copy()
    spec["blocks_down"] = tuple(spec["blocks_down"])
    spec["blocks_up"] = tuple(spec["blocks_up"])
    return SegResNetSpec(**cast("ConstructorSpec", spec)), cast(NumericArrays, arrays), metadata


BUNDLE_INPUT_KEYS = {
    "renal": {"arterial_path": "artery", "venous_path": "vein", "excretory_path": "excret"},
    "brats": {
        "t1c_path": "channel0",
        "t1_path": "channel1",
        "t2_path": "channel2",
        "flair_path": "channel3",
    },
    "wholebody": {"image_path": "ct"},
    "wholebody_lowres": {"image_path": "ct"},
}


def bundle_fixture(name: str) -> tuple[Arrays, dict[str, SavedFacts]]:
    """Return bundle arrays and saved-image metadata by case."""
    arrays, metadata = load_fixture(FIXTURES / f"medmlx_{name}.npz")
    metadata = cast(BundleMetadata, metadata)
    return arrays, {case["case"]: case["saved"] for case in metadata["cases"]}


def write_inputs(tmp_path: Path, name: str, case: str) -> InputPaths:
    """Write recorded NIfTI inputs and return paths keyed by runner role."""
    arrays, _ = bundle_fixture(name)
    paths: dict[str, str] = {}
    for role, key in BUNDLE_INPUT_KEYS[name].items():
        path = tmp_path / f"{case}_{key}.nii.gz"
        path.write_bytes(arrays[f"{case}::{key}"].tobytes())
        paths[role] = str(path)
    return cast(InputPaths, paths)
