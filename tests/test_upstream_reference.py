"""Provenance and lazy public contracts without an MLX backend."""

from __future__ import annotations

import json
import subprocess
import sys
from typing import cast

from fixture_contracts import ConstructorSpec, SpecRecord
from hypothesis import given, settings
from hypothesis import strategies as st
from reference_data import FIXTURES, PINS, ROOT, load_fixture

from segresnet_mlx import SegResNetSpec, required_shapes
from segresnet_mlx.medmlx import BUNDLES
from segresnet_mlx.predictor import BUNDLE_CONFIGS, SlidingWindowConfig


@given(roi=st.lists(st.integers(1, 256), min_size=3, max_size=3), batch=st.integers(1, 8))
@settings(max_examples=25, database=None)
def test_inferer_json_round_trip(roi: list[int], batch: int) -> None:
    """Restore sliding-window settings from either supported JSON layout."""
    config = SlidingWindowConfig(cast(tuple[int, int, int], tuple(roi)), batch, 0.25)
    document = json.loads(
        json.dumps(
            {
                "inferer": {
                    "roi_size": config.roi_size,
                    "sw_batch_size": config.sw_batch_size,
                    "overlap": config.overlap,
                }
            }
        )
    )
    assert SlidingWindowConfig.from_bundle_config(document) == config
    assert SlidingWindowConfig.from_bundle_config(document["inferer"]) == config


def test_all_recordings_have_verified_upstream_provenance_and_size() -> None:
    """Check fixture metadata, size, and recorded weight shapes against pins."""
    fixtures = list(FIXTURES.glob("*.npz"))
    assert fixtures
    for path in fixtures:
        arrays, metadata = load_fixture(path)
        assert path.stat().st_size < 650_000
        assert metadata["reference_backend"] == "torch-cpu"
        assert metadata["torch_threads"] == 1
        if path.name.startswith(("segresnet_", "predictor_")):
            assert "output" in arrays
        if "spec" in metadata:
            spec: SpecRecord = metadata["spec"].copy()
            spec["blocks_down"] = tuple(spec["blocks_down"])
            spec["blocks_up"] = tuple(spec["blocks_up"])
            assert required_shapes(SegResNetSpec(**cast(ConstructorSpec, spec))) == {
                key.removeprefix("weight::"): value.shape
                for key, value in arrays.items()
                if key.startswith("weight::")
            }


def test_bundle_contracts_follow_upstream_settings() -> None:
    """Compare bundle topology and inferer settings with recorded configs."""
    assert PINS["monai_version"] == "1.6.0"
    # wholeBody_ct_segmentation shares one committed inference.json between the
    # highres and lowres bundles; its sw_batch_size/overlap resolve through
    # `displayable_configs` rather than appearing as literal inferer values.
    resolved_elsewhere = {"wholeBody_ct_segmentation", "wholeBody_ct_segmentation_lowres"}
    for name, config in BUNDLES.items():
        document = "wholeBody_ct_segmentation" if name in resolved_elsewhere else name
        recorded = json.loads(
            (ROOT / "tests/fixtures/bundle_configs" / f"{document}.json").read_text()
        )
        if name not in resolved_elsewhere:
            assert BUNDLE_CONFIGS[name] == SlidingWindowConfig.from_bundle_config(recorded)
            assert config.sw_batch_size == recorded["inferer"]["sw_batch_size"]
            assert config.overlap == recorded["inferer"]["overlap"]
        else:
            assert recorded["inferer"]["sw_batch_size"] == "@displayable_configs#sw_batch_size"
            assert recorded["inferer"]["overlap"] == "@displayable_configs#sw_overlap"
            # Pinned displayable_configs default (same inference.json revision):
            # highres=true, sw_overlap=0.25, sw_batch_size=1.
            assert config.sw_batch_size == 1
            assert config.overlap == 0.25
        assert config.mode == recorded["inferer"].get("mode", "constant")
        assert config.padding_mode == recorded["inferer"].get("padding_mode", "constant")
        definition = recorded["network_def"]
        assert config.spec.in_channels == definition["in_channels"]
        assert config.spec.out_channels == definition["out_channels"]
        assert config.spec.init_filters == definition["init_filters"]
        assert config.spec.upsample_mode == definition.get("upsample_mode", "nontrainable")


def test_all_public_exports_load_without_the_reference_stack_or_mlx() -> None:
    """Resolve public exports while MLX and reference frameworks are blocked."""
    code = """
import importlib.abc
import sys

class BlockFrameworks(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"mlx", "torch", "monai", "ignite"}:
            raise ModuleNotFoundError(f"Unexpected eager dependency: {fullname}")

sys.meta_path.insert(0, BlockFrameworks())
import segresnet_mlx
for name in segresnet_mlx.__all__:
    getattr(segresnet_mlx, name)
"""
    subprocess.run(
        [sys.executable, "-c", code], check=True, capture_output=True, text=True, timeout=30
    )
