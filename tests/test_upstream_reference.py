"""Bundle settings against the pinned upstream MONAI bundle configs."""

from __future__ import annotations

import json

from reference_data import ROOT

from segresnet_mlx.medmlx import BUNDLES
from segresnet_mlx.predictor import BUNDLE_CONFIGS, SlidingWindowConfig


def test_bundle_contracts_follow_upstream_settings() -> None:
    """Compare bundle topology and inferer settings with recorded configs."""
    for name, config in BUNDLES.items():
        recorded = json.loads((ROOT / "tests/fixtures/bundle_configs" / f"{name}.json").read_text())
        assert BUNDLE_CONFIGS[name] == SlidingWindowConfig.from_bundle_config(recorded)
        assert config.sw_batch_size == recorded["inferer"]["sw_batch_size"]
        assert config.overlap == recorded["inferer"]["overlap"]
        assert config.mode == recorded["inferer"].get("mode", "constant")
        assert config.padding_mode == recorded["inferer"].get("padding_mode", "constant")
        definition = recorded["network_def"]
        assert config.spec.in_channels == definition["in_channels"]
        assert config.spec.out_channels == definition["out_channels"]
        assert config.spec.init_filters == definition["init_filters"]
        assert config.spec.upsample_mode == definition.get("upsample_mode", "nontrainable")
