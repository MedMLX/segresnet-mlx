"""Bundle settings against the pinned upstream MONAI bundle configs."""

from __future__ import annotations

import json

from reference_data import ROOT

from segresnet_mlx.medmlx import BUNDLES
from segresnet_mlx.predictor import BUNDLE_CONFIGS, SlidingWindowConfig


def test_bundle_contracts_follow_upstream_settings() -> None:
    """Compare bundle topology and inferer settings with recorded configs."""
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
