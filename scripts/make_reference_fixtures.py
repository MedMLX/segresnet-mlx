"""Record seeded MONAI SegResNet and SlidingWindowInferer outputs on Torch CPU.

No implementation under test is imported. See docs/technical.md ("Tests and
regeneration") for the pinned upstream source files and the macOS arm64 recordings.
"""

from __future__ import annotations

from argparse import ArgumentParser
from collections.abc import Callable
from pathlib import Path
from typing import cast

import numpy as np
import torch
from monai.inferers.inferer import SlidingWindowInferer
from reference_contracts import HostArray, Inferer, Metadata, Spec, from_numpy
from reference_support import Reference, seeded_network


def _spec(incoming: int, outgoing: int, mode: str, decoder: bool) -> Spec:
    """Construct a reduced SegResNet topology for an upstream recording."""
    return Spec(
        in_channels=incoming,
        out_channels=outgoing,
        init_filters=4,
        blocks_down=(1, 1) if decoder else (1,),
        blocks_up=(1,) if decoder else (),
        num_groups=2,
        norm_eps=1e-5,
        upsample_mode=mode,
    )


def network_fixture(reference: Reference, name: str, spec: Spec, seed: int) -> None:
    """Record seeded network outputs and intermediate MONAI operators."""
    rng = np.random.default_rng(seed)
    network, _ = seeded_network(spec, rng, dtype=np.float64)
    host = rng.normal(size=(2, spec["in_channels"], 4, 6, 8)).astype(np.float32)
    arrays: dict[str, HostArray] = {"input": host}
    with torch.no_grad():
        values = from_numpy(host)
        arrays["output"] = network(values).numpy()
        initial = network.convInit(values)
        residual = network.down_layers[0](initial)
        arrays.update(encoder_initial=initial.numpy(), encoder_residual=residual.numpy())
        if spec["blocks_up"]:
            downsampled = network.down_layers[1][0](residual)
            encoded = network.down_layers[1][1](downsampled)
            arrays.update(encoder_downsampled=downsampled.numpy(), encoder_output=encoded.numpy())
        if spec["upsample_mode"] == "deconv":
            values = rng.normal(size=(1, spec["init_filters"], 2, 3, 2)).astype(np.float32)
            arrays.update(
                deconv_input=values,
                deconv_output=network.up_samples[0][1](from_numpy(values)).numpy(),
            )
        if not spec["blocks_up"]:
            grouped = rng.normal(size=(2, 2, 389)).astype(np.float32)
            arrays.update(
                group_input=grouped, group_output=from_numpy(grouped).mean(-1, keepdim=True).numpy()
            )
    reference.save(name, arrays, {"spec": spec, "seed": seed})


def predictor_fixture(
    reference: Reference,
    name: str,
    model_id: str,
    config: Inferer,
    kind: str,
    seed: int,
) -> None:
    """Record one sliding-window case through MONAI's ``SlidingWindowInferer``.

    ``kind`` selects the per-window network: a ``standin*`` function (elementwise, or
    using a per-window mean for ``standin_reduction``) so window placement and
    blending are tested independently of SegResNet, or a seeded SegResNet with or
    without a decoder. ``config`` is the inferer section, possibly with a reduced
    ``roi_size`` so the synthetic input stays small.
    """
    definition = reference.config(model_id)["network_def"]
    rng = np.random.default_rng(seed)
    actual_roi = config["roi_size"] == reference.config(model_id)["inferer"]["roi_size"]
    batch = 1 if actual_roi and model_id.startswith("brats") else 2
    spatial = (3, 5, 7) if actual_roi else (11, 9, 3)
    host = rng.normal(size=(batch, definition["in_channels"], *spatial)).astype(np.float32)
    arrays: dict[str, HostArray] = {"input": host}
    metadata: Metadata = {"model_id": model_id, "kind": kind, "seed": seed, "inferer": config}
    batches: list[list[int]] = []
    if kind.startswith("standin"):
        coefficients = rng.normal(0, 0.25, (definition["out_channels"], 2)).astype(np.float32)
        arrays["coefficients"] = coefficients

        def standin(values: torch.Tensor) -> torch.Tensor:
            """Produce deterministic per-window outputs without SegResNet."""
            origin = (
                values.mean((2, 3, 4), keepdim=True)
                if kind == "standin_reduction"
                else values[:, :, :1, :1, :1]
            )
            return torch.cat(
                [
                    (
                        values[:, index % values.shape[1] : index % values.shape[1] + 1]
                        + origin[:, index % values.shape[1] : index % values.shape[1] + 1]
                    )
                    * scale
                    + bias
                    for index, (scale, bias) in enumerate(coefficients)
                ],
                dim=1,
            )

        network: Callable[[torch.Tensor], torch.Tensor] = standin
    else:
        spec = _spec(
            definition["in_channels"],
            definition["out_channels"],
            definition.get("upsample_mode", "nontrainable"),
            kind == "decoder_network",
        )
        network, _ = seeded_network(spec, rng)
        metadata["spec"] = spec

    def measured(values: torch.Tensor) -> torch.Tensor:
        """Record each window batch shape before invoking the network."""
        batches.append(list(values.shape))
        return network(values)

    with torch.no_grad():
        arrays["output"] = cast(
            torch.Tensor,
            SlidingWindowInferer(
                config["roi_size"],
                config["sw_batch_size"],
                config["overlap"],
                mode=config.get("mode", "constant"),
                padding_mode=config.get("padding_mode", "constant"),
            )(
                from_numpy(arrays["input"]),
                measured,
            ),
        ).numpy()
    arrays["window_batches"] = np.asarray(batches, dtype=np.int64)
    reference.save(name, arrays, metadata)


def predictors(reference: Reference) -> None:
    """Record renal and BraTS sliding-window configurations."""
    names = ("brats_mri_segmentation", "renalStructures_CECT_segmentation")
    for index, model_id in enumerate(names):
        actual = reference.config(model_id)["inferer"]
        reduced = cast(Inferer, dict(actual, roi_size=[6, 6, 4] if index == 0 else [6, 6, 6]))
        prefix = "predictor_brats" if index == 0 else "predictor_renal"
        for suffix, config, kind in (
            ("standin", reduced, "standin"),
            (
                "strided_reduction",
                cast(Inferer, dict(reduced, sw_batch_size=1)),
                "standin_reduction",
            ),
            ("actual_roi", actual, "standin"),
            ("no_decoder_network", reduced, "no_decoder_network"),
            ("decoder_network", reduced, "decoder_network"),
        ):
            predictor_fixture(reference, f"{prefix}_{suffix}", model_id, config, kind, 1800 + index)
        if index == 1:
            predictor_fixture(
                reference,
                f"{prefix}_single_window_batch",
                model_id,
                cast(Inferer, dict(reduced, sw_batch_size=1)),
                "standin",
                1801,
            )


def wholebody_predictors(reference: Reference) -> None:
    """``wholeBody_ct_segmentation``: gaussian blending, replicate padding.

    Its ``sw_batch_size``/``overlap`` resolve through the bundle's
    ``displayable_configs`` (not literal in ``inferer``); their pinned defaults
    (``sw_batch_size=1``, ``sw_overlap=0.25``) are substituted here.
    """
    model_id = "wholeBody_ct_segmentation"
    raw = reference.config(model_id)["inferer"]
    actual = cast(Inferer, dict(raw, sw_batch_size=1, overlap=0.25))
    reduced = cast(Inferer, dict(actual, roi_size=[6, 6, 4]))
    prefix = "predictor_wholebody"
    for suffix, config, kind in (
        ("standin", reduced, "standin"),
        ("strided_reduction", cast(Inferer, dict(reduced, sw_batch_size=1)), "standin_reduction"),
        ("actual_roi", actual, "standin"),
        ("no_decoder_network", reduced, "no_decoder_network"),
        ("decoder_network", reduced, "decoder_network"),
    ):
        predictor_fixture(reference, f"{prefix}_{suffix}", model_id, config, kind, 1802)


def main() -> None:
    """Record network and sliding-window fixtures from the upstream root."""
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--predictor-only", action="store_true")
    args = parser.parse_args()
    reference = Reference.load(args.bundle_root)
    if not args.predictor_only:
        spec = cast(Spec, _spec(4, 3, "nontrainable", False) | {"blocks_down": (2,)})
        network_fixture(reference, "segresnet_no_decoder", spec, 1729)
        for mode, seed in (("nontrainable", 1730), ("deconv", 1731)):
            network_fixture(reference, f"segresnet_{mode}", _spec(3, 6, mode, True), seed)
    predictors(reference)
    wholebody_predictors(reference)


if __name__ == "__main__":
    main()
