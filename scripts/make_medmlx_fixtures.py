"""Record pinned MONAI bundle transforms and seeded end-to-end reference cases.

Only the original MONAI/model-zoo code executes the reference math. See
docs/technical.md ("Tests and regeneration") for source preparation and macOS arm64
fixture regeneration.
"""

from __future__ import annotations

import gzip
import json
import sys
import tempfile
from argparse import ArgumentParser
from pathlib import Path
from typing import cast

import numpy as np
from reference_contracts import (
    Case,
    Float64Array,
    HostArray,
    Parser,
    Reader,
    SavedFacts,
    Spec,
    from_numpy,
)
from reference_support import Reference, seeded_network

SEED = 7081


def _nifti_bytes(data: HostArray, affine: Float64Array) -> HostArray:
    """Encode voxel data and affine as deterministic compressed NIfTI bytes."""
    from _nifti import nib

    image = nib.Nifti1Image(data, affine)
    image.set_sform(affine, code=1)
    image.set_qform(affine, code=1)
    return np.frombuffer(gzip.compress(image.to_bytes(), mtime=0), dtype=np.uint8)


def _write(path: Path, payload: HostArray) -> str:
    """Write a recorded byte array and return its path."""
    path.write_bytes(payload.tobytes())
    return str(path)


def _saved_facts(directory: Path) -> SavedFacts:
    """Read the saved prediction and its NIfTI geometry and dtype."""
    from _nifti import nib

    (path,) = [path for path in directory.rglob("*.nii.gz") if "label" not in path.name]
    image = nib.load(str(path))
    return {
        "dtype": str(image.get_data_dtype()),
        "shape": list(image.shape),
        "qform_code": int(image.header["qform_code"]),
        "sform_code": int(image.header["sform_code"]),
        "affine": np.asarray(image.affine).tolist(),
        "data": np.asanyarray(image.dataobj),
    }


def _affines() -> dict[str, Float64Array]:
    """Construct axis-aligned, flipped, and oblique reference affines."""
    oblique = np.eye(4)
    angle = 0.2
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle), 0.0], [np.sin(angle), np.cos(angle), 0.0], [0, 0, 1]]
    )
    oblique[:3, :3] = rotation @ np.diag([1.5, 1.25, 2.0])
    oblique[:3, 3] = [12.5, -30.25, 4.0]
    return {
        "axis": np.array(
            [[1.5, 0, 0, -20.0], [0, 1.25, 0, 10.0], [0, 0, 2.0, -5.0], [0, 0, 0, 1.0]]
        ),
        "flipped": np.array(
            [[-0.7, 0, 0, 30.0], [0, 0.9, 0, -10.0], [0, 0, -3.0, 5.0], [0, 0, 0, 1.0]]
        ),
        "oblique": oblique.astype(np.float32).astype(np.float64),
    }


def renal(reference: Reference) -> None:
    """Record renal preprocessing, seeded inference, and postprocessing cases."""
    import torch
    from _nifti import nib
    from monai.bundle.config_parser import ConfigParser
    from monai.data.meta_tensor import MetaTensor

    bundle = reference.bundle_root / "renalStructures_CECT_segmentation"
    sys.path.insert(0, str(bundle))
    parser = cast(Parser, ConfigParser())
    parser.read_config(str(bundle / "configs/inference.json"))
    rng = np.random.default_rng(SEED)
    arrays: dict[str, HostArray] = {}
    cases: list[Case] = []
    for case, affine in _affines().items():
        shape = (9, 8, 6)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs: dict[str, object] = {}
            for key in ("artery", "vein", "excret"):
                volume = rng.normal(0.0, 600.0, shape).astype(np.float32)
                payload = _nifti_bytes(volume, affine)
                arrays[f"{case}::{key}"] = payload
                inputs[key] = _write(root / f"{key}.nii.gz", payload)
            label = nib.Nifti1Image(np.zeros(shape, dtype=np.uint8), affine)
            nib.save(label, str(root / "label.nii.gz"))
            inputs["label"] = str(root / "label.nii.gz")
            parser["output_dir"] = str(root / "out")
            parser["dataset_dir"] = str(root)
            parser["device"] = "$torch.device('cpu')"
            data = parser.get_parsed_content("preprocessing")(inputs)
            image = np.asarray(cast("torch.Tensor", data["image"]), dtype=np.float32)
            _record_runner(parser, data, arrays, case, 3, 6, "deconv", 4)
            # Half-precision values keep the fixture small; MONAI receives float32.
            logits = rng.normal(0.0, 2.0, (6, *image.shape[1:])).astype(np.float16)
            data["pred"] = MetaTensor(from_numpy(logits.astype(np.float32)))
            # Skip the second SaveImaged (label key); only the prediction file is read back.
            for transform in parser.get_parsed_content("postprocessing").transforms[:5]:
                data = transform(data)
            saved = _saved_facts(root / "out")
        arrays[f"{case}::image"] = image
        arrays[f"{case}::logits"] = logits
        arrays[f"{case}::mask"] = saved.pop("data")
        cases.append({"case": case, "saved": saved})
    reference.save("medmlx_renal", arrays, {"cases": cases, "seed": SEED})


def brats(reference: Reference) -> None:
    """Record BraTS preprocessing, seeded inference, and postprocessing cases."""
    import torch
    from monai.bundle.config_parser import ConfigParser
    from monai.data.meta_tensor import MetaTensor

    config = reference.config("brats_mri_segmentation")
    rng = np.random.default_rng(SEED + 1)
    arrays: dict[str, HostArray] = {}
    cases: list[Case] = []
    for case, affine in _affines().items():
        shape = (10, 9, 7)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            files: list[str] = []
            for channel in range(4):
                volume = rng.gamma(2.0, 150.0, shape).astype(np.float32)
                volume[:3] = 0.0
                volume[:, -2:] = 0.0
                payload = _nifti_bytes(volume, affine)
                arrays[f"{case}::channel{channel}"] = payload
                files.append(_write(root / f"modality{channel}.nii.gz", payload))
            parser = cast(
                Parser,
                ConfigParser(
                    {
                        "image_key": "image",
                        "output_dir": str(root / "out"),
                        "output_ext": ".nii.gz",
                        "output_dtype": "uint8",
                        "output_postfix": "seg",
                        "separate_folder": True,
                        "preprocessing": config["preprocessing"],
                        "postprocessing": config["postprocessing"],
                    }
                ),
            )
            data = parser.get_parsed_content("preprocessing")({"image": files})
            image = np.asarray(cast("torch.Tensor", data["image"]), dtype=np.float32)
            _record_runner(parser, data, arrays, case, 4, 3, "nontrainable", 1)
            logits = rng.normal(0.0, 2.0, (3, *image.shape[1:])).astype(np.float16)
            data["pred"] = MetaTensor(from_numpy(logits.astype(np.float32)))
            for transform in parser.get_parsed_content("postprocessing").transforms:
                data = transform(data)
            saved = _saved_facts(root / "out")
        arrays[f"{case}::image"] = image
        arrays[f"{case}::logits"] = logits
        arrays[f"{case}::mask"] = saved.pop("data")
        cases.append({"case": case, "saved": saved})
    reference.save("medmlx_brats", arrays, {"cases": cases, "seed": SEED + 1})


def whole_body(reference: Reference) -> None:
    """``wholeBody_ct_segmentation`` (1.5 mm) and its ``_lowres`` (3 mm) sibling.

    One network and inferer; ``displayable_configs#highres`` only changes the
    resampling target. Each variant gets its own fixture file with the same
    per-case key shape as ``renal``/``brats``, so the generic medmlx-level tests
    extend to it by adding one entry to their parametrize lists.
    """
    _whole_body_variant(reference, "medmlx_wholebody", highres=True, seed=SEED + 2)
    _whole_body_variant(reference, "medmlx_wholebody_lowres", highres=False, seed=SEED + 3)


def _whole_body_variant(reference: Reference, name: str, *, highres: bool, seed: int) -> None:
    """Record one whole-body resolution using the pinned bundle transforms."""
    import torch
    from monai.bundle.config_parser import ConfigParser
    from monai.data.meta_tensor import MetaTensor

    bundle = reference.bundle_root / "wholeBody_ct_segmentation"
    rng = np.random.default_rng(seed)
    arrays: dict[str, HostArray] = {}
    cases: list[Case] = []
    for case, affine in _affines().items():
        # 105 output channels dwarf renal/BraTS; keep the volume tiny so the
        # (channels, *spatial) logits/mask arrays stay well under the size cap.
        shape = (5, 4, 3)
        parser = cast(Parser, ConfigParser())
        parser.read_config(str(bundle / "configs/inference.json"))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            volume = rng.normal(0.0, 400.0, shape).astype(np.float32)
            payload = _nifti_bytes(volume, affine)
            arrays[f"{case}::ct"] = payload
            image_path = _write(root / "image.nii.gz", payload)
            parser["output_dir"] = str(root / "out")
            parser["device"] = "$torch.device('cpu')"
            parser["displayable_configs"] = {
                "highres": highres,
                "sw_overlap": 0.25,
                "sw_batch_size": 1,
            }
            data = parser.get_parsed_content("preprocessing")({"image": image_path})
            image = np.asarray(cast("torch.Tensor", data["image"]), dtype=np.float32)
            _record_wholebody_runner(parser, data, arrays, case, image)
            logits = rng.normal(0.0, 2.0, (105, *image.shape[1:])).astype(np.float16)
            data["pred"] = MetaTensor(from_numpy(logits.astype(np.float32)))
            for transform in parser.get_parsed_content("postprocessing").transforms:
                data = transform(data)
            saved = _saved_facts(root / "out")
        arrays[f"{case}::image"] = image
        arrays[f"{case}::logits"] = logits
        arrays[f"{case}::mask"] = saved.pop("data")
        cases.append({"case": case, "saved": saved})
    reference.save(name, arrays, {"cases": cases, "seed": seed})


def _record_wholebody_runner(
    parser: Parser,
    data: dict[str, object],
    arrays: dict[str, HostArray],
    case: str,
    image: HostArray,
) -> None:
    """A tiny seeded network through the bundle's actual gaussian/replicate inferer."""
    import torch
    from monai.data.meta_tensor import MetaTensor
    from monai.inferers.inferer import SlidingWindowInferer

    for suffix, decoder, seed in (("runner", False, 5), ("runner_decoder", True, 6)):
        spec: Spec = Spec(
            in_channels=1,
            out_channels=105,
            init_filters=8,
            blocks_down=(1, 1) if decoder else (1,),
            blocks_up=(1,) if decoder else (),
            num_groups=8,
            norm_eps=1e-5,
            upsample_mode="nontrainable",
        )
        network, _ = seeded_network(spec, np.random.default_rng(seed), scale=0.2)
        if case == "axis":
            arrays[f"{suffix}::spec"] = np.array(json.dumps(spec))
        values = np.asarray(image, dtype=np.float32)[None]
        with torch.no_grad():
            logits = cast(
                torch.Tensor,
                SlidingWindowInferer((8, 8, 8), 1, 0.25, mode="gaussian", padding_mode="replicate")(
                    from_numpy(values),
                    network,
                ),
            )
        result = dict(data, pred=MetaTensor(logits[0]))
        for transform in parser.get_parsed_content("postprocessing").transforms:
            if transform.__class__.__name__ == "SaveImaged":
                break
            result = transform(result)
        arrays[f"{case}::{suffix}_logits"] = logits.numpy()
        arrays[f"{case}::{suffix}_mask"] = np.asarray(cast("torch.Tensor", result["pred"]))[
            0
        ].astype(np.float32)


def _record_runner(
    parser: Parser,
    data: dict[str, object],
    arrays: dict[str, HostArray],
    case: str,
    incoming: int,
    outgoing: int,
    mode: str,
    batch: int,
) -> None:
    """Record seeded logits and masks with and without a decoder."""
    import torch
    from monai.data.meta_tensor import MetaTensor
    from monai.inferers.inferer import SlidingWindowInferer

    for suffix, decoder, seed in (("runner", False, 5), ("runner_decoder", True, 6)):
        spec: Spec = Spec(
            in_channels=incoming,
            out_channels=outgoing,
            init_filters=8,
            blocks_down=(1, 1) if decoder else (1,),
            blocks_up=(1,) if decoder else (),
            num_groups=8,
            norm_eps=1e-5,
            upsample_mode=mode if decoder else "nontrainable",
        )
        network, state = seeded_network(spec, np.random.default_rng(seed), scale=0.2)
        if case == "axis":
            arrays[f"{suffix}::spec"] = np.array(json.dumps(spec))
            arrays.update({f"{suffix}::weight::{key}": value for key, value in state.items()})
        values = np.asarray(cast("torch.Tensor", data["image"]), dtype=np.float32)[None]
        with torch.no_grad():
            logits = cast(
                torch.Tensor,
                SlidingWindowInferer((8, 8, 8), batch, 0.25 if incoming == 3 else 0.5)(
                    from_numpy(values),
                    network,
                ),
            )
        result = dict(data, pred=MetaTensor(logits[0]))
        for transform in parser.get_parsed_content("postprocessing").transforms:
            if transform.__class__.__name__ == "SaveImaged":
                break
            result = transform(result)
        arrays[f"{case}::{suffix}_logits"] = logits.numpy()
        arrays[f"{case}::{suffix}_mask"] = np.asarray(cast("torch.Tensor", result["pred"]))[
            0
        ].astype(
            np.float32 if incoming == 3 else np.uint8,
        )


def nifti_reader(reference: Reference) -> None:
    """Record MONAI reader behavior for differing NIfTI affine forms."""
    from _nifti import nib
    from monai.data.image_reader import NibabelReader

    rng = np.random.default_rng(SEED)
    volume = rng.normal(0.0, 600.0, (9, 8, 6)).astype(np.float32)
    affine = _affines()["axis"]
    arrays: dict[str, HostArray] = {}
    cases = ("disagree", "unset", "sform_spacing")
    with tempfile.TemporaryDirectory() as directory:
        for case in cases:
            image = nib.Nifti1Image(volume, affine)
            image.set_sform(affine, code=1)
            image.set_qform(affine, code=1)
            alternate = affine.copy()
            if case == "disagree":
                alternate[0, 3] += 3
                image.set_qform(alternate, code=1)
            elif case == "sform_spacing":
                alternate[:3, 0] *= 2
                image.set_sform(alternate, code=1)
            else:
                image.set_qform(None, code=0)
                image.set_sform(None, code=0)
            payload = np.frombuffer(gzip.compress(image.to_bytes(), mtime=0), dtype=np.uint8)
            path = _write(Path(directory) / f"{case}.nii.gz", payload)
            reader = cast(Reader, NibabelReader())
            voxels, metadata = reader.get_data(reader.read(path))
            arrays.update(
                {
                    f"{case}::nifti": payload,
                    f"{case}::data": voxels,
                    f"{case}::affine": np.asarray(metadata["affine"]),
                }
            )
    reference.save("nifti_reader", arrays, {"cases": cases, "seed": SEED})


def main() -> None:
    """Record all bundle and reader fixtures from the supplied upstream root."""
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-root", type=Path, required=True)
    arguments = parser.parse_args()
    reference = Reference.load(arguments.bundle_root)
    renal(reference)
    brats(reference)
    whole_body(reference)
    nifti_reader(reference)


if __name__ == "__main__":
    main()
