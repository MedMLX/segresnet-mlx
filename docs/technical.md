# segresnet-mlx technical notes

Full details behind the [README](../README.md).

## Overview

Native MLX implementation of [MONAI SegResNet](https://github.com/Project-MONAI/MONAI)
for MedMLX. [MedMLX's segment extra](https://github.com/MedMLX/medmlx#install) installs it with the
other segmentation models.

The supported target is **macOS arm64 with MLX on Metal**. Package metadata and
dependency resolution declare only that environment. Metal is required for
native inference; reference generation uses the upstream Torch CPU graph on
the same Mac.

The package provides strict safetensors loading, patch and sliding-window
raw-logit inference, bounded telemetry, and a NIfTI runner for renal CECT, BraTS
MRI and whole-body CT. Runtime code uses `medmlx-core`, NumPy and nibabel. Torch
and MONAI are used only for reference generation. Importing the
package or its model declaration does not import MLX.

**Qualification:** The reference is the original MONAI implementation. Seeded
public-network, operator, sliding-window and runner logit checks retain their
1e-6 absolute bound; encoder intermediates retain `2**-19`. Renal normalized
intensities are checked within one float32 rounding unit on [0, 1], and BraTS
normalized intensities within `2**-20` (about 9.54e-7). Saved fixture masks agree
exactly.

The released renal checkpoint is also checked on dense and sparse seeded inputs
with unchanged `rtol=atol=1e-4` and exact sign agreement. On 12 October 2026,
source-order FP32 convolution and GroupNorm removed the previously retained
sparse/full-case discrepancy. On the retained external three-phase CT case,
all 85,089,840 raw logits were bitwise identical to a fresh independently parsed
MONAI CPU bundle run, and all 5,935,600 exported mask voxels agreed. This was
measured on Apple M1 Max, macOS 27.0.1, MLX 0.32.3, Torch 2.14.1 and MONAI 1.6.0;
bitwise equality is an observation for those runs, not a promise for every
platform or input. The case has no reference labels and retains manual
registration/cropping and a denoising substitution. Anatomical and clinical
accuracy remain unqualified.

## Install

PyPI distributes code only. Public native weights are separate downloads:

| `bundle` | Hugging Face repository | File | Revision |
| --- | --- | --- | --- |
| `brats_mri_segmentation` | [MedMLX/segresnet-brats-mlx](https://huggingface.co/MedMLX/segresnet-brats-mlx) | `segresnet-brats.safetensors` | `14f9cda3649965729663c7365c49fd839c81a01b` |
| `renalStructures_CECT_segmentation` | [MedMLX/segresnet-renal-cect-mlx](https://huggingface.co/MedMLX/segresnet-renal-cect-mlx) | `segresnet-renal-cect.safetensors` | `fa67ec6dca88f1dbbf6f52d6061b34ddefe45057` |
| `wholeBody_ct_segmentation` | [MedMLX/segresnet-wholebody-mlx](https://huggingface.co/MedMLX/segresnet-wholebody-mlx) | `segresnet-wholebody-high.safetensors` | `9f5433e21545d138c2a677ea0541ba249881fc21` |
| `wholeBody_ct_segmentation_lowres` | [MedMLX/segresnet-wholebody-mlx](https://huggingface.co/MedMLX/segresnet-wholebody-mlx) | `segresnet-wholebody-low.safetensors` | `9f5433e21545d138c2a677ea0541ba249881fc21` |

Follow their weight cards and publisher terms. Pass the downloaded `.safetensors`
file as `weights_path` with its matching `bundle`; no conversion is needed. The
runner requires a file path, not a bundle directory or an original Torch `.pt`
checkpoint. Hugging Face cache file paths from `hf_hub_download` are accepted.
Loading validates the expected tensor inventory and finite values but does not
verify checkpoint identity or origin. Availability of weights does not extend the
qualification described above.

MLX installation and Metal runtime checks are owned by `medmlx-core`. There is no
device argument or backend fallback. `medmlx` probes the runtime in an isolated
process before inference and reports its availability and setup hint.

## Upstream reference

The network, inferer and transforms use **MONAI 1.6.0**, revision
[`eccefc57550b111ed781d82249dfe77872a0e918`](https://github.com/Project-MONAI/MONAI/tree/eccefc57550b111ed781d82249dfe77872a0e918).
The original bundle definitions come from MONAI model-zoo revision
[`db9eeeef7ba64ce2ced0230d9bd4c524f69b7f0c`](https://github.com/Project-MONAI/model-zoo/tree/db9eeeef7ba64ce2ced0230d9bd4c524f69b7f0c):

- `models/renalStructures_CECT_segmentation/configs/inference.json`
- `models/renalStructures_CECT_segmentation/scripts/my_transforms.py`
- `models/brats_mri_segmentation/configs/inference.json`
- `models/wholeBody_ct_segmentation/configs/inference.json`

`scripts/upstream_reference.json` is the canonical source/revision manifest.
It describes the historical recordings; current dev dependencies use MONAI 1.6.1
for its security fixes. Do not downgrade MONAI to rerun a recorder. Migrate its
explicit reference pins and regenerate the fixtures together before recording,
reviewing numerical changes and retaining accurate source records.
The recorder refuses different MONAI revisions or missing bundle files. The four
inference sections committed in `tests/fixtures/bundle_configs/` match those
original bundle files (one committed file covers both whole-body bundles; they
share one `inference.json`). The network uses their inference graph in float32
with dropout disabled, group normalization, ReLU, skip addition, and either
transposed convolution or trilinear 2x interpolation (`align_corners=False`).
Native MLX operations come from the shared core. Convolution and GroupNorm use
its source-order FP32 operators: channel-major fused multiply-add accumulation
and bounded four-lane Welford moments. Correcting only one of these operators
leaves amplified drift in the released renal checkpoint. The decoder uses the
core's phase-packed transposed convolution or trilinear interpolation.

The BraTS bundle enables mixed-precision evaluation upstream. The reference
contract uses its float32 evaluation graph; mixed-precision arithmetic is
outside this qualification. The renal label input exists only to export a
reference label in the upstream bundle; it is not needed to segment an image and
is omitted from the runner.

The whole-body bundle's `sw_batch_size` and `overlap` resolve through its
`displayable_configs` indirection rather than appearing as literal `inferer`
values; the pinned revision's defaults (`sw_batch_size=1`, `sw_overlap=0.25`,
`highres=true`) are substituted explicitly rather than evaluated generically,
consistent with reading bundle sections without evaluating expressions. Unlike
renal and BraTS, it blends windows with a Gaussian weight and pads with
`replicate` rather than zeros, and its `highres`/`lowres` variants share one
network and inferer, differing only in the resampling target (1.5 mm vs. 3 mm)
and checkpoint file (`model.pt` vs. `model_lowres.pt`).

## Public API

```python
from pathlib import Path

import numpy as np
from segresnet_mlx import (
    BUNDLE_CONFIGS,
    SegResNetSpec,
    forward_patch,
    load_model,
    predict_logits,
)

spec = SegResNetSpec(3, 6, 32, upsample_mode="deconv")
network = load_model(Path("segresnet-renal-cect.safetensors"), spec=spec)
patch = np.zeros((1, 3, 32, 32, 32), dtype=np.float32)
logits = forward_patch(network, patch)
logits = predict_logits(
    patch,
    BUNDLE_CONFIGS["renalStructures_CECT_segmentation"],
    network=network,
)
```

`SegResNet(spec, weights, mx=mx)` accepts converted MLX weights and returns MLX
raw logits. `forward_patch(network, values)` accepts finite float32 NumPy NCDHW
arrays and returns synchronized float32 NumPy logits. Patch spatial sizes must
be divisible by `2 ** len(spec.blocks_up)`.

`predict_logits(values, configuration, network=network, telemetry=None)` accepts
preprocessed float32 NumPy or MLX `(N, C, D, H, W)` arrays and returns NumPy logits
at the input spatial size. It does not normalize intensities, apply activation,
map labels or read/write images. `SlidingWindowConfig.from_bundle_config(config)`
reads the inferer section (or that section alone) without evaluating expressions.
Renal and BraTS use the upstream defaults: constant blending and zero padding;
the whole-body bundles set gaussian blending and replicate padding. Sigma scale
0.125 and no buffering apply to every bundle.

| Bundle | ROI | Window batch | Overlap | Mode | Padding |
| --- | --- | --- | --- | --- | --- |
| `brats_mri_segmentation` | (240, 240, 160) | 1 | 0.5 | constant | constant |
| `renalStructures_CECT_segmentation` | (96, 96, 96) | 4 | 0.25 | constant | constant |
| `wholeBody_ct_segmentation` | (96, 96, 96) | 1 | 0.25 | gaussian | replicate |
| `wholeBody_ct_segmentation_lowres` | (96, 96, 96) | 1 | 0.25 | gaussian | replicate |

Original volume dimensions need not satisfy the patch divisor. Core inference
pads, places windows, stitches on the NumPy host and crops padding. At batch size
1 the predictor preserves upstream window strides using core window coordinates.

`required_shapes(spec)` defines the consumed parameter keys and shapes.
`load_model` validates the safetensors parameter names, shapes and finite float32
values against `required_shapes(spec)`.

`SegResNet` and `load_model` accept `arithmetic_profile="torch-order-fp32"` by
default. `compensated-fp32` uses paired Metal accumulators for convolution and
GroupNorm, rounding layer outputs to float32. `compensated-fp48` carries paired
values through the hidden layers and requires `upsample_mode="deconv"` (renal
among the runner's bundles). With this profile, the runner stitches in float64 on
the host before exporting float32 logits. All profiles execute the network on Metal.

`WindowTelemetry(path, record)` records bounded progress and final timing
statistics using schema `medmlx.segresnet.mlx.v1`. Use `telemetry.session()` for
caller-owned inference, or `attach(engine, events=Events)` with the caller's event
enum. `BundlePredictor(network, telemetry=...)` synchronizes each callback and
records host-export timing and peak memory. Its `predict_logits` method keeps the
MLX allocator cache across windows and clears it once after stitching; it leaves
completion to the caller; the module-level function finishes its
session after stitching. Window counts count callback batches.

## Use with medmlx

The `medmlx.models` entry point declares `segresnet_mlx.medmlx:MODEL` as model
`segresnet`. `run` takes keyword-only arguments:

| Argument | Meaning |
| --- | --- |
| `bundle` | `renalStructures_CECT_segmentation`, `brats_mri_segmentation`, `wholeBody_ct_segmentation` or `wholeBody_ct_segmentation_lowres` |
| `arterial_path`, `venous_path`, `excretory_path` | Renal: three CT phases on one grid |
| `t1c_path`, `t1_path`, `t2_path`, `flair_path` | BraTS: four MRI modalities on one grid |
| `image_path` | Whole-body: one CT volume |
| `weights_path` | Path to the matching native `.safetensors` file, as listed above |
| `output_dir` | Created if missing |
| `overwrite` | False by default; refuses an existing mask unless true |
| `arithmetic_profile` | `torch-order-fp32` (default), `compensated-fp32` or `compensated-fp48`; the last supports only the renal bundle |

It writes `<bundle>_mask.nii.gz` on the input shape and affine and returns
`{"outputs": {"mask": absolute_path}, "details": {...}}` with JSON-compatible
labels, shapes, window settings, MLX device, peak memory and timing.
Current source requires real, finite 3D NIfTI volumes with values representable in
float32, on a common invertible voxel grid. Spatial units must be mm or unspecified
(assumed mm); written masks declare mm. The reader applies
MONAI's header spacing/form correction in memory, then uses nibabel's selected
affine, including its sform/qform precedence and base affine when neither form
is coded. The output keeps that effective input grid; inconsistent header spacing
can therefore change the selected affine. The runner does not impose additional
qform/sform agreement rules or rewrite input files.

Renal preprocessing orders artery/vein/excretory channels, reorients to LPS,
resamples to 0.8 mm, and clips/scales [-1000, 1000] HU to [0, 1]. Postprocessing
inverts the spatial transforms, applies sigmoid and threshold `>= 0.5`, then
merges classes with the lowest nonzero channel winning. Sigmoid evaluates its
negative tail without exponent overflow while preserving the source float32
expression near the 0.5 threshold. Labels 1 artery, 2 vein,
3 ureter, 4 neoplasm and 5 parenchyma are saved as float32.

BraTS normalizes nonzero voxels per modality, applies sigmoid, inversion and
threshold `>= 0.5`, then gives enhancing tumor (4) priority over tumor core (1)
and whole tumor outside the core (2). It saves uint8 masks. These dtypes and
priorities follow the original bundle writers and transforms.

Whole-body preprocessing reorients the single CT channel to RAS, resamples to
1.5 mm (`wholeBody_ct_segmentation`) or 3 mm (`wholeBody_ct_segmentation_lowres`),
normalizes nonzero voxels, then rescales by the volume's own min/max to [-1, 1].
Postprocessing takes a softmax over the 105 channels, argmax to one label per
voxel on the resampled grid, then inverts back to the source grid with
nearest-neighbor resampling (matching `Invertd(nearest_interp=True)`) and saves
float32 labels 0-104: 0 background, 1 spleen, 2 kidney_right, 3 kidney_left,
4 gallbladder, 5 liver, 6 stomach, 7 aorta, 8 inferior_vena_cava,
9 portal_vein_and_splenic_vein, 10 pancreas, 11 adrenal_gland_right,
12 adrenal_gland_left, 13-17 lung lobes (upper/lower left, upper/middle/lower
right), 18-41 vertebrae L5-C1, 42 esophagus, 43 trachea, 44-49 heart chambers
and pulmonary artery, 50 brain, 51-54 iliac artery/vein (left/right),
55 small_bowel, 56 duodenum, 57 colon, 58-69 left ribs 1-12, 70-81 right ribs
1-12, 82-83 humerus, 84-85 scapula, 86-87 clavicula, 88-89 femur, 90-91 hip,
92 sacrum, 93 face, 94-99 gluteus maximus/medius/minimus (left/right),
100-101 autochthon, 102-103 iliopsoas, 104 urinary_bladder — the TotalSegmentator
104-structure set the bundle was trained on, as recorded in its `metadata.json`
`channel_def`.

Bad inputs raise `InvalidInputError`; missing, unconverted or mismatched weights
raise `AssetNotReadyError`; inference failures raise `ModelExecutionError`.

## Tests and regeneration on macOS arm64

The scripts execute only pinned upstream MONAI, Torch, NumPy and nibabel, never
the package under test. No checkpoints are downloaded. Tests regenerate seeded
model parameters from MONAI and compare against the recorded outputs; parameter
arrays are not stored in the fixtures. Seeds 1729-1731 cover
patch networks, encoder intermediates, group means and transposed convolution;
1800-1802 cover padded/overlapping windows, cross-image batches, both original
ROIs, channel expansion/contraction, single-window batches, and (1802) the
whole-body bundle's gaussian/replicate predictor path. Seeds 7081-7082
cover axis-aligned, flipped and oblique NIfTI cases, plus reader cases with unset,
disagreeing or inconsistent-spacing forms; 7083-7084 add the same cases for the
whole-body highres/lowres bundles. Runner seeds 5 and 6 add upstream end-to-end
logits and masks for decoder-free and reduced decoder graphs, reused across all
four bundles. Telemetry is checked as a package lifecycle contract rather than
a model fixture.

Each `.npz` uses fixture schema version 2 and records upstream revisions, Torch
version/thread count, seeded inputs/weights and outputs. Fixtures are well under 1 MB each. The
recordings live in `tests/fixtures/darwin-arm64/` and load directly, without
platform or backend selection. Each records its `darwin-arm64` recording host.
Reference math runs on Torch CPU with one thread. A missing recording fails with
regeneration instructions. Creating a recording does not qualify its MLX comparison.
Decoder-free cases use `no_decoder` names; all native graph tests use Metal.

```bash
export UV_CACHE_DIR="$TMPDIR/uv-cache"
uv venv "$TMPDIR/venv-segresnet-mlx" --python 3.13
uv pip install --python "$TMPDIR/venv-segresnet-mlx/bin/python" -e '.[dev,conversion]'

# Download only the four pinned upstream source files into temporary storage.
UPSTREAM="$TMPDIR/segresnet-upstream"
REV=db9eeeef7ba64ce2ced0230d9bd4c524f69b7f0c
for FILE in renalStructures_CECT_segmentation/configs/inference.json \
            renalStructures_CECT_segmentation/scripts/my_transforms.py \
            brats_mri_segmentation/configs/inference.json \
            wholeBody_ct_segmentation/configs/inference.json; do
    mkdir -p "$UPSTREAM/$(dirname "$FILE")"
    curl --fail --location \
        "https://raw.githubusercontent.com/Project-MONAI/model-zoo/$REV/models/$FILE" \
        --output "$UPSTREAM/$FILE"
done
PYTHONDONTWRITEBYTECODE=1 "$TMPDIR/venv-segresnet-mlx/bin/python" \
    scripts/make_reference_fixtures.py --bundle-root "$UPSTREAM"
PYTHONDONTWRITEBYTECODE=1 "$TMPDIR/venv-segresnet-mlx/bin/python" \
    scripts/make_medmlx_fixtures.py --bundle-root "$UPSTREAM"
HYPOTHESIS_STORAGE_DIRECTORY="$TMPDIR/hypothesis-segresnet-mlx" \
    PYTHONDONTWRITEBYTECODE=1 "$TMPDIR/venv-segresnet-mlx/bin/python" \
    -m pytest -q -p no:cacheprovider
"$TMPDIR/venv-segresnet-mlx/bin/ruff" check . --no-cache
```

Both recorders require macOS arm64 and write to the fixed recording directory.
`--predictor-only` on the first recorder updates only sliding-window cases.
Numerical network/reduction tests require Metal; preprocessing,
writer, source metadata and telemetry tests can run without it.
Use `-m 'not mlx'` with pytest to run that portion of the suite in a sandbox
without Metal. Full-suite collection does not load MLX; selecting a Metal test
loads the shared runtime when the test executes.
The strict Pyright and annotation checks are configured in `make typecheck`.
To include the released-checkpoint regression tests, stage the original bundle
and download the published renal safetensors file, then run:

```bash
MEDMLX_TEST_RENAL_WEIGHTS=/path/to/segresnet-renal-cect.safetensors \
MEDMLX_TEST_CHECKPOINT_ROOT=/path/to/model-zoo \
  uv run --locked --extra dev python -m pytest -q tests/test_torch_oracles.py
```

The root must contain `renalStructures_CECT_segmentation/models/model.pt`.
These checks cover all three arithmetic profiles with two independent seeds
for both the dense 8-cube and sparse 96-cube controls. Without these assets, those 12 tests are skipped; the remaining fixture and runtime tests still run.

## Optional streaming inference and export

Streaming requires segresnet-mlx 0.3.1 or later.

`run(..., streaming=True, scratch_dir="scratch")` uses an ROI-depth host
ring and mapped logits instead of a full resident class-volume blend buffer.
Whole-body softmax/argmax is followed by the original nearest label restoration;
renal logits are restored one channel at a time before sigmoid/threshold and
lowest-nonzero-class selection; BraTS uses bounded activation blocks with its
existing label priority. Compensated FP48 retains float64 blending and rounds
mapped logits once to float32 before postprocessing. Run details record the option.

Temporary files are removed on success and failure. They require disk space;
file mapping bounds explicit intermediates but does not cap OS page caching.
The final label volume remains resident. `streaming=False` remains the default.

Streaming uses `medmlx_core.volume_workspace`, included in the pinned
`medmlx-core` 0.2.5; default calls are unchanged.
`BundlePredictor.predict_logits(..., workspace=workspace)` and
`predict_wide_logits(..., workspace=workspace)` expose the same storage option to
callers that own export. Keep the workspace open until logits are consumed.
`tests/test_transforms.py` compares every bundle's streamed labels with its original
axis-aligned, flipped and oblique fixtures. Integrated model/streaming qualification
and memory/latency measurements require Metal; host postprocessing parity alone
is not a network qualification.

### Measured streaming export (2026-10-28)

The real renal and whole-body restoration paths were measured with core
sliding-window inference and a synthetic, patch-dependent Metal predictor.
Apple M5 Max, 128 GiB, macOS 27.2, Python 3.13.17, MLX 0.32.4,
`MLX_ENABLE_TF32=0`; input `(1, 1, 384, 128, 128)`, ROI 96 cubed, overlap 0.5,
six or 105 classes, predictor batch four or one respectively. Restoration used
the nonidentity original grid `(400, 136, 132)` and diagonal affine scale
`network_shape / original_shape`.

| Case | Dense median (s) | Streaming median (s) | Dense peak RSS (GiB) | Streaming peak RSS (GiB) |
| --- | ---: | ---: | ---: | ---: |
| Renal | 1.671 | 1.517 | 1.468 | 0.847 |
| Whole-body | 6.569 | 5.602 | 11.083 | 7.420 |

Three fresh processes per mode alternated order, with one predictor-batch warmup
and allocator-cache clearing. Times include stitching, restoration and mapped-file
cleanup; they exclude inputs/imports and output serialization. RSS is the largest
Darwin process high-water mark, including resident mapped pages. Other host
conversion jobs were present. Renal time ranges were 1.648–1.767 s dense and
1.507–1.561 s streaming; whole-body ranges were 6.545–6.634 s and 5.445–5.860 s.
All 7,180,800 output labels matched exactly in every paired repetition, exercising
all six/105 labels. Peak MLX was unchanged (108.0/361.1 MiB); this reduces host
working memory. Scratch storage peaked at 0.141/4.922 GiB before cleanup.

The whole-body probe reduced peak RSS by 33.0% and time by 14.7%; renal reduced
RSS by 42.3% and time by 9.2%. These are storage/restoration measurements, not
released-weight network speedups, BraTS measurements or complete FP48 qualification.
The measurement scripts and raw samples were retained outside the repository.
