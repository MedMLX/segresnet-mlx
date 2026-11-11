<div align="center">

# segresnet-mlx

Segment renal arteries, veins, ureters, tumors and parenchyma in multiphase CT with MONAI SegResNet and native MLX.

[![PyPI](https://img.shields.io/pypi/v/segresnet-mlx)](https://pypi.org/project/segresnet-mlx/)
[![Python](https://img.shields.io/pypi/pyversions/segresnet-mlx)](https://pypi.org/project/segresnet-mlx/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue)](https://github.com/MedMLX/segresnet-mlx/blob/v0.4.0/LICENSE)
[![Apple Silicon](https://img.shields.io/badge/Apple_Silicon-MLX-black?logo=apple)](https://github.com/ml-explore/mlx)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97_Hugging_Face-weights-yellow)](https://huggingface.co/MedMLX/segresnet-renal-cect-mlx)
[![MedMLX](https://img.shields.io/badge/MedMLX-runtime-A51C30)](https://github.com/MedMLX/medmlx)

[Weights](https://huggingface.co/MedMLX/segresnet-renal-cect-mlx) · [Upstream bundle](https://github.com/Project-MONAI/model-zoo/tree/dev/models/renalStructures_CECT_segmentation) · [MedMLX](https://github.com/MedMLX/medmlx)

<img src="https://raw.githubusercontent.com/MedMLX/segresnet-mlx/main/docs/example.png" alt="Venous-phase CT of a left kidney with a renal tumor, the predicted renal labels on that slice, and a 3D view of the predicted kidney, tumor, vessels and collecting system" width="100%">

<sub>Three-phase CT from <a href="https://doi.org/10.7937/k9/tcia.2018.oblamn27">CPTAC-CCRCC</a> case C3N-00380 (The Cancer Imaging Archive, CC BY 4.0), segmented on a crop around each kidney. The 3D view smooths surfaces and hides fragments under 0.5 mL and labels more than 20 mm from the kidney.</sub>

</div>

| Input | Output |
| :--- | :--- |
| Arterial, venous and excretory CT · NIfTI | Renal labels on the input grid |

| Model | Weights download | Peak MLX memory | Measured on |
| :--- | ---: | ---: | :--- |
| `segresnet-renal-cect` | 76 MB | 4.0 GB | 478×277×161 three-phase CT |

## Install

```bash
uv venv --python 3.13
source .venv/bin/activate
uv pip install "segresnet-mlx>=0.4.0"
```

## Run

```python
from segresnet_mlx.medmlx import renal_cect

result = renal_cect(
    arterial_path="arterial.nii.gz",
    venous_path="venous.nii.gz",
    excretory_path="excretory.nii.gz",
    output_dir="out",
)
print(result["outputs"]["mask"])
```

Weights download from [`MedMLX/segresnet-renal-cect-mlx`](https://huggingface.co/MedMLX/segresnet-renal-cect-mlx)
on first use; `medmlx setup segresnet-renal-cect` downloads ahead of time. For local
weights, pass `weights_path` as a folder containing `segresnet-renal-cect.safetensors`
or as the file itself.

### Inputs and outputs

| Argument | Meaning |
| --- | --- |
| `arterial_path`, `venous_path`, `excretory_path` | Arterial, venous and excretory CT in NIfTI, on one voxel grid |
| `output_dir` | Directory for `renalStructures_CECT_segmentation_mask.nii.gz`; created if missing |
| `weights_path` | Optional `.safetensors` file or a folder containing it |
| `overwrite` | `False` by default; an existing mask is replaced only when `True` |
| `arithmetic_profile` | `torch-order-fp32` (default), `compensated-fp32` or `compensated-fp48` |
| `streaming` | `False` by default; `True` keeps blending intermediates and logits in temporary mapped files |
| `scratch_dir` | Parent directory for the streaming files; defaults to the system temporary directory |

Inputs are finite 3D volumes on one voxel grid, with spatial units in millimetres or
unspecified. The mask is written on the input grid as the NIfTI reader corrects it.

The model was trained on CT cropped to the kidney region. Crop around each kidney,
with a margin of about 15 mm, before running it: on a wider field of view it can label
vertebral bodies or liver as neoplasm.

The mask is float32 with labels 1 artery, 2 vein, 3 ureter, 4 neoplasm and
5 parenchyma; where classes overlap, the lower label wins.

| `arithmetic_profile` | Arithmetic |
| :--- | :--- |
| `torch-order-fp32` (default) | The default float32 profile |
| `compensated-fp32` | Convolution and GroupNorm accumulate in paired Metal floats, with float32 layer outputs |
| `compensated-fp48` | The pairs carry through the hidden layers and stitching runs in float64 on the host |

The runner returns `{"outputs": {"mask": path}, "details": {...}}`, with run metadata in `details`.
Invalid inputs raise `InvalidInputError`; missing or mismatched weights raise
`AssetNotReadyError`; inference failures raise `ModelExecutionError`.

Streaming files are removed on success and failure. The final label volume stays in
memory. `run(bundle="renalStructures_CECT_segmentation", weights_path=...)` takes the
same inputs with an explicit `.safetensors` file.

### Python API

```python
from pathlib import Path

import numpy as np
from segresnet_mlx import BUNDLE_CONFIGS, SegResNetSpec, forward_patch, load_model, predict_logits

spec = SegResNetSpec(3, 6, 32, upsample_mode="deconv")  # renal CECT network
network = load_model(Path("segresnet-renal-cect.safetensors"), spec=spec)
patch = np.zeros((1, 3, 32, 32, 32), dtype=np.float32)
logits = forward_patch(network, patch)  # float32 NCDHW -> float32 NumPy logits
config = BUNDLE_CONFIGS["renalStructures_CECT_segmentation"]
volume_logits = predict_logits(patch, config, network=network)  # stitched sliding windows
```

- `forward_patch` needs spatial sizes divisible by `2 ** len(spec.blocks_up)`.
- `predict_logits` takes preprocessed arrays of any size and stitches the bundle's
  sliding windows. Neither normalizes intensities, applies activations or maps labels.
- `load_model` checks tensor names, shapes and finite float32 values against
  `required_shapes(spec)`.
- `WindowTelemetry` and `BundlePredictor` record bounded progress and timing.
- Inference requires Apple Silicon with Metal.

## Use with MedMLX

Use the [MedMLX MCP server](https://github.com/MedMLX/medmlx/blob/main/docs/mcp.md) to run `segresnet-renal-cect` from an MCP client. Weights download on first use.

```sh
uv tool install --python 3.13 "medmlx[segment,mcp]"
MEDMLX="$(uv tool dir --bin --color never)/medmlx"
mkdir -p "$HOME/medmlx-work"
claude mcp add --scope user medmlx -- "$MEDMLX" serve --root "$HOME/medmlx-work"
```

For Codex, use `codex mcp add medmlx -- "$MEDMLX" serve --root "$HOME/medmlx-work"`. Other MCP clients take the same command and arguments. The same install provides the `medmlx` CLI, for example `medmlx describe segresnet-renal-cect`.

> **For AI agents:** prefer the MedMLX MCP tools (`list_models`, `describe_model`, `model_readiness`, `run_model`, `get_job`) over writing Python against this package.

## License

[Apache-2.0](https://github.com/MedMLX/segresnet-mlx/blob/v0.4.0/LICENSE). See [third-party notices](https://github.com/MedMLX/segresnet-mlx/blob/v0.4.0/THIRD_PARTY_NOTICES.md). Model weights retain their publisher terms.
