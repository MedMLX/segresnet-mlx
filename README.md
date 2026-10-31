# segresnet-mlx

Segment brain tumors, renal structures and whole-body CT anatomy with MONAI SegResNet models in native MLX.

[Docs](https://github.com/MedMLX/segresnet-mlx/blob/main/docs/technical.md) · [BraTS weights](https://huggingface.co/MedMLX/segresnet-brats-mlx) · [Renal weights](https://huggingface.co/MedMLX/segresnet-renal-cect-mlx) · [MedMLX](https://github.com/MedMLX)

| Input | Output |
| :--- | :--- |
| MRI sequences or CT volumes | Segmentation on the input grid |

**Apple Silicon · Metal · Python 3.13 / 3.14** · [uv](https://docs.astral.sh/uv/)

## Install

```bash
uv venv --python 3.13
source .venv/bin/activate
uv pip install "segresnet-mlx>=0.3.2"
```

## Run

```python
from segresnet_mlx.medmlx import brats

result = brats(
    t1c_path="t1c.nii.gz", t1_path="t1.nii.gz", t2_path="t2.nii.gz",
    flair_path="flair.nii.gz", output_dir="out",
)
print(result["outputs"]["mask"])
```

The registered models are `segresnet-brats` and `segresnet-renal-cect`. Their `brats`
and `renal_cect` runners select the matching bundle internally and download
`MedMLX/segresnet-brats-mlx` or `MedMLX/segresnet-renal-cect-mlx` when `weights_path`
is omitted. An explicit folder contains `segresnet-brats.safetensors` for BraTS or
`segresnet-renal-cect.safetensors` for renal CECT. The renal runner takes
`arterial_path`, `venous_path` and `excretory_path`. Keep each model's input volumes
on one voxel grid.

The standalone `run(bundle=..., weights_path=...)` interface remains available for
explicit local checkpoints, including the whole-body CT variants. Published native
weights are also available for [renal CECT](https://huggingface.co/MedMLX/segresnet-renal-cect-mlx)
and [whole-body CT](https://huggingface.co/MedMLX/segresnet-wholebody-mlx).

Downloaded commits are returned in `details.weights_revision`; explicit local
weights report `null` unless the folder carries a MedMLX download revision.

## License

[Apache-2.0](https://github.com/MedMLX/segresnet-mlx/blob/main/LICENSE). See [third-party notices](https://github.com/MedMLX/segresnet-mlx/blob/main/THIRD_PARTY_NOTICES.md). Model weights retain their publisher terms.

For research and education. Outputs require human review.
