# Technical notes

## Device-side window stitching

`predict_logits(..., accumulation="mlx")` keeps the padded input, window patches
and an ROI-depth FP32 score/count slab on Metal and exports completed planes
only. The default `accumulation="numpy"` blends on the host as before. The MLX
route needs `sliding_window_inference(accumulation="mlx")` from `medmlx-core`
0.3.1 or later; the default route works with 0.3.0.

Comparison on real CT (HCC-TACE-Seg HCC_063 contrast CT, two 3-channel 96^3 crops
with liver voxels retained, the recorded decoder-network fixture weights,
Gaussian blending, overlap 0.25, ROI 32^3 with window batch 2, and ROI 64^3 with
window batch 1):

| Comparison | Result |
| :--- | :--- |
| `accumulation="mlx"` vs `"numpy"` logits | bit-identical (max abs difference 0) |
| `"numpy"` vs the previous release's predictor | bit-identical |
| Argmax masks, `"mlx"` vs `"numpy"` | 0 differing voxels of 1,769,472 |
| Six recorded MONAI predictor fixtures, `"mlx"` vs `"numpy"` | bit-identical; both within 1e-6 of MONAI (largest 8.9e-8) |

Peak MLX memory on the 96^3 crops was 59 MB with Metal slabs against 28 MB with
host blending at ROI 32^3, and 156 MB against 116 MB at ROI 64^3. The slab adds a fixed device allocation of one ROI-depth of scores and counts.
