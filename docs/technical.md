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

## Renal runner on a kidney-region crop

The measurements used `segresnet-renal-cect` on an Apple M1 Max with 64 GB of
memory, macOS 27.2.1, MLX 0.32.4, and `medmlx-core` 0.3.1. The baseline was
`49a86c7`. A 270×160×63 crop of the HCC-TACE-Seg HCC_063 contrast CT was supplied
to all three phase inputs. The crop uses voxel ranges `[108:378, 217:377, 13:76]`
on the prepared LPS grid. The runner resampled it to 290×172×195 and used 36
windows with the default `torch-order-fp32` profile, 96³ ROI, batch size 4, and
overlap 0.25.

Perfkit alternated fresh workers over ten paired rounds for loading and six for
each transform. Preprocessing used preloaded voxels. Postprocessing used the
same baseline model logits and inverse geometry in every worker. Each cell shows
the median warm time in milliseconds and peak process footprint in Perfkit MB.

| Measured stage | Baseline (ms / MB) | Corner reuse (ms / MB) | Single NIfTI read (ms / MB) |
| :--- | ---: | ---: | ---: |
| Load three inputs | 162.8 / 94.3 | — | 92.2 / 93.4 |
| `preprocess_renal` | 1295.1 / 612.1 | 969.4 / 618.9 | 1296.3 / 611.4 |
| `postprocess_renal` | 617.1 / 485.4 | 434.6 / 547.1 | 619.4 / 482.3 |

One decode per NIfTI cut loading time by 43.4%, with a paired interval of
43.1–43.5% faster. The paired intervals for its warm preprocessing and
postprocessing times were −2.5% to +1.3% and −0.5% to +2.8%. Corner reuse cut
both transform times and raised peak postprocessing footprint by 12.7%, with a
paired interval of +12.0% to +15.1%. The follow-up full-run comparison qualified
corner reuse for adoption on both CT sizes. A dash indicates an unmeasured
comparison.

Full-run calls included loading weights and inputs, both transforms, inference,
and compressed mask writing. Perfkit marked the time and footprint comparison
diagnostic. Its six paired rounds cover part of the five-variant order schedule,
and the standard timing output omits the MLX allocator peak. A separate,
balanced ten-round pair read `details.peak_mlx_bytes` after a single run in each
fresh worker.

| Measured full-run variant | First call (s) | Warm call (s) | Peak process footprint (Perfkit MB) | Peak MLX (decimal MB) |
| :--- | ---: | ---: | ---: | ---: |
| Baseline, host blending | 5.884 | 5.794 | 7912.7 | 4056.0 |
| Corner reuse | 5.355 | 5.305 | 8003.7 | 4056.0 |
| Single NIfTI read | 5.814 | 5.742 | 7932.1 | 4056.0 |
| Both host patches | 5.288 | 5.238 | 8002.0 | 4056.0 |
| Baseline with Metal blending | 5.660 | 5.571 | 8686.6 | 4436.3 |

Each host patch produced bit-identical loaded data, preprocessing, inverse
resampling, and masks on the recorded real-CT fixtures and on the larger crop
with its original affine and with 0.2 mm of x/z shear per slice. Saved masks
matched byte for byte on the timed crop. Metal blending matched the host logits
and labels bit for bit on that crop and added 380.3 MB of peak MLX memory.

## MLX buffer cache during inference

The uncapped runner was sampled on the same HCC_063 crop with the single NIfTI
read and default host stitching. MLX allocator counters and macOS physical
footprint were sampled together at stage boundaries and about every 4 ms during
the run on the Apple M1 Max with MLX 0.32.4. Memory in the table is in decimal MB.

| Stage | Active MLX | Cached MLX | Peak active MLX | Current process footprint |
| :--- | ---: | ---: | ---: | ---: |
| After loading CT and weights | 75.9 | 57.1 | 133.0 | 322.1 |
| After preprocessing | 75.9 | 57.1 | 133.0 | 837.9 |
| End of inference, before clearing the cache | 75.9 | 7118.7 | 4056.0 | 8200.5 |
| After inference | 75.9 | 0.0 | 4056.0 | 7258.0 |
| After postprocessing | 75.9 | 0.0 | 4056.0 | 1177.7 |

The 8.20 GB process high-water mark coincided with 7.12 GB of cached MLX buffers
(87%) and 76 MB of live MLX allocations, mainly model weights. The runner also
held three source CT arrays (32.7 MB total), the preprocessed input (116.7 MB),
and the stitched logits (233.4 MB). The host stitcher's 38.9 MB count grid had
already gone out of scope. The roughly 623 MB difference after subtracting the
available allocator counters and live arrays includes other process and driver
memory. These counters do not independently attribute that remainder. The MLX
cache counter dropped immediately at the clear, while macOS footprint fell
through postprocessing.

With the cache disabled during inference, the same probe recorded a 4.00 GB
process peak. At the highest simultaneous sample, 2.87 GB was active MLX memory
and the MLX cache was empty. The host stitcher uses a 233.4 MB score volume
and a 38.9 MB count grid. After preprocessing, the three decoded CT arrays can
be released once the runner retains the arterial shape and affine for validation
and writing. Those buffers total 32.7 MB on the crop and 311.4 MB on the full CT.
On the 550×550×307 full-CT inference grid, the preprocessed input, scores and
count grid occupy 1.11 GB, 2.23 GB and 371 MB respectively. The effect of releasing
the source CTs earlier on peak footprint was not measured.

### Cache limit sweep

The renal runs used `medmlx-core` 0.3.1 and cached checkpoint revision `6b9f350`.
They supplied one scan to all three phase inputs, using the 270×160×63 crop and
the complete 512×512×99 HCC_063 contrast CT. At 0.8 mm, the full scan resampled
to 550×550×307 and used 256 windows with the same settings as the crop.

The 240×240×155 `BRATS_001` case used four Medical Segmentation Decathlon brain
MRI sequences (T1gd, T1w, T2w and FLAIR). BraTS was removed in `segresnet-mlx`
0.4.0, so those runs used the last release that registered `segresnet-brats`
(0.3.6), its locked `medmlx-core` 0.2.6, and cached checkpoint revision `dd339bb`.
It used one 240×240×160 window, batch size 1 and the default `torch-order-fp32`
profile under MLX 0.32.4.

Perfkit 0.3.0.dev0 used six balanced rounds per input with one uncounted
warm-up process per variant. The six variants differed only in the cache limit
set for inference through `mx.set_cache_limit`. Each fresh process imported the
runtime, loaded inputs and weights, ran preprocessing, inference and
postprocessing, and wrote the compressed NIfTI mask. `/usr/bin/time -p` measured
process wall time outside the GPU lock wait. Inference time came from the runner's `details.seconds`,
active MLX peak from `details.peak_mlx_bytes`, and process peak from macOS's
lifetime physical footprint counter. The table gives medians across the counted
processes and Perfkit's 96.9% paired intervals for wall-time change. Memory units
are decimal MB.

| Input | Cache limit | Wall (s) | Inference (s) | Peak MLX (MB) | Peak footprint (MB) | Wall-time change, 96.9% interval |
| :--- | :--- | ---: | ---: | ---: | ---: | :--- |
| HCC_063 crop | Uncapped | 5.955 | 3.633 | 4056.0 | 8189.6 | — |
| HCC_063 crop | 0 | 7.030 | 4.708 | 4056.0 | 3995.7 | +17.5% to +18.8% |
| HCC_063 crop | 256 MiB | 7.000 | 4.678 | 4056.0 | 4013.8 | +17.3% to +18.2% |
| HCC_063 crop | 512 MiB | 6.710 | 4.393 | 4056.0 | 5081.4 | +12.4% to +13.1% |
| HCC_063 crop | 1 GiB | 6.545 | 4.223 | 4056.0 | 5592.8 | +9.4% to +11.5% |
| HCC_063 crop | 2 GiB | 6.330 | 4.003 | 4056.0 | 6430.3 | +5.9% to +7.1% |
| HCC_063 full CT | Uncapped | 46.235 | 25.500 | 4056.0 | 14685.0 | — |
| HCC_063 full CT | 0 | 54.760 | 34.168 | 4056.0 | 10093.3 | +17.9% to +19.0% |
| HCC_063 full CT | 256 MiB | 54.615 | 34.001 | 4056.0 | 10109.5 | +17.6% to +19.0% |
| HCC_063 full CT | 512 MiB | 52.430 | 31.856 | 4056.0 | 11251.2 | +12.7% to +14.3% |
| HCC_063 full CT | 1 GiB | 51.035 | 30.460 | 4056.0 | 11830.7 | +9.9% to +11.0% |
| HCC_063 full CT | 2 GiB | 49.230 | 28.624 | 4056.0 | 12829.2 | +5.9% to +7.0% |
| BRATS_001 | Uncapped | 1.465 | 0.636 | 5084.9 | 12242.9 | — |
| BRATS_001 | 0 | 1.650 | 0.825 | 5084.9 | 4171.8 | +11.5% to +13.0% |
| BRATS_001 | 256 MiB | 1.620 | 0.790 | 5084.9 | 4891.4 | +10.1% to +11.6% |
| BRATS_001 | 512 MiB | 1.610 | 0.780 | 5084.9 | 4891.1 | +9.5% to +10.3% |
| BRATS_001 | 1 GiB | 1.570 | 0.739 | 5084.9 | 6087.9 | +6.8% to +8.2% |
| BRATS_001 | 2 GiB | 1.555 | 0.727 | 5084.9 | 7084.6 | +5.4% to +6.2% |

All 36 counted masks per input matched byte for byte. The 2 GiB limit was the
fastest capped variant on all three inputs and lowered process footprint by
21.5% on the crop, 12.6% on the full CT and 42.1% on the MRI. Even at that limit,
each full-run wall-time interval was entirely above the allowed +3%. SegResNet
retains MLX's default cache limit.

### Corner reuse, paired full-run comparison

The follow-up used the single-read renal runner and MLX's default cache limit
on the same M1 Max. Perfkit alternated the baseline and the corner-reuse patch
over eight counted rounds per input, using the full-process timing and memory
counters above.

| Input | Wall (s), baseline → patch | Wall-time change, 99.2% interval | Peak footprint (MB), baseline → patch | Footprint change | Peak MLX (MB), baseline → patch |
| :--- | ---: | :--- | ---: | ---: | ---: |
| HCC_063 crop | 5.960 → 5.445 | −9.0% to −8.4% | 8190.7 → 8194.8 | +0.05% | 4056.0 → 4056.0 |
| HCC_063 full CT | 46.135 → 40.920 | −11.7% to −11.0% | 14682.0 → 14694.2 | +0.08% | 4056.0 → 4056.0 |

All 16 counted masks per input matched byte for byte. Both variants on both
inputs recorded the same MLX peak range of 4,056,027,300 to 4,056,027,820 bytes.
The largest process footprint increases in individual pairs were 0.12% on the
crop and 0.33% on the full CT, below the 2% limit. Median inference time stayed
at 3.63 s on the crop and 25.46 s on the full CT.

### Restoring from cropped window logits

A volume smaller than the 96-voxel ROI on one axis is padded for inference, and
its logits come back as a cropped view of the padded array. `ndarray.take` makes
a non-contiguous source C-contiguous before it gathers, so each resampling block
copied the whole logits volume to read a few planes. [NumPy 2.5.4 take implementation](https://github.com/numpy/numpy/blob/v2.5.4/numpy/_core/src/multiarray/item_selection.c#L223-L237)
For a non-contiguous source the resampler now gathers each block's depth planes
from that block's own range of planes. Contiguous logits take the same path as
before.

The comparison used slices 40 to 59 of the HCC_063 contrast CT (512×512×20 at
0.86×0.86×2.5 mm) for all three phase inputs on the same M1 Max. The runner
resampled the slab to 550×550×60 and padded it to 96 planes. Perfkit alternated
the corner-reuse runner with and without the plane-range gather over six
counted rounds, using the full-process timing and memory counters above.

| Measure | Corner reuse | With plane-range gather | 96.9% paired interval |
| :--- | ---: | ---: | :--- |
| Wall (s) | 13.115 | 9.475 | −27.8% to −27.3% |
| Postprocessing and writing (s) | 4.557 | 0.956 | −79.2% to −78.7% |
| Inference (s) | 6.186 | 6.182 | −0.3% to +0.1% |
| Peak footprint (MB) | 9596.2 | 9594.0 | −0.1% to +0.4% |
| Peak MLX (MB) | 4056.0 | 4056.0 | unchanged |

All seven written masks per variant, including the warm-up, matched byte for
byte.
