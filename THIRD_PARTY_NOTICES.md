# Third-party notices

MedMLX-authored additions are licensed under [Apache-2.0](LICENSE).

## MONAI SegResNet and model bundles

The native MLX network, inferer and transforms are adapted from
[MONAI](https://github.com/Project-MONAI/MONAI), with the numerical reference
pinned to revision `eccefc57550b111ed781d82249dfe77872a0e918` (MONAI 1.6.0).
Copyright (c) MONAI Consortium. These source portions retain Apache-2.0 terms;
the full license is included in [LICENSE](LICENSE).

Bundle definitions originate in the
[MONAI model zoo](https://github.com/Project-MONAI/model-zoo), revision
`db9eeeef7ba64ce2ced0230d9bd4c524f69b7f0c`. The source files and bundle versions
are identified in [scripts/upstream_reference.json](scripts/upstream_reference.json).
The port changes execution to native MLX and host NumPy image processing.

Pretrained checkpoints and input datasets retain their publisher terms.
Dependencies declared in [pyproject.toml](pyproject.toml) retain their respective
licenses independently of the package source license.

## Example image

`docs/example.png` shows model outputs over a brain MRI from the
[Medical Segmentation Decathlon](http://medicaldecathlon.com/). The source image
is shared under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/);
the repository's Apache-2.0 license does not relicense it.
