Imaging tests use the converted HCC-TACE-Seg HCC_063 contrast CT and liver mask
at `~/.cache/medmlx/test-data/hcc_063_ct.nii.gz` and `hcc_063_liver.nii.gz`.
The real-input references are generated into `work/real-reference`, outside Git.
The renal runner uses the same CT for all three aligned channels; the controlled
fixture affines do not describe the original scan geometry.

Run the `record_network_references.py`, `record_predictor_references.py`,
`record_reader_reference.py`, and `record_renal_reference.py` scripts in an
environment with MONAI 1.6.0, Torch 2.14, and the project test dependencies.
The renal recorder takes `--bundle` pointing to the
`8ccbcd9573c004dc435d1913c0459ccd1a110ed1` bundle snapshot. Then run the
changed tests in `test_equivalence.py`, `test_predictor.py`, `test_transforms.py`,
`test_medmlx.py`, and `test_torch_oracles.py`. The last file needs the released
source and native weights through its documented environment variables.
