"""Output replacement must preserve every source modality."""

from pathlib import Path

import pytest
from medmlx_core.errors import InvalidInputError

from segresnet_mlx import medmlx


@pytest.mark.parametrize("alias_kind", ["direct", "symlink", "hardlink", "separate"])
def test_output_aliases_are_rejected_before_reading_the_scan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, alias_kind: str
) -> None:
    """Reject output aliases of the input, including symbolic and hard links."""
    bundle = "wholeBody_ct_segmentation"
    output = tmp_path / "out"
    output.mkdir()
    destination = output / f"{bundle}_mask.nii.gz"
    source = destination if alias_kind == "direct" else tmp_path / "scan.nii.gz"
    source.write_bytes(b"original scan")
    if alias_kind == "symlink":
        destination.symlink_to(source)
    elif alias_kind == "hardlink":
        destination.hardlink_to(source)
    weights = tmp_path / "weights.safetensors"
    weights.write_bytes(b"unused weights")

    def stop_before_loading(*_args: object, **_kwargs: object) -> None:
        """Signal that input loading was reached for a distinct output path."""
        raise RuntimeError("input loading reached")

    monkeypatch.setattr(medmlx, "load_inputs", stop_before_loading)
    error = RuntimeError if alias_kind == "separate" else InvalidInputError
    message = "input loading reached" if alias_kind == "separate" else "must differ"
    with pytest.raises(error, match=message):
        medmlx.run(
            bundle=bundle,
            image_path=str(source),
            weights_path=str(weights),
            output_dir=str(output),
            overwrite=True,
        )
    assert source.read_bytes() == b"original scan"
    if alias_kind != "separate":
        assert destination.samefile(source)
