"""Typed protocols for the NiBabel NIfTI, header and orientation calls used here."""

from __future__ import annotations

from collections.abc import Sequence
from os import PathLike
from typing import Protocol, cast

import nibabel
import numpy as np
from nibabel.filebasedimages import ImageFileError as ImageFileError
from reference_contracts import Float64Array, HostArray

type PathArg = str | PathLike[str]


class ArrayBuffer(Protocol):
    def __array__(
        self, dtype: np.dtype[np.generic] | None = None, copy: bool | None = None
    ) -> HostArray:
        """Expose image data through NumPy's array conversion protocol."""
        ...


class NiftiHeader(Protocol):
    def get_zooms(self) -> tuple[float, ...]:
        """Return the voxel spacing stored in the header."""
        ...
    def set_zooms(self, zooms: Sequence[float] | Float64Array) -> None:
        """Set voxel spacing in the header."""
        ...
    def __getitem__(self, key: str) -> HostArray:
        """Return a named NIfTI header field."""
        ...


class NiftiImage(Protocol):
    def to_bytes(self) -> bytes:
        """Serialize the image as NIfTI bytes."""
        ...

    shape: tuple[int, ...]
    affine: Float64Array
    dataobj: ArrayBuffer | HostArray
    header: NiftiHeader

    def __init__(self, dataobj: HostArray, affine: HostArray) -> None:
        """Create an image from voxel data and its affine."""
        ...
    def get_data_dtype(self) -> np.dtype[np.generic]:
        """Return the stored voxel-data dtype."""
        ...
    def get_sform(self) -> Float64Array:
        """Return the header's sform affine."""
        ...
    def get_qform(self) -> Float64Array:
        """Return the header's qform affine."""
        ...
    def set_sform(self, affine: Float64Array | None, code: int | None = None) -> None:
        """Set the header's sform affine and optional code."""
        ...
    def set_qform(self, affine: Float64Array | None, code: int | None = None) -> None:
        """Set the header's qform affine and optional code."""
        ...


class Orientations(Protocol):
    def axcodes2ornt(self, axcodes: str, labels: tuple[tuple[str, str], ...]) -> Float64Array:
        """Convert axis codes to an orientation array."""
        ...
    def ornt_transform(self, start_ornt: Float64Array, end_ornt: Float64Array) -> Float64Array:
        """Return the transform between two orientation arrays."""
        ...
    def inv_ornt_aff(self, ornt: Float64Array, shape: Sequence[int]) -> Float64Array:
        """Return the inverse voxel-coordinate affine for an orientation change."""
        ...
    def aff2axcodes(
        self, aff: Float64Array, labels: tuple[tuple[str, str], ...]
    ) -> tuple[str | None, ...]:
        """Return axis codes implied by an affine."""
        ...


class Nibabel(Protocol):
    Nifti1Image: type[NiftiImage]
    orientations: Orientations

    def load(self, filename: PathArg) -> NiftiImage:
        """Load an image from a NIfTI path."""
        ...
    def save(self, img: NiftiImage, filename: PathArg) -> None:
        """Write an image to a NIfTI path."""
        ...
    def io_orientation(self, affine: Float64Array) -> Float64Array:
        """Return the input-to-output orientation implied by an affine."""
        ...


# Bound to the real module; this runner admits only NIfTI images.
nib = cast(Nibabel, nibabel)
