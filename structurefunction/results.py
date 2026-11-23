"""Result container for structure functions."""

from __future__ import annotations

from dataclasses import dataclass

import astropy.units as u

from structurefunction.types import BoolArray, ErrorMethod, FloatArray


@dataclass(frozen=True, slots=True)
class SFResult:
    """A binned, noise-debiased structure function and its uncertainties.

    Values are plain arrays in ``sf_unit`` (the input unit squared, or ``None``
    for unitless input); ``result.sf * result.sf_unit`` gives a Quantity.
    See :mod:`structurefunction.io` for reading and writing.
    """

    sf: FloatArray
    """Noise-debiased structure function per bin."""
    sf_err: FloatArray
    """Square root of the covariance diagonal (NaN without resampling)."""
    noise_bias: FloatArray
    """Noise bias subtracted per bin, ``<sum_i c_i^2 sigma_i^2>``."""
    saturation: float
    """Large-scale plateau, ``sum c^2 (Var(f) - <sigma^2>)`` (noise removed)."""
    sf_unit: u.UnitBase | None
    """Unit of ``sf``, ``sf_err``, ``noise_bias`` and ``saturation``, or ``None``."""
    cov: FloatArray | None
    """Bin-to-bin covariance of ``sf`` (in ``unit**4``), or ``None``."""
    separation: u.Quantity
    """Weighted mean separation of the contributing pairs or triplets."""
    bin_edges: u.Quantity
    """Separation bin edges."""
    count: FloatArray
    """Number of contributing pairs or triplets per bin."""
    weight: FloatArray
    """Sum of stencil weights per bin."""
    n_point: int
    """Number of points in the stencil."""
    n_regions: int
    """Number of spatial regions used for resampling."""
    error_method: ErrorMethod
    """How ``cov`` was estimated."""
    min_count: int
    """Bins with fewer contributions are flagged as unreliable."""
    reliable: BoolArray
    """Bins with at least ``min_count`` contributions and a finite estimate."""
    region_size: u.Quantity | None
    """Typical diameter of the resampling regions (``None`` if unknown)."""
