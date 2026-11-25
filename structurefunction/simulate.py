"""Mock catalogues for testing and calibration."""

from __future__ import annotations

from typing import TYPE_CHECKING

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from scipy.interpolate import RegularGridInterpolator

if TYPE_CHECKING:
    from structurefunction.types import FloatArray


def gaussian_random_field(
    beta: float, n_grid: int, rng: np.random.Generator | int | None = None
) -> FloatArray:
    """A periodic 2D Gaussian random field with power spectrum ``P(k) ~ k^-beta``.

    ``P(k)`` is the shell-averaged (1D) spectrum, as in Seta et al. (2023), so
    the field's structure function goes as ``r^(beta - 1)``. The field is
    normalised to unit variance.

    Args:
        beta: Spectral slope.
        n_grid: Grid size along each axis.
        rng: Random generator or seed.
    """
    gen = np.random.default_rng(rng)
    k = np.fft.fftfreq(n_grid)
    k_magnitude = np.hypot(*np.meshgrid(k, k, indexing="ij"))
    k_magnitude[0, 0] = np.inf  # no mean
    # A 2D density |F|^2 ~ k^-(beta + 1) gives a shell-averaged k^-beta.
    amplitude = k_magnitude ** (-(beta + 1) / 2)
    noise = gen.normal(size=(n_grid, n_grid)) + 1j * gen.normal(size=(n_grid, n_grid))
    field = np.fft.ifft2(amplitude * noise).real
    return np.asarray(field / field.std(), dtype=np.float64)


def sample_field(
    field: FloatArray,
    size: u.Quantity,
    n_sources: int,
    rng: np.random.Generator | int | None = None,
) -> tuple[SkyCoord, FloatArray]:
    """Sample a square field at random positions on a region of sky.

    The field covers ``size`` x ``size`` centred on (lon, lat) = (size/2, 0).
    The region should be small enough that the sky is close to flat.

    Returns:
        The source positions and the interpolated field values.
    """
    gen = np.random.default_rng(rng)
    size_deg = float(size.to_value(u.deg))
    axis = np.linspace(0.0, size_deg, field.shape[0])
    interpolate = RegularGridInterpolator((axis, axis), field)
    lon = gen.uniform(0.0, size_deg, n_sources)
    lat_offset = gen.uniform(0.0, size_deg, n_sources)
    values = np.asarray(interpolate(np.column_stack((lon, lat_offset))), np.float64)
    coords = SkyCoord(lon * u.deg, (lat_offset - size_deg / 2) * u.deg)
    return coords, values
