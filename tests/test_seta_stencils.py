"""Seta et al. (2023) Fig. 5, on irregularly sampled points.

A 2D Gaussian random field with shell-averaged power ``P(k) ~ k^-beta`` has
``SF ~ r^(beta - 1)`` while the stencil can resolve it: up to ``r^2`` for two
points and ``r^4`` for three (their Eqs. 3 and 7).
"""

from __future__ import annotations

from typing import Literal

import astropy.units as u
import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from scipy.interpolate import RegularGridInterpolator

from structurefunction import StencilOptions, structure_function


def field(beta: float, n_grid: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    k = np.fft.fftfreq(n_grid)
    kk = np.hypot(*np.meshgrid(k, k, indexing="ij"))
    kk[0, 0] = np.inf
    # 2D density |F|^2 ~ k^-(beta + 1) gives shell-averaged power ~ k^-beta.
    amp = kk ** (-(beta + 1) / 2)
    noise = rng.normal(size=(n_grid, n_grid)) + 1j * rng.normal(size=(n_grid, n_grid))
    f = np.fft.ifft2(amp * noise).real
    return f / f.std()


def slope(beta: float, n_point: Literal[2, 3]) -> float:
    n_grid = 1024
    size_deg = 10.0
    f = field(beta, n_grid, seed=1)
    axis = np.linspace(0, size_deg, n_grid)
    interp = RegularGridInterpolator((axis, axis), f)
    rng = np.random.default_rng(2)
    n = 30000
    x = rng.uniform(0, size_deg, n)
    y = rng.uniform(-size_deg / 2, size_deg / 2, n)
    vals = np.asarray(interp(np.column_stack((x, y + size_deg / 2))), dtype=np.float64)
    c = SkyCoord(x * u.deg, y * u.deg)
    # Well above the grid scale, well below the box.
    edges = np.geomspace(0.1, 0.5, 5) * u.deg
    res = structure_function(
        vals,
        None,
        c,
        n_point=n_point,
        bins=edges,
        patches=1,
        error_method="none",
        stencil=StencilOptions(angle_tol_deg=5.0, sep_tol=0.05, mode="exact"),
    )
    return float(np.polyfit(np.log(res.separation.value), np.log(res.sf), 1)[0])


@pytest.mark.slow
@pytest.mark.parametrize(
    ("beta", "n_point", "expected"),
    [
        (2.0, 2, 1.0),  # resolved by both stencils
        (2.0, 3, 1.0),
        (4.0, 2, 2.0),  # two points saturate at r^2
        (4.0, 3, 3.0),  # three points still resolve r^(beta - 1)
    ],
)
def test_stencil_slopes(beta: float, n_point: Literal[2, 3], expected: float) -> None:
    assert abs(slope(beta, n_point) - expected) < 0.3
