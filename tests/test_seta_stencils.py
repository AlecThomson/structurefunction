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

from structurefunction import StencilOptions, structure_function
from structurefunction.simulate import gaussian_random_field, sample_field


def slope(beta: float, n_point: Literal[2, 3]) -> float:
    field = gaussian_random_field(beta, 1024, rng=1)
    coords, values = sample_field(field, 10 * u.deg, 30000, rng=2)
    # Well above the grid scale, well below the box.
    edges = np.geomspace(0.1, 0.5, 5) * u.deg
    res = structure_function(
        values,
        None,
        coords,
        n_point=n_point,
        bins=edges,
        regions=1,
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
