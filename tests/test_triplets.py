from __future__ import annotations

import math

import astropy.units as u
import numpy as np
import pytest
from astropy.coordinates import SkyCoord

from structurefunction import StencilOptions, structure_function

from conftest import make_catalogue
from reference import sf3_exact, unit_vectors

EDGES = np.geomspace(0.2, 4.0, 7) * u.deg


def test_exact_matches_reference() -> None:
    cat = make_catalogue(250, seed=11)
    opts = StencilOptions(angle_tol_deg=12.0, sep_tol=0.15, mode="exact")
    res = structure_function(
        cat.values,
        cat.errors,
        cat.coords,
        n_point=3,
        bins=EDGES,
        weights=cat.weights,
        patches=4,
        stencil=opts,
    )
    sf, bias, cnt = sf3_exact(
        unit_vectors(cat.lon, cat.lat),
        cat.values,
        cat.errors,
        cat.weights,
        EDGES.to_value(u.rad),
        math.radians(12.0),
        0.15,
    )
    np.testing.assert_allclose(res.count, cnt, rtol=1e-12)
    np.testing.assert_allclose(res.sf, sf, rtol=1e-9)
    np.testing.assert_allclose(res.noise_bias, bias, rtol=1e-9)


def grid(step_deg: float, n_side: int) -> SkyCoord:
    lon, lat = np.meshgrid(
        np.arange(n_side) * step_deg, (np.arange(n_side) - n_side / 2) * step_deg
    )
    return SkyCoord(lon.ravel() * u.deg, lat.ravel() * u.deg)


def test_binned_is_subset_of_exact_on_grid() -> None:
    c = grid(0.1, 30)
    rng = np.random.default_rng(3)
    f = rng.normal(size=len(c))
    edges = np.array([0.09, 0.11, 0.19, 0.21]) * u.deg  # 1 and 2 grid steps
    exact, binned = (
        structure_function(
            f,
            None,
            c,
            n_point=3,
            bins=edges,
            patches=1,
            error_method="none",
            stencil=StencilOptions(1.0, 0.01, mode),
        )
        for mode in ("exact", "binned")
    )
    # Tiny tolerances on a regular grid: only axis-aligned stencils exist and
    # both modes find all of them.
    np.testing.assert_array_equal(binned.count[[0, 2]], exact.count[[0, 2]])
    np.testing.assert_allclose(binned.sf[[0, 2]], exact.sf[[0, 2]], rtol=1e-9)


def test_white_noise_saturates_at_six_variance() -> None:
    rng = np.random.default_rng(4)
    n = 5000
    c = SkyCoord(rng.uniform(0, 20, n) * u.deg, rng.uniform(-10, 10, n) * u.deg)
    f = rng.normal(0, 1.5, n)
    for mode in ("exact", "binned"):
        res = structure_function(
            f,
            None,
            c,
            n_point=3,
            bins=np.geomspace(0.5, 4, 5) * u.deg,
            patches=40,
            stencil=StencilOptions(mode=mode),
            rng=1,
        )
        assert np.all(np.abs(res.sf - 6 * np.var(f)) < 4 * res.sf_err), mode


def test_gradient_cancels_in_three_point() -> None:
    rng = np.random.default_rng(8)
    n = 4000
    lon = rng.uniform(0, 10, n)
    c = SkyCoord(lon * u.deg, rng.uniform(-5, 5, n) * u.deg)
    f = 5.0 * lon
    edges = np.geomspace(0.3, 2.0, 5) * u.deg
    d2 = structure_function(f, None, c, bins=edges, patches=1, error_method="none")
    d3 = structure_function(
        f,
        None,
        c,
        n_point=3,
        bins=edges,
        patches=1,
        error_method="none",
        stencil=StencilOptions(angle_tol_deg=5.0, sep_tol=0.05, mode="exact"),
    )
    # A pure gradient dominates D2 but only leaks into D3 through the
    # tolerances: suppressed by roughly (tolerance)^2.
    assert np.all(d3.sf < 0.02 * d2.sf)


def test_three_point_needs_sky_positions() -> None:
    c = SkyCoord([0, 1, 2] * u.deg, [0, 0, 0] * u.deg, [1, 2, 3] * u.pc)
    with pytest.raises(NotImplementedError):
        structure_function([1.0, 2.0, 3.0], None, c, n_point=3)
