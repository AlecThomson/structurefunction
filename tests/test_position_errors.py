from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord

from structurefunction import (
    AxisPositionErrors,
    CircularPositionError,
    SFResult,
    StencilOptions,
    structure_function,
)
from structurefunction._kernels.brute import brute_pairs
from structurefunction._kernels.common import new_accumulator
from structurefunction._units import position_covariance, positions
from structurefunction.geometry import tangent_basis

from conftest import make_catalogue

if TYPE_CHECKING:
    from structurefunction.options import PositionErrors
    from structurefunction.types import FloatArray

EDGES = np.geomspace(0.05, 2.0, 7) * u.deg


def run(
    cat_seed: int,
    pos: PositionErrors | None,
    tol: float = 0.0,
    n_point: Literal[2, 3] = 2,
) -> SFResult[FloatArray]:
    cat = make_catalogue(400, seed=cat_seed, size_deg=4.0)
    return structure_function(
        cat.values,
        cat.errors,
        cat.coords,
        n_point=n_point,
        bins=EDGES,
        patches=4,
        bin_tolerance=tol,
        pos_errors=pos,
        stencil=StencilOptions(mode="exact"),
    )


def test_zero_errors_match_no_errors() -> None:
    base = run(1, None)
    zero = run(1, CircularPositionError(0 * u.arcsec))
    np.testing.assert_allclose(zero.sf, base.sf, rtol=1e-12)


def test_circular_equals_equal_axes() -> None:
    circ = run(2, CircularPositionError(30 * u.arcsec))
    axes = run(2, AxisPositionErrors(30 * u.arcsec, 30 * u.arcsec))
    np.testing.assert_allclose(axes.sf, circ.sf, rtol=1e-12)


def test_ra_conventions_and_time_units_agree() -> None:
    cat = make_catalogue(300, seed=3, size_deg=4.0)
    lat = cat.lat + 70.0  # high declination, where cos(dec) matters
    c = SkyCoord(cat.lon * u.deg, lat * u.deg)
    on_sky = 20 * u.arcsec
    coord = (on_sky / np.cos(np.radians(lat))).to(u.arcsec)
    coord_time = coord.to_value(u.arcsec) / 15.0 * u.s  # seconds of RA

    def go(err: AxisPositionErrors) -> SFResult[FloatArray]:
        return structure_function(
            cat.values,
            None,
            c,
            bins=EDGES,
            patches=1,
            error_method="none",
            pos_errors=err,
            bin_tolerance=0.0,
        )

    a = go(AxisPositionErrors(on_sky, 10 * u.arcsec, "on_sky"))
    b = go(AxisPositionErrors(coord, 10 * u.arcsec, "coordinate"))
    t = go(AxisPositionErrors(coord_time, 10 * u.arcsec, "coordinate"))
    np.testing.assert_allclose(b.sf, a.sf, rtol=1e-10)
    np.testing.assert_allclose(t.sf, a.sf, rtol=1e-10)


def test_tree_matches_brute_with_errors() -> None:
    cat = make_catalogue(400, seed=4, size_deg=4.0)
    err = CircularPositionError(2 * u.arcmin)
    tree = structure_function(
        cat.values,
        cat.errors,
        cat.coords,
        bins=EDGES,
        patches=1,
        error_method="none",
        bin_tolerance=0.0,
        pos_errors=err,
    )
    pos = positions(cat.coords)
    east, north = tangent_basis(pos.lon, pos.lat)
    pcov = np.column_stack(position_covariance(err, pos))
    acc = brute_pairs(
        pos.xyz,
        cat.values,
        cat.errors**2,
        np.ones_like(cat.values),
        np.zeros(cat.values.size, np.intp),
        EDGES.to_value(u.rad),
        new_accumulator(2, EDGES.size - 1, 1),
        angular=True,
        east=east,
        north=north,
        pcov=pcov,
        use_pos=True,
    )
    w = acc[2, :, 0, 0]
    sf = (acc[2, :, 0, 1] - acc[2, :, 0, 2]) / w
    np.testing.assert_allclose(tree.sf, sf, rtol=1e-10)
    np.testing.assert_allclose(tree.count, acc[2, :, 0, 4], rtol=1e-10)


def test_soft_binning_matches_monte_carlo_jitter() -> None:
    rng = np.random.default_rng(9)
    cat = make_catalogue(300, seed=5, size_deg=2.0)
    # The Gaussian approximation needs sigma_sep << separation.
    sigma = 1 * u.arcmin
    soft = structure_function(
        cat.values,
        None,
        cat.coords,
        bins=EDGES,
        patches=1,
        error_method="none",
        pos_errors=CircularPositionError(sigma),
        bin_tolerance=0.0,
    )
    counts = []
    for _ in range(200):
        dl = rng.normal(0, sigma.to_value(u.deg), cat.lon.size)
        db = rng.normal(0, sigma.to_value(u.deg), cat.lon.size)
        c = SkyCoord(
            (cat.lon + dl / np.cos(np.radians(cat.lat))) * u.deg,
            (cat.lat + db) * u.deg,
        )
        r = structure_function(
            cat.values,
            None,
            c,
            bins=EDGES,
            patches=1,
            error_method="none",
            bin_tolerance=0.0,
        )
        counts.append(r.count)
    mc = np.mean(counts, axis=0)
    np.testing.assert_allclose(soft.count, mc, rtol=0.03)


def test_three_point_soft_binning_runs() -> None:
    res = run(6, CircularPositionError(1 * u.arcmin), n_point=3)
    assert np.all(np.isfinite(res.sf[res.count > 0]))
