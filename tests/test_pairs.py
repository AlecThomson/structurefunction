from __future__ import annotations

import astropy.units as u
import numpy as np
import pytest
from astropy.coordinates import SkyCoord, search_around_sky

from structurefunction import SFResult, structure_function

from conftest import Catalogue, make_catalogue
from reference import sf2, unit_vectors

EDGES = np.geomspace(0.1, 6.0, 9) * u.deg


def run(cat: Catalogue, tol: float = 0.0) -> SFResult:
    return structure_function(
        cat.values,
        cat.errors,
        cat.coords,
        bins=EDGES,
        weights=cat.weights,
        bin_tolerance=tol,
        regions=8,
        rng=0,
    )


def test_exact_tree_matches_reference(small: Catalogue) -> None:
    res = run(small)
    xyz = unit_vectors(small.lon, small.lat)
    sf, bias, cnt = sf2(
        xyz, small.values, small.errors, small.weights, EDGES.to_value(u.rad)
    )
    np.testing.assert_allclose(res.sf, sf, rtol=1e-10)
    np.testing.assert_allclose(res.noise_bias, bias, rtol=1e-10)
    np.testing.assert_array_equal(res.count, cnt)


def test_tolerance_is_close_to_exact() -> None:
    cat = make_catalogue(3000, seed=3)
    exact = run(cat, tol=0.0)
    approx = run(cat, tol=0.05)
    # Approximation error well inside the statistical error.
    assert np.all(np.abs(approx.sf - exact.sf) < 0.1 * exact.sf_err)


def test_counts_match_search_around_sky(small: Catalogue) -> None:
    res = run(small)
    c = small.coords
    i, j, sep, _ = search_around_sky(c, c, EDGES[-1])
    keep = i < j
    cnt, _ = np.histogram(sep[keep].to_value(u.deg), EDGES.to_value(u.deg))
    np.testing.assert_array_equal(res.count, cnt)


def test_frame_independent(small: Catalogue) -> None:
    icrs = run(small)
    gal = small.coords.galactic
    res = structure_function(
        small.values,
        small.errors,
        gal,
        bins=EDGES,
        weights=small.weights,
        bin_tolerance=0.0,
        regions=8,
        rng=0,
    )
    np.testing.assert_allclose(res.sf, icrs.sf, rtol=1e-9)


def test_units_round_trip(small: Catalogue) -> None:
    plain = run(small)
    unit = u.rad / u.m**2
    res = structure_function(
        small.values * unit,
        (small.errors * unit).to(u.rad / u.cm**2),
        small.coords,
        bins=EDGES,
        weights=small.weights,
        bin_tolerance=0.0,
        regions=8,
        rng=0,
    )
    assert res.sf_unit == unit**2
    assert plain.sf_unit is None
    np.testing.assert_allclose(res.sf, plain.sf, rtol=1e-10)


def test_mixed_units_raise(small: Catalogue) -> None:
    with pytest.raises(TypeError, match="both"):
        structure_function(small.values * u.rad, small.errors, small.coords)


def test_white_noise_saturates_at_two_variance() -> None:
    rng = np.random.default_rng(5)
    n = 4000
    c = SkyCoord(rng.uniform(0, 20, n) * u.deg, rng.uniform(-10, 10, n) * u.deg)
    f = rng.normal(0, 2.0, n)
    res = structure_function(f, None, c, bins=6, regions=32, rng=1)
    assert np.all(np.abs(res.sf - 2 * np.var(f)) < 4 * res.sf_err)
    np.testing.assert_allclose(res.saturation, 2 * np.var(f), rtol=1e-12)


def test_noise_is_debiased() -> None:
    rng = np.random.default_rng(6)
    n = 4000
    c = SkyCoord(rng.uniform(0, 20, n) * u.deg, rng.uniform(-10, 10, n) * u.deg)
    sigma = rng.uniform(0.5, 1.5, n)
    f = rng.normal(0, sigma)
    res = structure_function(f, sigma, c, bins=5, regions=40, rng=1)
    assert np.all(np.abs(res.sf) < 4 * res.sf_err)


def test_distances_give_3d_separations() -> None:
    rng = np.random.default_rng(7)
    n = 300
    c = SkyCoord(
        rng.uniform(0, 360, n) * u.deg,
        np.degrees(np.arcsin(rng.uniform(-1, 1, n))) * u.deg,
        rng.uniform(10, 100, n) * u.pc,
    )
    f = rng.normal(size=n)
    edges = np.geomspace(5, 150, 6) * u.pc
    res = structure_function(f, None, c, bins=edges, regions=4, bin_tolerance=0.0)
    i, j = np.triu_indices(n, 1)
    d = c[i].separation_3d(c[j]).to_value(u.pc)
    cnt, _ = np.histogram(d, edges.to_value(u.pc))
    np.testing.assert_array_equal(res.count, cnt)
    assert res.separation.unit == u.pc


def test_matches_treecorr(small: Catalogue) -> None:
    treecorr = pytest.importorskip("treecorr")

    res = run(small)
    f = small.values - np.average(small.values, weights=small.weights)
    edges = EDGES.to_value(u.deg)
    cfg = {
        "min_sep": edges[0],
        "max_sep": edges[-1],
        "nbins": edges.size - 1,
        "sep_units": "deg",
        "bin_slop": 0.0,
        "metric": "Arc",
    }
    cat_f = treecorr.Catalog(
        ra=small.lon,
        dec=small.lat,
        ra_units="deg",
        dec_units="deg",
        k=f,
        w=small.weights,
    )
    cat_f2 = treecorr.Catalog(
        ra=small.lon,
        dec=small.lat,
        ra_units="deg",
        dec_units="deg",
        k=f**2,
        w=small.weights,
    )
    kk = treecorr.KKCorrelation(cfg)
    kk.process(cat_f)
    nk = treecorr.NKCorrelation(cfg)
    nk.process(cat_f, cat_f2)
    # <(f1-f2)^2> = <f1^2> + <f2^2> - 2<f1 f2>; NK averages f^2 over pairs.
    sf_tc = 2 * nk.xi - 2 * kk.xi
    bias_free = res.sf + res.noise_bias
    np.testing.assert_allclose(bias_free, sf_tc, rtol=1e-6)
