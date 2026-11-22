from __future__ import annotations

import astropy.units as u
import numpy as np
import pytest
from astropy.coordinates import SkyCoord

from structurefunction import ErrorMethod, FitModel, SFResult, structure_function
from structurefunction.fitting import (
    broken_power_law,
    fit_sf,
    power_law,
)
from structurefunction.types import FloatArray

TRUTH = np.array([0.5, 0.0, 1.6, 0.2])  # log10 A, log10 r_b [deg], slopes


def injected(theta: FloatArray, seed: int, broken: bool = True) -> SFResult[FloatArray]:
    """An SFResult drawn from a known model with a known 5% covariance."""
    rng = np.random.default_rng(seed)
    edges = np.geomspace(0.05, 20, 15)
    r = np.sqrt(edges[:-1] * edges[1:])
    truth = broken_power_law(r, theta) if broken else power_law(r, theta[[0, 2]], 1.0)
    # Neighbouring bins correlated, as for real structure functions.
    corr = 0.5 ** np.abs(np.subtract.outer(np.arange(r.size), np.arange(r.size)))
    cov = np.outer(0.05 * truth, 0.05 * truth) * corr
    sf = rng.multivariate_normal(truth, cov)
    return SFResult[FloatArray](
        sf=sf,
        sf_err=np.sqrt(np.diag(cov)),
        noise_bias=np.zeros_like(sf),
        saturation=np.asarray(np.nan),
        cov=cov,
        separation=r * u.deg,
        bin_edges=edges * u.deg,
        count=np.full(r.size, 1e4),
        weight=np.full(r.size, 1e4),
        n_point=2,
        n_patch=500,
        error_method=ErrorMethod.JACKKNIFE,
        min_count=10,
    )


def test_map_recovers_injected_broken_power_law() -> None:
    fit = fit_sf(injected(TRUTH, 1), model="broken_power_law", sampler="map")
    sigma = 0.5 * (fit.upper - fit.lower)
    assert np.all(np.abs(fit.map - TRUTH) < 4 * sigma)


def test_emcee_recovers_and_brackets() -> None:
    fit = fit_sf(
        injected(TRUTH, 2),
        model="broken_power_law",
        sampler="emcee",
        n_steps=2000,
        n_burn=500,
        rng=0,
    )
    assert np.all(fit.lower < TRUTH + 4 * (fit.upper - fit.median))
    assert np.all(TRUTH - 4 * (fit.median - fit.lower) < fit.upper)


def test_bic_prefers_the_true_model() -> None:
    broken_data = injected(TRUTH, 3)
    single_data = injected(TRUTH, 4, broken=False)
    for data, winner in ((broken_data, "broken"), (single_data, "single")):
        single = fit_sf(data, model="power_law", sampler="map")
        broken = fit_sf(data, model="broken_power_law", sampler="map")
        assert (broken.bic < single.bic) == (winner == "broken")


def test_smooth_model_runs() -> None:
    fit = fit_sf(injected(TRUTH, 5), model="smooth_broken_power_law", sampler="map")
    assert fit.model == FitModel.SMOOTH_BROKEN_POWER_LAW
    assert np.isfinite(fit.chi2)


def test_model_is_continuous_at_break() -> None:
    theta = np.array([0.0, 0.0, 2.0, 0.5])
    r = np.array([1 - 1e-9, 1 + 1e-9])
    v = broken_power_law(r, theta)
    np.testing.assert_allclose(v[0], v[1], rtol=1e-6)


def test_fit_needs_covariance() -> None:
    rng = np.random.default_rng(0)
    c = SkyCoord(rng.uniform(0, 5, 200) * u.deg, rng.uniform(0, 5, 200) * u.deg)
    res = structure_function(rng.normal(size=200), None, c, error_method="none")
    with pytest.raises(ValueError, match="covariance"):
        fit_sf(res)


def test_nautilus_gives_evidence() -> None:
    pytest.importorskip("nautilus")
    fit = fit_sf(injected(TRUTH, 6), model="power_law", sampler="nautilus", rng=0)
    assert fit.log_evidence is not None
    assert np.isfinite(fit.log_evidence)
