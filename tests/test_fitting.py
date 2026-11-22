from __future__ import annotations

from dataclasses import replace

import astropy.units as u
import numpy as np
import pytest
from astropy.coordinates import SkyCoord

from structurefunction import ErrorMethod, FitModel, SFResult, structure_function
from structurefunction.fitting import (
    broken_power_law,
    compare_models,
    fit_sf,
    power_law,
)
from structurefunction.types import FitModelLike, FloatArray

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
        n_regions=500,
        error_method=ErrorMethod.JACKKNIFE,
        min_count=10,
        reliable=np.ones(r.size, dtype=bool),
        region_size=None,
        resampling_reliable=np.ones(r.size, dtype=bool),
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


@pytest.mark.slow
def test_nautilus_gives_evidence() -> None:
    pytest.importorskip("nautilus")
    fit = fit_sf(injected(TRUTH, 6), model="power_law", sampler="nautilus", rng=0)
    assert fit.log_evidence is not None
    assert np.isfinite(fit.log_evidence)


def flat_result(level: float, seed: int) -> SFResult[FloatArray]:
    """A flat structure function with a known 5% diagonal covariance."""
    template = injected(TRUTH, seed)
    rng = np.random.default_rng(seed)
    cov = np.diag((0.05 * level) ** 2 * np.ones(template.sf.size))
    sf = rng.multivariate_normal(np.full(template.sf.size, level), cov)
    return replace(template, sf=sf, sf_err=np.sqrt(np.diag(cov)), cov=cov)


def test_constant_recovers_flat_injection() -> None:
    fit = fit_sf(flat_result(50.0, 7), model="constant", sampler="map")
    assert abs(10 ** fit.map[0] - 50.0) < 2.0


def test_compare_models_prefers_the_true_model() -> None:
    models: tuple[FitModelLike, ...] = ("constant", "power_law", "broken_power_law")
    wins = {"constant": 0, "broken_power_law": 0}
    for seed in range(10):
        for data, truth in (
            (flat_result(50.0, 100 + seed), "constant"),
            (injected(TRUTH, 200 + seed), "broken_power_law"),
        ):
            table, fits = compare_models(data, models=models)
            assert list(table["bic"]) == sorted(table["bic"])
            assert set(fits) == {FitModel(m) for m in models}
            wins[truth] += table["model"][0] == truth
    # BIC picks the right model most of the time, not every time.
    assert wins["constant"] >= 7
    assert wins["broken_power_law"] >= 9


def test_one_parameter_evidence_matches_analytic() -> None:
    # Flat data with a Gaussian likelihood: the evidence integral over the
    # amplitude prior can be checked by direct quadrature in A itself.
    data = flat_result(50.0, 10)
    fit = fit_sf(data, model="constant", sampler="nautilus", likelihood="gaussian")
    assert data.cov is not None
    inv = np.linalg.inv(data.cov)
    lo, hi = np.log10(np.abs(data.sf).min()) - 3, np.log10(np.abs(data.sf).max()) + 3
    grid = np.linspace(lo, hi, 200001)
    resid = data.sf[None, :] - 10 ** grid[:, None]
    log_l = -0.5 * np.einsum("ij,jk,ik->i", resid, inv, resid)
    peak = log_l.max()
    expected = peak + np.log(np.trapezoid(np.exp(log_l - peak), grid) / (hi - lo))
    assert fit.log_evidence is not None
    assert abs(fit.log_evidence - expected) < 1e-3
