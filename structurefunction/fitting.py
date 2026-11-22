"""Power-law model fits to structure functions using the full covariance.

The covariance is estimated from ``n_regions`` resamples, so it is itself noisy.
The default likelihood marginalises over that noise (Sellentin & Heavens
2016, MNRAS 456, L132): ``ln L = -(n_s / 2) ln(1 + chi^2 / (n_s - 1))``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import astropy.units as u
import numpy as np
from astropy.table import QTable
from capn_crunch import BaseOptions
from scipy.optimize import least_squares, minimize

from structurefunction.logging import logger
from structurefunction.types import (
    ErrorMethod,
    FitModel,
    FitModelLike,
    FloatArray,
    Likelihood,
    LikelihoodLike,
    Sampler,
    SamplerLike,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from structurefunction.results import SFResult


def power_law(r: FloatArray, theta: FloatArray, pivot: float) -> FloatArray:
    """``A (r / pivot)^alpha`` with ``theta = (log10 A, alpha)``."""
    return np.asarray(10.0 ** theta[0] * (r / pivot) ** theta[1], dtype=np.float64)


def broken_power_law(r: FloatArray, theta: FloatArray) -> FloatArray:
    """Sharp break: ``theta = (log10 A, log10 r_b, alpha_1, alpha_2)``."""
    x = r / 10.0 ** theta[1]
    return np.asarray(
        10.0 ** theta[0] * x ** np.where(x < 1.0, theta[2], theta[3]),
        dtype=np.float64,
    )


def smooth_broken_power_law(
    r: FloatArray, theta: FloatArray, smoothness: float = 0.1
) -> FloatArray:
    """Smooth break over ~``smoothness`` decades; same parameters as the sharp one.

    ``A x^a1 ((1 + x^(1/s)) / 2)^((a2 - a1) s)`` with ``x = r / r_b``.
    """
    x = r / 10.0 ** theta[1]
    log_term = np.logaddexp(0.0, np.log(x) / smoothness) - math.log(2.0)
    return np.asarray(
        10.0 ** theta[0]
        * x ** theta[2]
        * np.exp((theta[3] - theta[2]) * smoothness * log_term),
        dtype=np.float64,
    )


@dataclass(frozen=True, slots=True)
class FitResult:
    """Posterior summary of a model fit.

    Parameters are ``log10_amplitude`` (in the structure function's unit),
    ``log10_break`` (in the separation unit) and slopes.
    """

    model: FitModel
    """The fitted model."""
    names: tuple[str, ...]
    """Parameter names, in the order of the parameter arrays."""
    map: FloatArray
    """Maximum a-posteriori parameters."""
    median: FloatArray
    """Posterior median of each parameter."""
    lower: FloatArray
    """16th percentile (Laplace approximation for ``sampler='map'``)."""
    upper: FloatArray
    """84th percentile."""
    samples: FloatArray
    """Equal-weight posterior samples, shape ``(n_sample, n_param)``."""
    log_evidence: float | None
    """``ln Z`` (nautilus only)."""
    chi2: float
    """Chi-squared at the best fit."""
    n_data: int
    """Number of bins fitted."""
    aic: float
    """Akaike information criterion at the best fit."""
    bic: float
    """Bayesian information criterion at the best fit."""
    pivot: float
    """Reference separation of the plain power law (separation unit)."""
    separation_unit: u.UnitBase
    """Unit of separations for the break and pivot."""
    sf_unit: u.UnitBase | None
    """Unit of the structure function, or ``None`` for plain arrays."""


def evaluate(
    fit: FitResult,
    separation: FloatArray | u.Quantity,
    theta: FloatArray | None = None,
) -> FloatArray:
    """Model structure function at ``separation`` for ``theta`` (default: best fit)."""
    r = (
        separation.to_value(fit.separation_unit)
        if isinstance(separation, u.Quantity)
        else np.asarray(separation, dtype=np.float64)
    )
    parameters = fit.map if theta is None else theta
    return _model_fn(fit.model, fit.pivot)(r, parameters)


class FitPriors(BaseOptions):
    """Uniform prior bounds; ``None`` derives them from the data."""

    log10_amplitude: tuple[float, float] | None = None
    """Range of log10 amplitude (default: data range +/- 3 dex)."""
    log10_break: tuple[float, float] | None = None
    """Range of log10 break separation (default: range of the bins)."""
    slope: tuple[float, float] = (0.0, 6.0)
    """Structure functions do not decrease with separation, so slopes are >= 0."""


def constant(r: FloatArray, theta: FloatArray) -> FloatArray:
    """``A`` at every separation, with ``theta = (log10 A,)``."""
    return np.full(r.shape, 10.0 ** theta[0])


def _model_fn(
    model: FitModel, pivot: float
) -> Callable[[FloatArray, FloatArray], FloatArray]:
    if model == FitModel.CONSTANT:
        return constant
    if model == FitModel.POWER_LAW:
        return lambda r, th: power_law(r, th, pivot)
    if model == FitModel.BROKEN_POWER_LAW:
        return broken_power_law
    return smooth_broken_power_law


def _parameter_names(model: FitModel) -> tuple[str, ...]:
    if model == FitModel.CONSTANT:
        return ("log10_amplitude",)
    if model == FitModel.POWER_LAW:
        return ("log10_amplitude", "slope")
    return ("log10_amplitude", "log10_break", "slope_1", "slope_2")


def _bounds(
    model: FitModel,
    amplitude: tuple[float, float],
    log_break: tuple[float, float],
    slope: tuple[float, float],
) -> tuple[FloatArray, FloatArray]:
    """Lower and upper prior bounds, in parameter order."""
    if model == FitModel.CONSTANT:
        ranges = [amplitude]
    elif model == FitModel.POWER_LAW:
        ranges = [amplitude, slope]
    else:
        ranges = [amplitude, log_break, slope, slope]
    return np.array([r[0] for r in ranges]), np.array([r[1] for r in ranges])


def _starting_points(
    model: FitModel, log_amplitude: float, log_break: tuple[float, float]
) -> list[FloatArray]:
    """Several starts: the break makes the likelihood multi-modal."""
    if model == FitModel.CONSTANT:
        return [np.array([log_amplitude])]
    if model == FitModel.POWER_LAW:
        return [np.array([log_amplitude, s]) for s in (0.0, 0.5, 1.0, 2.0)]
    return [
        np.array([log_amplitude, b, s1, s2])
        for b in np.linspace(log_break[0], log_break[1], 7)[1:-1]
        for s1 in (0.5, 1.5, 3.0)
        for s2 in (0.0, 0.5)
    ]


def fit_sf(
    result: SFResult,
    model: FitModelLike = FitModel.BROKEN_POWER_LAW,
    sampler: SamplerLike = Sampler.EMCEE,
    likelihood: LikelihoodLike = Likelihood.SELLENTIN_HEAVENS,
    priors: FitPriors | None = None,
    use: np.ndarray[tuple[int], np.dtype[np.bool_]] | None = None,
    n_walkers: int = 32,
    n_steps: int = 4000,
    n_burn: int = 1000,
    n_live: int = 500,
    rng: np.random.Generator | int | None = None,
) -> FitResult:
    """Fit a (broken) power law to a structure function.

    Args:
        result: Structure function with a covariance.
        model: ``power_law``, ``broken_power_law`` or ``smooth_broken_power_law``.
        sampler: ``map`` (Laplace errors), ``emcee`` or ``nautilus`` (evidence).
        likelihood: How the estimated covariance enters the likelihood.
        priors: Uniform prior bounds.
        use: Bins to fit (default: ``result.reliable``). Bins larger than the
            resampling regions have underestimated errors; to leave them out,
            pass ``result.reliable & result.resampling_reliable``.
        n_walkers: emcee walkers.
        n_steps: emcee steps per walker.
        n_burn: emcee steps discarded as burn-in.
        n_live: nautilus live points. With at most four parameters, 500 gives
            ``ln Z`` within about 0.01 of nautilus' much slower default.
        rng: Random generator or seed.

    Returns:
        MAP, posterior percentiles and samples, and goodness-of-fit summaries.

    Raises:
        ValueError: Without a covariance, or with too few bins.
    """
    model = FitModel(model)
    sampler = Sampler(sampler)
    likelihood = Likelihood(likelihood)
    priors = FitPriors() if priors is None else priors
    gen = np.random.default_rng(rng)
    if result.cov is None or result.error_method == ErrorMethod.NONE:
        msg = "fitting needs a covariance: compute the SF with jackknife or bootstrap"
        raise ValueError(msg)

    mask = result.reliable if use is None else use
    sep_unit = result.separation.unit
    r = np.asarray(result.separation.to_value(sep_unit))[mask]
    sf_unit = result.sf_unit
    y = result.sf[mask]
    cov = result.cov[np.ix_(mask, mask)]
    names = _parameter_names(model)
    n_par = len(names)
    n_data = int(y.size)
    if n_data <= n_par:
        msg = f"{n_data} usable bins for {n_par} parameters"
        raise ValueError(msg)

    chol = np.linalg.cholesky(cov)
    n_s = (
        result.n_regions
        if result.error_method == ErrorMethod.JACKKNIFE
        else n_data * 10
    )
    pivot = float(np.exp(np.mean(np.log(r))))
    fn = _model_fn(model, pivot)

    ypos = np.abs(y[y != 0]) if np.any(y != 0) else np.array([1.0])
    amp_lo, amp_hi = priors.log10_amplitude or (
        math.log10(ypos.min()) - 3.0,
        math.log10(ypos.max()) + 3.0,
    )
    brk_lo, brk_hi = priors.log10_break or (math.log10(r.min()), math.log10(r.max()))
    lo, hi = _bounds(model, (amp_lo, amp_hi), (brk_lo, brk_hi), priors.slope)

    def whitened(theta: FloatArray) -> FloatArray:
        return np.asarray(np.linalg.solve(chol, y - fn(r, theta)), dtype=np.float64)

    def chi2(theta: FloatArray) -> float:
        z = whitened(theta)
        return float(z @ z)

    def log_like(theta: FloatArray) -> float:
        c2 = chi2(theta)
        if likelihood == Likelihood.SELLENTIN_HEAVENS:
            return -0.5 * n_s * math.log1p(c2 / (n_s - 1.0))
        if likelihood == Likelihood.HARTLAP:
            return -0.5 * c2 * (n_s - n_data - 2.0) / (n_s - 1.0)
        return -0.5 * c2

    def log_post(theta: FloatArray) -> float:
        if np.any(theta < lo) or np.any(theta > hi):
            return -np.inf
        return log_like(theta)

    # Multi-start least squares (the break makes the problem multi-modal),
    # then refine on the actual log-posterior.
    starts = _starting_points(
        model, math.log10(float(np.median(ypos))), (brk_lo, brk_hi)
    )
    best = None
    for x0 in starts:
        sol = least_squares(whitened, np.clip(x0, lo, hi), bounds=(lo, hi))
        if best is None or sol.cost < best.cost:
            best = sol
    assert best is not None  # noqa: S101

    def neg_log_post(theta: FloatArray) -> float:
        return -log_post(theta)

    ref = minimize(
        neg_log_post,
        np.asarray(best.x, dtype=np.float64),
        method="Nelder-Mead",
        options={"xatol": 1e-8, "fatol": 1e-10, "maxiter": 20000},
    )
    theta_map = np.asarray(ref.x if np.isfinite(ref.fun) else best.x)

    log_z: float | None = None
    if sampler == Sampler.MAP:
        # The likelihood (not the bounded posterior) so that steps across a
        # prior edge stay finite when the best fit sits on it.
        cov_theta = _laplace(log_like, theta_map)
        samples = np.clip(gen.multivariate_normal(theta_map, cov_theta, 20000), lo, hi)
    elif sampler == Sampler.EMCEE:
        import emcee

        scale = 1e-3 * (hi - lo)
        p0 = np.clip(
            theta_map + scale * gen.standard_normal((n_walkers, n_par)),
            lo + 1e-9 * (hi - lo),
            hi - 1e-9 * (hi - lo),
        )
        ens = emcee.EnsembleSampler(n_walkers, n_par, log_post)
        seed = int(gen.integers(2**31 - 1))
        ens.random_state = np.random.RandomState(seed).get_state()
        ens.run_mcmc(p0, n_steps, progress=False)
        samples = ens.get_chain(discard=n_burn, flat=True)
    elif len(names) == 1:
        # One parameter: integrate the posterior directly over the prior range
        # (nautilus needs at least two parameters).
        grid = np.linspace(lo[0], hi[0], 4001)
        log_l = np.array([log_like(np.array([x])) for x in grid])
        peak = log_l.max()
        density = np.exp(log_l - peak)
        log_z = float(peak + np.log(np.trapezoid(density, grid) / (hi[0] - lo[0])))
        cdf = np.cumsum(density)
        cdf /= cdf[-1]
        samples = np.interp(gen.uniform(size=20000), cdf, grid)[:, None]
    else:
        try:
            from nautilus import Prior
            from nautilus import Sampler as NautilusSampler
        except ImportError as err:
            msg = "sampler='nautilus' needs the 'evidence' extra"
            raise ImportError(msg) from err
        prior = Prior()
        for name, a, b in zip(names, lo, hi, strict=True):
            prior.add_parameter(name, dist=(float(a), float(b)))
        naut = NautilusSampler(
            prior,
            lambda d: log_like(np.array([d[k] for k in names])),
            n_live=n_live,
            n_networks=1,
            seed=int(gen.integers(2**31 - 1)),
        )
        naut.run(verbose=False)
        pts, _, _ = naut.posterior(equal_weight=True)
        samples = np.asarray(pts)
        log_z = float(naut.log_z)

    pct = np.percentile(samples, [16, 50, 84], axis=0)
    c2 = chi2(theta_map)
    best_log_like = -0.5 * c2
    logger.info("Fit %s: chi2 = %.2f for %d bins", model, c2, n_data)
    return FitResult(
        model=model,
        names=names,
        map=theta_map,
        median=pct[1],
        lower=pct[0],
        upper=pct[2],
        samples=samples,
        log_evidence=log_z,
        chi2=c2,
        n_data=n_data,
        aic=2 * n_par - 2 * best_log_like,
        bic=n_par * math.log(n_data) - 2 * best_log_like,
        pivot=pivot,
        separation_unit=sep_unit,
        sf_unit=sf_unit,
    )


def _laplace(log_post: Callable[[FloatArray], float], theta: FloatArray) -> FloatArray:
    """Covariance from the inverse Hessian of ``-log_post`` (central differences)."""
    n = theta.size
    h = 1e-4 * np.maximum(np.abs(theta), 1.0)
    hess = np.empty((n, n))
    f0 = log_post(theta)
    for i in range(n):
        for j in range(i, n):
            ei = np.zeros(n)
            ej = np.zeros(n)
            ei[i] = h[i]
            ej[j] = h[j]
            if i == j:
                val = (log_post(theta + ei) - 2 * f0 + log_post(theta - ei)) / h[i] ** 2
            else:
                val = (
                    log_post(theta + ei + ej)
                    - log_post(theta + ei - ej)
                    - log_post(theta - ei + ej)
                    + log_post(theta - ei - ej)
                ) / (4 * h[i] * h[j])
            hess[i, j] = hess[j, i] = val
    cov = np.linalg.pinv(-hess)
    # Symmetrise and guard against tiny negative eigenvalues.
    cov = 0.5 * (cov + cov.T)
    vals, vecs = np.linalg.eigh(cov)
    return np.asarray((vecs * np.clip(vals, 1e-12, None)) @ vecs.T)


def compare_models(
    result: SFResult,
    models: tuple[FitModelLike, ...] = (
        FitModel.CONSTANT,
        FitModel.POWER_LAW,
        FitModel.BROKEN_POWER_LAW,
        FitModel.SMOOTH_BROKEN_POWER_LAW,
    ),
    sampler: SamplerLike = Sampler.MAP,
    likelihood: LikelihoodLike = Likelihood.SELLENTIN_HEAVENS,
    rng: np.random.Generator | int | None = None,
) -> tuple[QTable, dict[FitModel, FitResult]]:
    """Fit several models to the same bins and compare them.

    Args:
        result: Structure function with a covariance.
        models: Models to fit.
        sampler: ``map`` for AIC/BIC only; ``nautilus`` adds the evidence.
        likelihood: How the estimated covariance enters the likelihood.
        rng: Random generator or seed.

    Returns:
        A table sorted by BIC (lowest first) with chi-squared, AIC, BIC and
        their differences from the best model (and ``log_evidence`` with
        ``nautilus``), and the fits by model.
    """
    gen = np.random.default_rng(rng)
    fits = {
        FitModel(m): fit_sf(
            result, model=m, sampler=sampler, likelihood=likelihood, rng=gen
        )
        for m in models
    }
    table = QTable()
    table["model"] = [str(m) for m in fits]
    table["n_params"] = [len(f.names) for f in fits.values()]
    table["chi2"] = [f.chi2 for f in fits.values()]
    table["n_bins"] = [f.n_data for f in fits.values()]
    table["aic"] = [f.aic for f in fits.values()]
    table["bic"] = [f.bic for f in fits.values()]
    table["delta_bic"] = table["bic"] - np.min(table["bic"])
    table["delta_aic"] = table["aic"] - np.min(table["aic"])
    if all(f.log_evidence is not None for f in fits.values()):
        log_z = np.array([f.log_evidence for f in fits.values()], dtype=np.float64)
        table["log_evidence"] = log_z
        table["delta_log_evidence"] = log_z - log_z.max()
    table.sort("bic")
    return table, fits
