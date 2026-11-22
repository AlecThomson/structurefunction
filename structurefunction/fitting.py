"""Power-law model fits to structure functions using the full covariance.

The covariance is estimated from ``n_patch`` resamples, so it is itself noisy.
The default likelihood marginalises over that noise (Sellentin & Heavens
2016, MNRAS 456, L132): ``ln L = -(n_s / 2) ln(1 + chi^2 / (n_s - 1))``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import astropy.units as u
import numpy as np
from scipy.optimize import least_squares, minimize

from structurefunction.log import logger
from structurefunction.typedefs import (
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
    names: tuple[str, ...]
    map: FloatArray
    """Maximum a-posteriori parameters."""
    median: FloatArray
    lower: FloatArray
    """16th percentile (Laplace approximation for ``sampler='map'``)."""
    upper: FloatArray
    """84th percentile."""
    samples: FloatArray
    """Equal-weight posterior samples, shape ``(n_sample, n_param)``."""
    log_evidence: float | None
    """``ln Z`` (nautilus only)."""
    chi2: float
    n_data: int
    aic: float
    bic: float
    pivot: float
    """Reference separation of the plain power law (separation unit)."""
    separation_unit: u.UnitBase
    sf_unit: u.UnitBase | None

    def evaluate(
        self, separation: FloatArray | u.Quantity, theta: FloatArray | None = None
    ) -> FloatArray:
        """Model structure function at ``separation`` for ``theta`` (default MAP)."""
        r = (
            separation.to_value(self.separation_unit)
            if isinstance(separation, u.Quantity)
            else np.asarray(separation, dtype=np.float64)
        )
        th = self.map if theta is None else theta
        return _model_fn(self.model, self.pivot)(r, th)


@dataclass(frozen=True, slots=True)
class FitPriors:
    """Uniform prior bounds; ``None`` derives them from the data."""

    log10_amplitude: tuple[float, float] | None = None
    log10_break: tuple[float, float] | None = None
    slope: tuple[float, float] = (-2.0, 6.0)


def _model_fn(
    model: FitModel, pivot: float
) -> Callable[[FloatArray, FloatArray], FloatArray]:
    if model == FitModel.POWER_LAW:
        return lambda r, th: power_law(r, th, pivot)
    if model == FitModel.BROKEN_POWER_LAW:
        return broken_power_law
    return smooth_broken_power_law


def fit_sf(  # noqa: PLR0913, PLR0915, C901
    result: SFResult[FloatArray] | SFResult[u.Quantity],
    model: FitModelLike = FitModel.BROKEN_POWER_LAW,
    sampler: SamplerLike = Sampler.EMCEE,
    likelihood: LikelihoodLike = Likelihood.SELLENTIN_HEAVENS,
    priors: FitPriors | None = None,
    use: np.ndarray[tuple[int], np.dtype[np.bool_]] | None = None,
    n_walkers: int = 32,
    n_steps: int = 4000,
    n_burn: int = 1000,
    rng: np.random.Generator | int | None = None,
) -> FitResult:
    """Fit a (broken) power law to a structure function.

    Args:
        result: Structure function with a covariance.
        model: ``power_law``, ``broken_power_law`` or ``smooth_broken_power_law``.
        sampler: ``map`` (Laplace errors), ``emcee`` or ``nautilus`` (evidence).
        likelihood: How the estimated covariance enters the likelihood.
        priors: Uniform prior bounds.
        use: Bins to fit (default: ``result.reliable``).
        n_walkers: emcee walkers.
        n_steps: emcee steps per walker.
        n_burn: emcee steps discarded as burn-in.
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
    sf_unit = result.sf.unit if isinstance(result.sf, u.Quantity) else None
    y = np.asarray(u.Quantity(result.sf).value if sf_unit else result.sf)[mask]
    cov = result.cov[np.ix_(mask, mask)]
    names = (
        ("log10_amplitude", "slope")
        if model == FitModel.POWER_LAW
        else ("log10_amplitude", "log10_break", "slope_1", "slope_2")
    )
    n_par = len(names)
    n_data = int(y.size)
    if n_data <= n_par:
        msg = f"{n_data} usable bins for {n_par} parameters"
        raise ValueError(msg)

    chol = np.linalg.cholesky(cov)
    n_s = (
        result.n_patch if result.error_method == ErrorMethod.JACKKNIFE else n_data * 10
    )
    pivot = float(np.exp(np.mean(np.log(r))))
    fn = _model_fn(model, pivot)

    ypos = np.abs(y[y != 0]) if np.any(y != 0) else np.array([1.0])
    amp_lo, amp_hi = priors.log10_amplitude or (
        math.log10(ypos.min()) - 3.0,
        math.log10(ypos.max()) + 3.0,
    )
    brk_lo, brk_hi = priors.log10_break or (math.log10(r.min()), math.log10(r.max()))
    lo = np.array(
        [amp_lo, priors.slope[0]]
        if model == FitModel.POWER_LAW
        else [amp_lo, brk_lo, priors.slope[0], priors.slope[0]]
    )
    hi = np.array(
        [amp_hi, priors.slope[1]]
        if model == FitModel.POWER_LAW
        else [amp_hi, brk_hi, priors.slope[1], priors.slope[1]]
    )

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
    starts = []
    amp0 = math.log10(float(np.median(ypos)))
    if model == FitModel.POWER_LAW:
        starts = [np.array([amp0, s]) for s in (0.5, 1.0, 2.0)]
    else:
        for b in np.linspace(brk_lo, brk_hi, 7)[1:-1]:
            starts.extend(np.array([amp0, b, s1, 0.0]) for s1 in (0.5, 1.5, 3.0))
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
        cov_theta = _laplace(log_post, theta_map)
        samples = gen.multivariate_normal(theta_map, cov_theta, 20000)
    elif sampler == Sampler.EMCEE:
        import emcee  # noqa: PLC0415

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
    else:
        try:
            from nautilus import Prior  # noqa: PLC0415
            from nautilus import Sampler as NautilusSampler  # noqa: PLC0415
        except ImportError as err:
            msg = "sampler='nautilus' needs the 'evidence' extra"
            raise ImportError(msg) from err
        prior = Prior()
        for name, a, b in zip(names, lo, hi, strict=True):
            prior.add_parameter(name, dist=(float(a), float(b)))
        naut = NautilusSampler(
            prior,
            lambda d: log_like(np.array([d[k] for k in names])),
            seed=int(gen.integers(2**31 - 1)),
        )
        naut.run(verbose=False)
        pts, _, _ = naut.posterior(equal_weight=True)
        samples = np.asarray(pts)
        log_z = float(naut.log_z)

    pct = np.percentile(samples, [16, 50, 84], axis=0)
    c2 = chi2(theta_map)
    log_l = -0.5 * c2
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
        aic=2 * n_par - 2 * log_l,
        bic=n_par * math.log(n_data) - 2 * log_l,
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
