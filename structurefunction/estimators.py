"""Turn accumulated sums into structure functions and resampled covariances."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from structurefunction._kernels.common import Kind, Stat
from structurefunction.types import CrossPatchWeight, ErrorMethod

if TYPE_CHECKING:
    from structurefunction.types import FloatArray


@dataclass(frozen=True, slots=True)
class Estimate:
    """Per-bin estimates from accumulated sums (internal units)."""

    sf: FloatArray
    """Noise-debiased structure function."""
    noise_bias: FloatArray
    """Weighted mean noise bias that was subtracted."""
    separation: FloatArray
    """Weighted mean separation of contributing pairs/triplets."""
    count: FloatArray
    """Number of contributing pairs/triplets (fractional with soft binning)."""
    weight: FloatArray
    """Sum of stencil weights."""
    cov: FloatArray | None
    """Covariance of ``sf`` between bins, or ``None``."""
    samples: FloatArray | None
    """Resampled ``sf`` (jackknife or bootstrap), shape ``(n_sample, n_bin)``."""


def _sf_from_totals(tot: FloatArray) -> tuple[FloatArray, FloatArray]:
    """``(sf, bias)`` from totals of shape ``(..., n_bin, n_stat)``."""
    w = tot[..., Stat.WEIGHT]
    with np.errstate(invalid="ignore", divide="ignore"):
        bias = np.where(w > 0, tot[..., Stat.ERROR_BIAS] / w, np.nan)
        sf = np.where(w > 0, tot[..., Stat.SQUARED_DIFF] / w, np.nan) - bias
    return sf, bias


def estimate(
    acc: FloatArray,
    error_method: ErrorMethod,
    cross_patch_weight: CrossPatchWeight,
    n_boot: int,
    rng: np.random.Generator,
) -> Estimate:
    """Structure function, bias, separations and covariance.

    Args:
        acc: Accumulator ``(kind, bin, patch, stat)`` summed over threads.
        error_method: Resampling scheme.
        cross_patch_weight: Jackknife weight for cross-patch contributions.
        n_boot: Number of bootstrap resamples.
        rng: Random generator for the bootstrap.

    Returns:
        The estimates; ``cov`` is ``None`` for ``ErrorMethod.NONE``.
    """
    mark = acc[Kind.MARK]  # (bin, patch, stat)
    total = mark.sum(axis=1)  # (bin, stat)
    sf, bias = _sf_from_totals(total)
    w = total[:, Stat.WEIGHT]
    with np.errstate(invalid="ignore", divide="ignore"):
        sep = np.where(w > 0, total[:, Stat.SEPARATION] / w, np.nan)
    count = total[:, Stat.COUNT]
    n_patch = acc.shape[2]

    cov: FloatArray | None = None
    samples: FloatArray | None = None
    if error_method == ErrorMethod.JACKKNIFE and n_patch > 1:
        alpha = 0.5 if cross_patch_weight == CrossPatchWeight.MATCH else 1.0
        removed = acc[Kind.AUTO] + alpha * acc[Kind.CROSS]  # (bin, patch, stat)
        loo = total[:, None, :] - removed  # (bin, patch, stat)
        samples, _ = _sf_from_totals(np.moveaxis(loo, 1, 0))  # (patch, bin)
        mean = _nanmean(samples)
        resid = samples - mean
        cov = (n_patch - 1) / n_patch * _nan_outer_sum(resid)
    elif error_method == ErrorMethod.BOOTSTRAP and n_patch > 1:
        counts = rng.multinomial(n_patch, np.full(n_patch, 1.0 / n_patch), n_boot)
        boot = np.einsum("sp,bpk->sbk", counts.astype(np.float64), mark)
        samples, _ = _sf_from_totals(boot)
        mean = _nanmean(samples)
        cov = _nan_outer_sum(samples - mean) / (n_boot - 1)
    return Estimate(
        sf=sf,
        noise_bias=bias,
        separation=sep,
        count=count,
        weight=w,
        cov=cov,
        samples=samples,
    )


def _nan_outer_sum(resid: FloatArray) -> FloatArray:
    """``sum_s r_s r_s^T`` treating NaN residuals (empty bins) as missing."""
    finite = np.isfinite(resid)
    r = np.where(finite, resid, 0.0)
    out = r.T @ r
    # Bins never populated in any sample stay NaN.
    seen = finite.any(axis=0)
    out[~seen, :] = np.nan
    out[:, ~seen] = np.nan
    return out


def _nanmean(samples: FloatArray) -> FloatArray:
    """Mean over samples, NaN (without a warning) for never-populated bins."""
    finite = np.isfinite(samples)
    n = finite.sum(axis=0)
    total = np.where(finite, samples, 0.0).sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, total / np.maximum(n, 1), np.nan)
