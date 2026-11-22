"""Shared type aliases and string enums for user-facing modes."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

import numpy as np
from numpy.typing import NDArray

type FloatArray = NDArray[np.float64]
type IntArray = NDArray[np.intp]
type BoolArray = NDArray[np.bool_]


class ErrorMethod(StrEnum):
    """How the structure function covariance is estimated."""

    JACKKNIFE = "jackknife"
    """Delete-one jackknife over spatial patches (default)."""
    BOOTSTRAP = "bootstrap"
    """Marked bootstrap over spatial patches (Loh 2008)."""
    NONE = "none"
    """No covariance; only the noise-debiased structure function."""


type ErrorMethodLike = ErrorMethod | Literal["jackknife", "bootstrap", "none"]


class CrossPatchWeight(StrEnum):
    """Weight given to pairs spanning two patches in jackknife resamples."""

    SIMPLE = "simple"
    """Drop any pair or triplet touching the removed patch."""
    MATCH = "match"
    """Drop cross-patch pairs with weight 1/2 (Mohammad & Percival 2022)."""


type CrossPatchWeightLike = CrossPatchWeight | Literal["simple", "match"]


class StencilMode(StrEnum):
    """Algorithm used to gather multi-point stencils."""

    EXACT = "exact"
    """Explicit triplet search with hard angle and spacing tolerances."""
    BINNED = "binned"
    """Polar-grid moment aggregation around each middle point (fast)."""


type StencilModeLike = StencilMode | Literal["exact", "binned"]


class FitModel(StrEnum):
    """Models that can be fitted to a structure function."""

    POWER_LAW = "power_law"
    BROKEN_POWER_LAW = "broken_power_law"
    SMOOTH_BROKEN_POWER_LAW = "smooth_broken_power_law"


type FitModelLike = (
    FitModel | Literal["power_law", "broken_power_law", "smooth_broken_power_law"]
)


class Sampler(StrEnum):
    """Posterior estimation method for fits."""

    MAP = "map"
    """Maximum a-posteriori with Laplace (Hessian) uncertainties."""
    EMCEE = "emcee"
    """Affine-invariant MCMC posterior samples."""
    NAUTILUS = "nautilus"
    """Nested sampling with Bayesian evidence (needs the ``evidence`` extra)."""


type SamplerLike = Sampler | Literal["map", "emcee", "nautilus"]


class Likelihood(StrEnum):
    """Likelihood used when fitting with an estimated covariance."""

    SELLENTIN_HEAVENS = "sellentin_heavens"
    """Multivariate t-likelihood marginalising covariance noise (default)."""
    HARTLAP = "hartlap"
    """Gaussian with the Hartlap-debiased inverse covariance."""
    GAUSSIAN = "gaussian"
    """Plain Gaussian, covariance taken as exact."""


type LikelihoodLike = Likelihood | Literal["sellentin_heavens", "hartlap", "gaussian"]


class RAErrorConvention(StrEnum):
    """Meaning of a right-ascension positional error column."""

    ON_SKY = "on_sky"
    """Error is an on-sky angle, i.e. already includes cos(dec)."""
    COORDINATE = "coordinate"
    """Error is in the RA coordinate itself; multiplied by cos(dec) internally."""


type RAErrorConventionLike = RAErrorConvention | Literal["on_sky", "coordinate"]


class Taper(StrEnum):
    """Apodising window used to weight sources around a sky-map cell centre."""

    COSINE = "cosine"
    """``(1 + cos(pi d / R)) / 2`` for ``d < R``, zero beyond."""
    GAUSSIAN = "gaussian"
    """``exp(-d^2 / 2 s^2)`` with FWHM ``R``, truncated at ``3 s``."""


type TaperLike = Taper | Literal["cosine", "gaussian"]
