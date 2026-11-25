"""Jackknife regions: the areas of sky left out in turn to estimate errors.

The catalogue is split into compact areas of similar size by k-means
clustering of the source positions. The jackknife recomputes the structure
function with each region removed; the scatter of those estimates gives the
covariance between bins. There should be several times as many regions as
bins, or the covariance is too noisy to fit with.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy.cluster.vq import kmeans2

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

    from structurefunction.types import FloatArray, IntArray


@dataclass(frozen=True, slots=True)
class Regions:
    """Region labels and summary geometry."""

    labels: IntArray
    """Region index of each source, in ``[0, n_regions)``."""
    n_regions: int
    """Number of regions."""
    size: float
    """Typical region diameter (chord units).

    For each region, twice the 90th-percentile distance of its sources from
    the region centre; the median over regions.
    """


def make_regions(
    xyz: FloatArray,
    regions: int | ArrayLike,
    rng: np.random.Generator,
) -> Regions:
    """K-means regions on positions, or validate user-supplied labels.

    Args:
        xyz: Internal positions, shape ``(n, 3)``.
        regions: Number of regions, or an integer label per source.
        rng: Random generator used to seed k-means.

    Returns:
        The region labels (relabelled to be contiguous) and their size.
    """
    n = xyz.shape[0]
    if np.ndim(regions) == 0:
        k = int(np.asarray(regions))
        if k < 1:
            msg = f"number of regions must be >= 1, got {k}"
            raise ValueError(msg)
        k = min(k, n)
        if k == 1:
            labels = np.zeros(n, dtype=np.intp)
        else:
            seed = int(rng.integers(2**31 - 1))
            _, raw = kmeans2(xyz, k, minit="++", seed=seed)
            labels = np.asarray(raw, dtype=np.intp)
    else:
        labels = np.asarray(regions, dtype=np.intp)
        if labels.shape != (n,):
            msg = f"region labels must have shape ({n},), got {labels.shape}"
            raise ValueError(msg)
    # Relabel to 0..n_regions-1 (k-means can leave empty clusters).
    uniq, labels = np.unique(labels, return_inverse=True)
    labels = labels.astype(np.intp)
    n_regions = int(uniq.size)
    sizes = np.empty(n_regions)
    for p in range(n_regions):
        member = xyz[labels == p]
        centre = member.mean(axis=0)
        distances = np.sqrt(np.sum((member - centre) ** 2, axis=1))
        sizes[p] = 2.0 * np.percentile(distances, 90)
    return Regions(labels=labels, n_regions=n_regions, size=float(np.median(sizes)))


def recommend_jk_regions(
    n_sources: int,
    n_bins: int,
    preferred: int = 64,
    per_bin: int = 3,
    min_sources: int = 10,
) -> int:
    """Number of jackknife regions to use.

    ``preferred`` regions, raised to at least ``per_bin`` per bin (fewer make
    the covariance too noisy to fit with), and capped so each region holds at
    least ``min_sources`` sources.

    Args:
        n_sources: Number of sources.
        n_bins: Number of separation bins.
        preferred: Count to use when it gives enough regions per bin.
        per_bin: Minimum regions per bin.
        min_sources: Minimum sources per region.

    Returns:
        The number of regions.
    """
    count = max(preferred, per_bin * n_bins)
    return int(min(count, max(n_sources // min_sources, 2)))
