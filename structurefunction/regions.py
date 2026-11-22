"""Jackknife regions: the areas of sky left out in turn to estimate errors.

The catalogue is split into compact areas of similar size by k-means
clustering of the source positions. The jackknife recomputes the structure
function with each region removed; the scatter of those estimates gives the
covariance between bins. Regions must be larger than the separations of
interest, and there should be several times as many regions as bins.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import astropy.units as u
import numpy as np
from astropy_healpix import HEALPix
from scipy.cluster.vq import kmeans2

from structurefunction.logging import logger

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


def footprint_area(xyz: FloatArray, sources_per_pixel: float = 5.0) -> float:
    """Sky area covered by sources on the unit sphere, in steradians.

    Counts occupied HEALPix pixels at the finest resolution that still has
    about ``sources_per_pixel`` sources per occupied pixel.
    """
    lon = np.arctan2(xyz[:, 1], xyz[:, 0])
    lat = np.arcsin(np.clip(xyz[:, 2], -1.0, 1.0))
    area = 4.0 * np.pi
    nside = 1
    while nside <= 2**13:
        grid = HEALPix(nside=nside)
        occupied = np.unique(grid.lonlat_to_healpix(lon * u.rad, lat * u.rad)).size
        if xyz.shape[0] / occupied < sources_per_pixel:
            break
        area = occupied * float(grid.pixel_area.to_value(u.sr))
        nside *= 2
    return area


def recommend_jk_regions(
    xyz: FloatArray,
    angular: bool,
    largest_scale: float,
    n_bins: int,
    preferred: int = 64,
    per_bin: int = 3,
    min_sources: int = 10,
) -> int:
    """Number of jackknife regions to use.

    Regions should be at least as large as the largest separation of
    interest, or errors there are underestimated; and there should be at least
    ``per_bin`` regions per bin, or the covariance is too noisy to fit with.
    Up to ``preferred`` regions are used when both allow it.

    Args:
        xyz: Unit vectors (angular data) or 3D positions.
        angular: Whether positions are on the sky.
        largest_scale: Largest separation of interest (radians on the sky).
        n_bins: Number of separation bins.
        preferred: Count to use when the field is large enough.
        per_bin: Minimum regions per bin.
        min_sources: Minimum sources per region.

    Returns:
        The number of regions.
    """
    fewest = per_bin * n_bins
    most_for_sources = max(xyz.shape[0] // min_sources, 2)
    if not angular:
        return int(min(max(fewest, preferred), most_for_sources))
    # Equal-area regions: diameter 2 sqrt(area / (pi n)) >= largest_scale.
    area = footprint_area(xyz)
    most_for_scale = int(4.0 * area / (np.pi * largest_scale**2))
    count = max(fewest, min(preferred, most_for_scale))
    if count > most_for_scale:
        scale = np.degrees(2.0 * np.sqrt(area / (np.pi * count)))
        logger.info(
            "The field is too small for jackknife regions both larger than the "
            "binned scales and numerous enough for %d bins; errors are "
            "underestimated above about %.2g deg (see resampling_reliable). Use "
            "fewer bins or a smaller max_sep to avoid this.",
            n_bins,
            scale,
        )
    return int(min(count, most_for_sources))
