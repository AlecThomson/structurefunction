"""Accumulator layout and helpers shared by all kernels.

Accumulators have shape ``(slot, kind, bin, region, stat)``. ``slot`` is a
per-thread copy (summed at the end), ``kind`` is one of :class:`Kind` and
``stat`` one of :class:`Stat`.

For a pair or triplet whose sources lie in a set of regions:

* ``AUTO[p]`` receives it if every source is in region ``p``;
* ``CROSS[p]`` receives it, for each region ``p`` in the set, otherwise;
* ``MARK[p]`` receives the fraction of its sources that are in ``p``.

Summing ``MARK`` over regions gives the full-sample total, and leaving region
``p`` out removes ``AUTO[p] + CROSS[p]``.
"""

from __future__ import annotations

import math
from enum import IntEnum
from typing import NamedTuple

import numba
import numpy as np

from structurefunction.types import FloatArray


class Kind(IntEnum):
    """Accumulator kinds (axis 1)."""

    AUTO = 0
    """Pairs or triplets with every source in the region."""
    CROSS = 1
    """Pairs or triplets with sources in this and other regions."""
    MARK = 2
    """Share of each pair or triplet by the fraction of its sources in the region."""


class Stat(IntEnum):
    """Accumulated statistics (last axis)."""

    WEIGHT = 0
    """Sum of stencil weights."""
    SQUARED_DIFF = 1
    """Weighted sum of squared stencil differences."""
    ERROR_BIAS = 2
    """Weighted sum of the error term ``sum_i c_i^2 sigma_i^2``."""
    SEPARATION = 3
    """Weighted sum of separations."""
    COUNT = 4
    """Number of pairs or triplets."""


class Contribution(NamedTuple):
    """What one pair, triplet or group of them adds to a bin."""

    weight: float
    """Stencil weight (product of the source weights)."""
    squared_diff: float
    """Weight times the squared stencil difference."""
    error_bias: float
    """Weight times the error term ``sum_i c_i^2 sigma_i^2``."""
    separation: float
    """Weighted separation, i.e. ``weight * separation``."""
    n_terms: float
    """Number of pairs or triplets."""


def new_accumulator(n_slots: int, n_bins: int, n_regions: int) -> FloatArray:
    """Allocate zeroed per-thread accumulators."""
    return np.zeros(
        (n_slots, len(Kind), n_bins, n_regions, len(Stat)), dtype=np.float64
    )


@numba.njit(cache=True, inline="always")
def chord_to_separation(chord: float, angular: bool) -> float:
    """Straight-line distance to separation (great-circle angle, or unchanged)."""
    if not angular:
        return chord
    return 2.0 * math.asin(min(0.5 * chord, 1.0))


@numba.njit(cache=True, inline="always")
def distance(points_a: FloatArray, a: int, points_b: FloatArray, b: int) -> float:
    """Distance between row ``a`` of ``points_a`` and row ``b`` of ``points_b``.

    Rows are indexed directly rather than sliced: in numba a slice is a new
    array object, which costs noticeably in the innermost loops.
    """
    dx = points_a[a, 0] - points_b[b, 0]
    dy = points_a[a, 1] - points_b[b, 1]
    dz = points_a[a, 2] - points_b[b, 2]
    return math.sqrt(dx * dx + dy * dy + dz * dz)


@numba.njit(cache=True, inline="always")
def find_bin(separation: float, edges: FloatArray) -> int:
    """Index ``k`` with ``edges[k] <= separation < edges[k + 1]``, or -1."""
    if separation < edges[0] or separation >= edges[-1]:
        return -1
    return int(np.searchsorted(edges, separation, side="right")) - 1


@numba.njit(cache=True, inline="always")
def normal_cdf(x: float) -> float:
    """Standard normal cumulative distribution function."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


@numba.njit(cache=True, inline="always")
def gaussian_bin_fraction(
    separation: float, sigma: float, edges: FloatArray, bin_index: int
) -> float:
    """Share of a Gaussian (``separation``, ``sigma``) inside one bin."""
    upper = normal_cdf((edges[bin_index + 1] - separation) / sigma)
    lower = normal_cdf((edges[bin_index] - separation) / sigma)
    return upper - lower


@numba.njit(cache=True, inline="always")
def gaussian_bin_range(
    separation: float, sigma: float, edges: FloatArray
) -> tuple[int, int]:
    """First and last bins within 5 sigma of ``separation`` (-1 if none)."""
    first = find_bin(max(separation - 5.0 * sigma, edges[0]), edges)
    last = find_bin(min(separation + 5.0 * sigma, np.nextafter(edges[-1], 0.0)), edges)
    if first < 0 or last < 0:
        return -1, -1
    return first, last


@numba.njit(cache=True, inline="always")
def _add(
    acc: FloatArray,
    slot: int,
    kind: int,
    bin_index: int,
    region: int,
    contribution: Contribution,
    fraction: float,
) -> None:
    # Indexed in full rather than via a row slice, which is slower in numba.
    acc[slot, kind, bin_index, region, Stat.WEIGHT] += fraction * contribution.weight
    acc[slot, kind, bin_index, region, Stat.SQUARED_DIFF] += (
        fraction * contribution.squared_diff
    )
    acc[slot, kind, bin_index, region, Stat.ERROR_BIAS] += (
        fraction * contribution.error_bias
    )
    acc[slot, kind, bin_index, region, Stat.SEPARATION] += (
        fraction * contribution.separation
    )
    acc[slot, kind, bin_index, region, Stat.COUNT] += fraction * contribution.n_terms


@numba.njit(cache=True, inline="always")
def add_pair(
    acc: FloatArray,
    slot: int,
    bin_index: int,
    region_a: int,
    region_b: int,
    contribution: Contribution,
    fraction: float,
) -> None:
    """Add ``fraction`` of a pair (or of all pairs between two groups)."""
    if region_a == region_b:
        _add(acc, slot, Kind.AUTO, bin_index, region_a, contribution, fraction)
        _add(acc, slot, Kind.MARK, bin_index, region_a, contribution, fraction)
        return
    half = 0.5 * fraction
    _add(acc, slot, Kind.CROSS, bin_index, region_a, contribution, fraction)
    _add(acc, slot, Kind.CROSS, bin_index, region_b, contribution, fraction)
    _add(acc, slot, Kind.MARK, bin_index, region_a, contribution, half)
    _add(acc, slot, Kind.MARK, bin_index, region_b, contribution, half)


@numba.njit(cache=True, inline="always")
def add_triplet(
    acc: FloatArray,
    slot: int,
    bin_index: int,
    region_middle: int,
    region_a: int,
    region_b: int,
    contribution: Contribution,
    fraction: float,
) -> None:
    """Add ``fraction`` of a triplet (or of a group sharing these regions)."""
    if region_middle == region_a == region_b:
        _add(acc, slot, Kind.AUTO, bin_index, region_middle, contribution, fraction)
        _add(acc, slot, Kind.MARK, bin_index, region_middle, contribution, fraction)
        return
    # Each distinct region once in CROSS; MARK by the share of sources.
    _add(acc, slot, Kind.CROSS, bin_index, region_middle, contribution, fraction)
    if region_a != region_middle:
        _add(acc, slot, Kind.CROSS, bin_index, region_a, contribution, fraction)
    if region_b not in (region_middle, region_a):
        _add(acc, slot, Kind.CROSS, bin_index, region_b, contribution, fraction)
    third = fraction / 3.0
    _add(acc, slot, Kind.MARK, bin_index, region_middle, contribution, third)
    _add(acc, slot, Kind.MARK, bin_index, region_a, contribution, third)
    _add(acc, slot, Kind.MARK, bin_index, region_b, contribution, third)


@numba.njit(cache=True, inline="always")
def projected_position_var(
    positions: FloatArray,
    source: int,
    other: int,
    east: FloatArray,
    north: FloatArray,
    position_cov: FloatArray,
) -> float:
    """Positional variance of ``source`` along the direction to ``other``."""
    step_east = 0.0
    step_north = 0.0
    for axis in range(3):
        step = positions[other, axis] - positions[source, axis]
        step_east += step * east[source, axis]
        step_north += step * north[source, axis]
    length_sq: float = step_east * step_east + step_north * step_north
    var_east: float = position_cov[source, 0]
    var_north: float = position_cov[source, 1]
    if length_sq <= 0.0:
        return 0.5 * (var_east + var_north)
    projected: float = (
        step_east * step_east * var_east
        + step_north * step_north * var_north
        + 2.0 * step_east * step_north * position_cov[source, 2]
    )
    return projected / length_sq
