"""Exact two-point sums over individual source pairs."""

from __future__ import annotations

import math

import numba
import numpy as np

from structurefunction._kernels.common import (
    Contribution,
    add_pair,
    chord_to_separation,
    distance,
    find_bin,
    gaussian_bin_fraction,
    gaussian_bin_range,
    projected_position_var,
)
from structurefunction._kernels.containers import Binning, Sources
from structurefunction.types import FloatArray


@numba.njit(cache=True, inline="always")
def add_pair_to_bins(
    acc: FloatArray,
    slot: int,
    separation: float,
    sigma: float,
    bins: Binning,
    patch_a: int,
    patch_b: int,
    contribution: Contribution,
) -> None:
    """Add a pair to its bin, or spread it over bins if its separation is uncertain.

    ``sigma`` is the 1-sigma uncertainty of the separation (0 for none).
    Probability falling outside the binned range is dropped.
    """
    if sigma <= 0.0:
        bin_index = find_bin(separation, bins.edges)
        if bin_index >= 0:
            add_pair(acc, slot, bin_index, patch_a, patch_b, contribution, 1.0)
        return
    first, last = gaussian_bin_range(separation, sigma, bins.edges)
    if first < 0:
        return
    for bin_index in range(first, last + 1):
        fraction = gaussian_bin_fraction(separation, sigma, bins.edges, bin_index)
        if fraction > 0.0:
            add_pair(acc, slot, bin_index, patch_a, patch_b, contribution, fraction)


@numba.njit(cache=True, inline="always")
def position_var_towards(source: int, other: int, sources: Sources) -> float:
    """Positional variance of ``source`` along the direction to ``other``."""
    return projected_position_var(
        sources.positions,
        source,
        other,
        sources.east,
        sources.north,
        sources.position_cov,
    )


@numba.njit(cache=True, inline="always")
def add_source_pair(
    acc: FloatArray, slot: int, i: int, j: int, sources: Sources, bins: Binning
) -> None:
    """Add the pair of sources ``i`` and ``j``."""
    chord = distance(sources.positions, i, sources.positions, j)
    separation = chord_to_separation(chord, bins.angular)
    sigma = 0.0
    if bins.use_position_errors:
        sigma = math.sqrt(
            position_var_towards(i, j, sources) + position_var_towards(j, i, sources)
        )
    weight = sources.weights[i] * sources.weights[j]
    difference = sources.values[i] - sources.values[j]
    contribution = Contribution(
        weight,
        weight * difference * difference,
        weight * (sources.error_var[i] + sources.error_var[j]),
        weight * separation,
        1.0,
    )
    add_pair_to_bins(
        acc,
        slot,
        separation,
        sigma,
        bins,
        sources.patch[i],
        sources.patch[j],
        contribution,
    )


@numba.njit(cache=True, parallel=True)
def brute_pairs(sources: Sources, bins: Binning, acc: FloatArray) -> FloatArray:
    """Sum over every pair of sources (reference implementation).

    Rows are interleaved between threads to balance the triangular loop.

    Returns:
        Accumulator summed over threads, shape ``(kind, bin, patch, stat)``.
    """
    n_sources = sources.values.size
    n_slots = acc.shape[0]
    for slot in numba.prange(n_slots):
        for i in range(slot, n_sources, n_slots):
            for j in range(i + 1, n_sources):
                add_source_pair(acc, slot, i, j, sources, bins)
    return acc.sum(axis=0)


def no_position_errors(n_sources: int) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Placeholder east, north and covariance arrays without position errors."""
    return np.zeros((n_sources, 3)), np.zeros((n_sources, 3)), np.zeros((n_sources, 3))
