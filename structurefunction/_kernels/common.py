"""Accumulator layout and per-pair / per-triplet helpers shared by all kernels.

Accumulators have shape ``(slot, kind, bin, patch, stat)``. ``slot`` is a
thread-private copy (summed at the end), ``kind`` is one of :class:`Kind`
and ``stat`` one of :class:`Stat`.

For a pair or triplet touching patches ``P``:

* ``AUTO[p]`` receives it if every member is in patch ``p``.
* ``CROSS[p]`` receives it for each distinct ``p`` in ``P`` otherwise.
* ``MARK[p]`` receives the fraction of members that lie in ``p``.

``sum_p MARK[p]`` is the full-sample total; deleting patch ``p`` removes
``AUTO[p] + CROSS[p]`` (or ``AUTO[p] + CROSS[p] / 2`` for matched weighting).
"""

from __future__ import annotations

import math
from enum import IntEnum

import numba
import numpy as np

from structurefunction.types import FloatArray


class Kind(IntEnum):
    """Accumulator kinds (axis 1)."""

    AUTO = 0
    CROSS = 1
    MARK = 2


class Stat(IntEnum):
    """Accumulated statistics (last axis)."""

    W = 0
    """Sum of stencil weights."""
    WD = 1
    """Weighted sum of squared stencil differences."""
    WB = 2
    """Weighted sum of the noise bias ``sum_i c_i^2 sigma_i^2``."""
    WR = 3
    """Weighted sum of separations."""
    N = 4
    """Unweighted count of pairs / triplets."""


def new_accumulator(n_slots: int, n_bins: int, n_patch: int) -> FloatArray:
    """Allocate zeroed thread-private accumulators."""
    return np.zeros((n_slots, len(Kind), n_bins, n_patch, len(Stat)), dtype=np.float64)


@numba.njit(cache=True, inline="always")
def chord_to_sep(chord: float, angular: bool) -> float:  # noqa: FBT001
    """Chord length to separation (great-circle angle, or 3D distance)."""
    if not angular:
        return chord
    half = chord * 0.5
    half = min(half, 1.0)
    return 2.0 * math.asin(half)


@numba.njit(cache=True, inline="always")
def find_bin(sep: float, edges: FloatArray) -> int:
    """Index ``k`` with ``edges[k] <= sep < edges[k + 1]``, or -1."""
    if sep < edges[0] or sep >= edges[-1]:
        return -1
    return int(np.searchsorted(edges, sep, side="right")) - 1


@numba.njit(cache=True, inline="always")
def norm_cdf(x: float) -> float:
    """Standard normal CDF."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


@numba.njit(cache=True)
def add_pair(  # noqa: PLR0913
    acc: FloatArray,
    slot: int,
    b: int,
    pa: int,
    pb: int,
    w: float,
    wd: float,
    wb: float,
    wr: float,
    n: float,
) -> None:
    """Accumulate one (aggregated) pair contribution into bin ``b``."""
    if pa == pb:
        _add(acc, slot, Kind.AUTO, b, pa, w, wd, wb, wr, n, 1.0)
        _add(acc, slot, Kind.MARK, b, pa, w, wd, wb, wr, n, 1.0)
    else:
        _add(acc, slot, Kind.CROSS, b, pa, w, wd, wb, wr, n, 1.0)
        _add(acc, slot, Kind.CROSS, b, pb, w, wd, wb, wr, n, 1.0)
        _add(acc, slot, Kind.MARK, b, pa, w, wd, wb, wr, n, 0.5)
        _add(acc, slot, Kind.MARK, b, pb, w, wd, wb, wr, n, 0.5)


@numba.njit(cache=True)
def add_triplet(  # noqa: PLR0913
    acc: FloatArray,
    slot: int,
    b: int,
    p0: int,
    p1: int,
    p2: int,
    w: float,
    wd: float,
    wb: float,
    wr: float,
    n: float,
) -> None:
    """Accumulate one (aggregated) triplet contribution into bin ``b``."""
    if p0 == p1 == p2:
        _add(acc, slot, Kind.AUTO, b, p0, w, wd, wb, wr, n, 1.0)
        _add(acc, slot, Kind.MARK, b, p0, w, wd, wb, wr, n, 1.0)
        return
    third = 1.0 / 3.0
    # Each distinct patch once in CROSS; MARK by member fraction.
    _add(acc, slot, Kind.CROSS, b, p0, w, wd, wb, wr, n, 1.0)
    if p1 != p0:
        _add(acc, slot, Kind.CROSS, b, p1, w, wd, wb, wr, n, 1.0)
    if p2 not in (p0, p1):
        _add(acc, slot, Kind.CROSS, b, p2, w, wd, wb, wr, n, 1.0)
    _add(acc, slot, Kind.MARK, b, p0, w, wd, wb, wr, n, third)
    _add(acc, slot, Kind.MARK, b, p1, w, wd, wb, wr, n, third)
    _add(acc, slot, Kind.MARK, b, p2, w, wd, wb, wr, n, third)


@numba.njit(cache=True, inline="always")
def _add(  # noqa: PLR0913
    acc: FloatArray,
    slot: int,
    kind: int,
    b: int,
    p: int,
    w: float,
    wd: float,
    wb: float,
    wr: float,
    n: float,
    frac: float,
) -> None:
    acc[slot, kind, b, p, Stat.W] += frac * w
    acc[slot, kind, b, p, Stat.WD] += frac * wd
    acc[slot, kind, b, p, Stat.WB] += frac * wb
    acc[slot, kind, b, p, Stat.WR] += frac * wr
    acc[slot, kind, b, p, Stat.N] += frac * n


@numba.njit(cache=True, inline="always")
def projected_variance(  # noqa: PLR0913
    xi: FloatArray,
    xj: FloatArray,
    east_i: FloatArray,
    north_i: FloatArray,
    var_e: float,
    var_n: float,
    cov_en: float,
) -> float:
    """Variance of source ``i``'s position along the direction towards ``j``."""
    de = (xj[0] - xi[0]) * east_i[0] + (xj[1] - xi[1]) * east_i[1] + (
        xj[2] - xi[2]
    ) * east_i[2]
    dn = (xj[0] - xi[0]) * north_i[0] + (xj[1] - xi[1]) * north_i[1] + (
        xj[2] - xi[2]
    ) * north_i[2]
    norm2 = de * de + dn * dn
    if norm2 <= 0.0:
        return 0.5 * (var_e + var_n)
    return (de * de * var_e + dn * dn * var_n + 2.0 * de * dn * cov_en) / norm2


@numba.njit(cache=True)
def soft_bins(  # noqa: PLR0913
    acc: FloatArray,
    slot: int,
    sep: float,
    sigma: float,
    edges: FloatArray,
    pa: int,
    pb: int,
    p2: int,
    triplet: bool,  # noqa: FBT001
    w: float,
    wd: float,
    wb: float,
    n: float,
) -> None:
    """Spread one contribution over bins with Gaussian separation uncertainty.

    ``sigma <= 0`` reduces to hard binning. Contributions are scaled by the
    probability mass falling in each bin; mass outside the bin range is dropped.
    """
    if sigma <= 0.0:
        b = find_bin(sep, edges)
        if b < 0:
            return
        if triplet:
            add_triplet(acc, slot, b, pa, pb, p2, w, wd, wb, w * sep, n)
        else:
            add_pair(acc, slot, b, pa, pb, w, wd, wb, w * sep, n)
        return
    lo = find_bin(max(sep - 5.0 * sigma, edges[0]), edges)
    hi = find_bin(min(sep + 5.0 * sigma, np.nextafter(edges[-1], 0.0)), edges)
    if lo < 0 or hi < 0:
        return
    for k in range(lo, hi + 1):
        frac = norm_cdf((edges[k + 1] - sep) / sigma) - norm_cdf(
            (edges[k] - sep) / sigma
        )
        if frac <= 0.0:
            continue
        if triplet:
            add_triplet(
                acc, slot, k, pa, pb, p2, frac * w, frac * wd, frac * wb,
                frac * w * sep, frac * n,
            )  # fmt: skip
        else:
            add_pair(
                acc, slot, k, pa, pb, frac * w, frac * wd, frac * wb,
                frac * w * sep, frac * n,
            )  # fmt: skip
