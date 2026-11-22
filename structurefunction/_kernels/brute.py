"""Exact O(N^2) two-point kernel (reference and small-N path)."""

from __future__ import annotations

import math

import numba
import numpy as np

from structurefunction._kernels.common import (
    chord_to_sep,
    projected_variance,
    soft_bins,
)
from structurefunction.types import FloatArray, IntArray


@numba.njit(cache=True, inline="always")
def pair_into(  # noqa: PLR0913
    acc: FloatArray,
    slot: int,
    i: int,
    j: int,
    xyz: FloatArray,
    f: FloatArray,
    sig2: FloatArray,
    w: FloatArray,
    patch: IntArray,
    edges: FloatArray,
    angular: bool,  # noqa: FBT001
    east: FloatArray,
    north: FloatArray,
    pcov: FloatArray,
    use_pos: bool,  # noqa: FBT001
) -> None:
    """Accumulate the exact contribution of source pair ``(i, j)``."""
    dx = xyz[i, 0] - xyz[j, 0]
    dy = xyz[i, 1] - xyz[j, 1]
    dz = xyz[i, 2] - xyz[j, 2]
    sep = chord_to_sep(math.sqrt(dx * dx + dy * dy + dz * dz), angular)
    sigma = 0.0
    if use_pos:
        vi = projected_variance(
            xyz[i], xyz[j], east[i], north[i], pcov[i, 0], pcov[i, 1], pcov[i, 2]
        )
        vj = projected_variance(
            xyz[j], xyz[i], east[j], north[j], pcov[j, 0], pcov[j, 1], pcov[j, 2]
        )
        sigma = math.sqrt(vi + vj)
    wij = w[i] * w[j]
    df = f[i] - f[j]
    soft_bins(
        acc, slot, sep, sigma, edges, patch[i], patch[j], patch[j], False,  # noqa: FBT003
        wij, wij * df * df, wij * (sig2[i] + sig2[j]), 1.0,
    )  # fmt: skip


@numba.njit(cache=True, parallel=True)
def brute_pairs(  # noqa: PLR0913
    xyz: FloatArray,
    f: FloatArray,
    sig2: FloatArray,
    w: FloatArray,
    patch: IntArray,
    edges: FloatArray,
    acc: FloatArray,
    angular: bool,  # noqa: FBT001
    east: FloatArray,
    north: FloatArray,
    pcov: FloatArray,
    use_pos: bool,  # noqa: FBT001
) -> FloatArray:
    """All-pairs exact two-point accumulation.

    Rows are interleaved across thread slots for load balance over the
    triangular ``i < j`` loop.

    Returns:
        Accumulator summed over slots, shape ``(kind, bin, patch, stat)``.
    """
    n = xyz.shape[0]
    n_slots = acc.shape[0]
    for slot in numba.prange(n_slots):
        for i in range(slot, n, n_slots):
            for j in range(i + 1, n):
                pair_into(
                    acc, slot, i, j, xyz, f, sig2, w, patch, edges, angular,
                    east, north, pcov, use_pos,
                )  # fmt: skip
    return acc.sum(axis=0)


def empty_pos_arrays(n: int) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Placeholder east/north/covariance arrays when positional errors are off."""
    return np.zeros((n, 3)), np.zeros((n, 3)), np.zeros((n, 3))
