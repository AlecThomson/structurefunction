"""Geometry on internal positions: tangent bases, separation range, bin edges."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.spatial import KDTree

if TYPE_CHECKING:
    from structurefunction.types import FloatArray


def tangent_basis(lon: FloatArray, lat: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Unit east and north vectors at each position, shape ``(n, 3)`` each."""
    sin_lon, cos_lon = np.sin(lon), np.cos(lon)
    sin_lat, cos_lat = np.sin(lat), np.cos(lat)
    east = np.column_stack((-sin_lon, cos_lon, np.zeros_like(lon)))
    north = np.column_stack((-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat))
    return east, north


def chord_to_separation(chord: FloatArray, angular: bool) -> FloatArray:  # noqa: FBT001
    """Vectorised chord length to separation."""
    if not angular:
        return chord
    return 2.0 * np.arcsin(np.clip(chord * 0.5, 0.0, 1.0))


def separation_range(xyz: FloatArray, angular: bool) -> tuple[float, float]:  # noqa: FBT001
    """Smallest non-zero and (near-)largest pair separation.

    The minimum is exact (nearest neighbours via a KD-tree). The maximum uses a
    double-sweep farthest-point heuristic, which is exact for typical survey
    footprints and never more than a factor two low.
    """
    tree = KDTree(xyz)
    dist, _ = tree.query(xyz, k=2)
    nn = dist[:, 1]
    nn = nn[nn > 0.0]
    if nn.size == 0:
        msg = "all sources are coincident; no non-zero separations"
        raise ValueError(msg)
    centroid = xyz.mean(axis=0)
    a = int(np.argmax(np.sum((xyz - centroid) ** 2, axis=1)))
    best = 0.0
    for _ in range(3):
        d = np.sqrt(np.sum((xyz - xyz[a]) ** 2, axis=1))
        b = int(np.argmax(d))
        best = max(best, float(d[b]))
        a = b
    smin = float(chord_to_separation(np.array([nn.min()]), angular)[0])
    smax = float(chord_to_separation(np.array([best]), angular)[0])
    return smin, smax


def log_edges(smin: float, smax: float, n_bins: int) -> FloatArray:
    """Logarithmic bin edges, nudged outwards so the extremes are included."""
    if not 0.0 < smin < smax:
        msg = f"need 0 < min_sep < max_sep, got {smin}, {smax}"
        raise ValueError(msg)
    edges = np.geomspace(smin, smax, n_bins + 1)
    edges[0] = np.nextafter(edges[0], 0.0)
    edges[-1] = np.nextafter(edges[-1], np.inf)
    return edges
