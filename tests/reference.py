"""Slow, obviously-correct numpy references for the kernels."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from structurefunction.types import FloatArray


def unit_vectors(lon_deg: FloatArray, lat_deg: FloatArray) -> FloatArray:
    lon = np.radians(lon_deg)
    lat = np.radians(lat_deg)
    return np.column_stack(
        (np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat))
    )


def angsep(a: FloatArray, b: FloatArray) -> FloatArray:
    """Great-circle angle between unit vectors (radians), numerically stable."""
    cross = np.linalg.norm(np.cross(a, b), axis=-1)
    dot = np.sum(a * b, axis=-1)
    return np.arctan2(cross, dot)


def sf2(
    xyz: FloatArray,
    f: FloatArray,
    sigma: FloatArray,
    w: FloatArray,
    edges: FloatArray,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """(sf, bias, count) over all pairs; edges in radians."""
    i, j = np.triu_indices(f.size, k=1)
    s = angsep(xyz[i], xyz[j])
    ww = w[i] * w[j]
    b = np.digitize(s, edges) - 1
    ok = (b >= 0) & (b < edges.size - 1)
    nb = edges.size - 1
    wsum = np.bincount(b[ok], ww[ok], nb)
    wd = np.bincount(b[ok], (ww * (f[i] - f[j]) ** 2)[ok], nb)
    wb = np.bincount(b[ok], (ww * (sigma[i] ** 2 + sigma[j] ** 2))[ok], nb)
    cnt = np.bincount(b[ok], None, nb).astype(float)
    return (wd - wb) / wsum, wb / wsum, cnt


def sf3_exact(
    xyz: FloatArray,
    f: FloatArray,
    sigma: FloatArray,
    w: FloatArray,
    edges: FloatArray,
    angle_tol: float,
    sep_tol: float,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Triplet SF by direct enumeration around each middle point."""
    nb = edges.size - 1
    wsum = np.zeros(nb)
    wd = np.zeros(nb)
    wb = np.zeros(nb)
    cnt = np.zeros(nb)
    n = f.size
    for m in range(n):
        x0 = xyz[m]
        lon = np.arctan2(x0[1], x0[0])
        lat = np.arcsin(x0[2])
        east = np.array([-np.sin(lon), np.cos(lon), 0.0])
        north = np.array(
            [-np.sin(lat) * np.cos(lon), -np.sin(lat) * np.sin(lon), np.cos(lat)]
        )
        rel = xyz - x0
        bear = np.arctan2(rel @ east, rel @ north)
        s = angsep(xyz, x0[None, :])
        a, b = np.triu_indices(n, k=1)
        keep = (a != m) & (b != m)
        a, b = a[keep], b[keep]
        dang = np.abs(np.angle(np.exp(1j * (bear[b] - bear[a] - np.pi))))
        mean = 0.5 * (s[a] + s[b])
        ok = (dang <= angle_tol) & (np.abs(s[a] - s[b]) <= sep_tol * mean)
        a, b, mean = a[ok], b[ok], mean[ok]
        k = np.digitize(mean, edges) - 1
        ok = (k >= 0) & (k < nb)
        a, b, k = a[ok], b[ok], k[ok]
        ww = w[m] * w[a] * w[b]
        np.add.at(wsum, k, ww)
        np.add.at(wd, k, ww * (f[a] + f[b] - 2 * f[m]) ** 2)
        np.add.at(wb, k, ww * (sigma[a] ** 2 + sigma[b] ** 2 + 4 * sigma[m] ** 2))
        np.add.at(cnt, k, 1.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (wd - wb) / wsum, wb / wsum, cnt
