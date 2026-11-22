"""Three-point stencil kernels: D3(r) = <[f(x+) - 2 f(x0) + f(x-)]^2>.

For every middle source ``x0`` the tree is queried for neighbours in the
annulus of interest; bearings are measured in ``x0``'s tangent plane, where
great circles through ``x0`` are straight lines through the origin.

* ``exact``: explicit pairs of neighbours with opposite bearings (within
  ``angle_tol``) and matching arm lengths (within ``sep_tol``).
* ``binned``: neighbours (or whole tree nodes, when they fit) are aggregated
  into a polar grid of radial sub-bins x bearing sectors around ``x0``. Each
  cell is paired with the opposite sector in the same sub-bin. With
  ``a_i = f_i - f0`` the cell-pair sum is exact:

      sum_{i in A, j in B} w_i w_j (a_i + a_j)^2
          = W_B A2_A + W_A A2_B + 2 A1_A A1_B,

  where ``A1 = sum w a`` and ``A2 = sum w a^2 = M2 + W (mu - f0)^2 >= 0``.
"""

from __future__ import annotations

import math

import numba
import numpy as np

from structurefunction._kernels.common import (
    add_triplet,
    chord_to_sep,
    find_bin,
    projected_variance,
    soft_bins,
)
from structurefunction.types import FloatArray, IntArray


@numba.njit(cache=True)
def _grow(a: IntArray) -> IntArray:
    out = np.empty(a.size * 2, a.dtype)
    out[: a.size] = a
    return out

@numba.njit(cache=True, inline="always")
def _bearing(x: FloatArray, east: FloatArray, north: FloatArray) -> float:
    """Bearing of ``x`` in [0, 2 pi) in the tangent plane spanned by east/north."""
    e = x[0] * east[0] + x[1] * east[1] + x[2] * east[2]
    n = x[0] * north[0] + x[1] * north[1] + x[2] * north[2]
    b = math.atan2(e, n)
    if b < 0.0:
        b += (2.0 * math.pi)
    return b


@numba.njit(cache=True)
def _annulus(  # noqa: PLR0913
    m: int,
    xyz: FloatArray,
    angular: bool,  # noqa: FBT001
    s_lo: float,
    s_hi: float,
    start: IntArray,
    end: IntArray,
    left: IntArray,
    centre: FloatArray,
    radius: FloatArray,
    roots: IntArray,
    buf: IntArray,
) -> tuple[IntArray, int]:
    """Indices of sources with separation from ``m`` in ``[s_lo, s_hi)``."""
    count = 0
    stack = np.empty(256, np.intp)
    for r in range(roots.size):
        if roots[r] < 0:
            continue
        top = 0
        stack[top] = roots[r]
        top += 1
        while top > 0:
            top -= 1
            node = stack[top]
            d2 = 0.0
            for ax in range(3):
                dd = centre[node, ax] - xyz[m, ax]
                d2 += dd * dd
            d = math.sqrt(d2)
            if chord_to_sep(d + radius[node], angular) < s_lo:
                continue
            if chord_to_sep(max(d - radius[node], 0.0), angular) >= s_hi:
                continue
            if left[node] >= 0:
                if top + 2 >= stack.size:
                    stack = _grow(stack)
                stack[top] = left[node]
                stack[top + 1] = left[node] + 1  # children are consecutive
                top += 2
                continue
            for i in range(start[node], end[node]):
                if i == m:
                    continue
                c2 = 0.0
                for ax in range(3):
                    dd = xyz[i, ax] - xyz[m, ax]
                    c2 += dd * dd
                s = chord_to_sep(math.sqrt(c2), angular)
                if s_lo <= s < s_hi:
                    if count >= buf.size:
                        buf = _grow(buf)
                    buf[count] = i
                    count += 1
    return buf, count


@numba.njit(cache=True, parallel=True)
def exact_triplets(  # noqa: PLR0913, PLR0915, C901
    start: IntArray,
    end: IntArray,
    left: IntArray,
    centre: FloatArray,
    radius: FloatArray,
    roots: IntArray,
    xyz: FloatArray,
    f: FloatArray,
    sig2: FloatArray,
    w: FloatArray,
    patch: IntArray,
    edges: FloatArray,
    acc: FloatArray,
    angle_tol: float,
    sep_tol: float,
    east: FloatArray,
    north: FloatArray,
    pcov: FloatArray,
    use_pos: bool,  # noqa: FBT001
) -> FloatArray:
    """Explicit triplet search around every middle source (angular data)."""
    n = xyz.shape[0]
    n_slots = acc.shape[0]
    # Arms may differ by sep_tol around the mean separation r in [e0, eN).
    s_lo = edges[0] * (1.0 - sep_tol)
    s_hi = edges[-1] * (1.0 + sep_tol)
    for slot in numba.prange(n_slots):
        buf = np.empty(1024, np.intp)
        for m in range(slot, n, n_slots):
            buf, k = _annulus(
                m, xyz, True, s_lo, s_hi, start, end, left, centre, radius,  # noqa: FBT003
                roots, buf,
            )  # fmt: skip
            if k < 2:  # noqa: PLR2004
                continue
            idx = buf[:k].copy()
            bear = np.empty(k)
            sep = np.empty(k)
            for t in range(k):
                i = idx[t]
                rel = xyz[i] - xyz[m]
                bear[t] = _bearing(rel, east[m], north[m])
                c2 = rel[0] * rel[0] + rel[1] * rel[1] + rel[2] * rel[2]
                sep[t] = chord_to_sep(math.sqrt(c2), True)  # noqa: FBT003
            o = np.argsort(bear)
            idx = idx[o]
            bear = bear[o]
            sep = sep[o]
            # Wrapped copy so windows crossing 2 pi are contiguous.
            bear2 = np.concatenate((bear, bear + (2.0 * math.pi)))
            for ta in range(k):
                target = bear[ta] + math.pi
                lo = np.searchsorted(bear2, target - angle_tol, side="left")
                hi = np.searchsorted(bear2, target + angle_tol, side="right")
                for tb2 in range(lo, hi):
                    tb = tb2 % k
                    a = idx[ta]
                    b = idx[tb]
                    if a >= b:
                        continue  # each unordered pair once
                    sa = sep[ta]
                    sb = sep[tb]
                    mean = 0.5 * (sa + sb)
                    if abs(sa - sb) > sep_tol * mean:
                        continue
                    sigma = 0.0
                    if use_pos:
                        va = projected_variance(
                            xyz[a], xyz[m], east[a], north[a],
                            pcov[a, 0], pcov[a, 1], pcov[a, 2],
                        )  # fmt: skip
                        vb = projected_variance(
                            xyz[b], xyz[m], east[b], north[b],
                            pcov[b, 0], pcov[b, 1], pcov[b, 2],
                        )  # fmt: skip
                        # The middle point's error cancels in the mean arm.
                        sigma = 0.5 * math.sqrt(va + vb)
                    wt = w[m] * w[a] * w[b]
                    df = f[a] + f[b] - 2.0 * f[m]
                    bias = sig2[a] + sig2[b] + 4.0 * sig2[m]
                    soft_bins(
                        acc, slot, mean, sigma, edges, patch[m], patch[a],
                        patch[b], True, wt, wt * df * df, wt * bias, 1.0,  # noqa: FBT003
                    )  # fmt: skip
    return acc.sum(axis=0)


@numba.njit(cache=True, parallel=True)
def binned_triplets(  # noqa: PLR0912, PLR0913, PLR0915, C901
    start: IntArray,
    end: IntArray,
    left: IntArray,
    centre: FloatArray,
    radius: FloatArray,
    moments: FloatArray,
    node_patch: IntArray,
    roots: IntArray,
    xyz: FloatArray,
    f: FloatArray,
    sig2: FloatArray,
    w: FloatArray,
    patch: IntArray,
    edges: FloatArray,
    sub_edges: FloatArray,
    sub_to_bin: IntArray,
    n_sector: int,
    n_patch: int,
    acc: FloatArray,
    east: FloatArray,
    north: FloatArray,
) -> FloatArray:
    """Polar-grid moment aggregation around every middle source (angular data).

    ``sub_edges`` are radial sub-bin edges nested within ``edges``;
    ``sub_to_bin`` maps each sub-bin to its output bin. ``n_sector`` is even.
    """
    n = xyz.shape[0]
    n_slots = acc.shape[0]
    n_sub = sub_edges.size - 1
    n_cell = n_sub * n_sector
    half = n_sector // 2
    width = (2.0 * math.pi) / n_sector
    s_lo = sub_edges[0]
    s_hi = sub_edges[-1]
    for slot in numba.prange(n_slots):
        # cell stats: W, A1, A2, S, SR, N  per (patch, cell)
        cells = np.zeros((n_patch, n_cell, 6))
        touched = np.zeros((n_patch, n_cell), np.bool_)
        t_patch = np.empty(64, np.intp)
        t_cell = np.empty(64, np.intp)
        patch_seen = np.zeros(n_patch, np.bool_)
        plist = np.empty(n_patch, np.intp)
        stack = np.empty(256, np.intp)
        for m in range(slot, n, n_slots):
            n_t = 0
            n_p = 0
            f0 = f[m]
            em = east[m]
            nm = north[m]
            for r in range(roots.size):
                if roots[r] < 0:
                    continue
                top = 0
                stack[top] = roots[r]
                top += 1
                while top > 0:
                    top -= 1
                    node = stack[top]
                    rel0 = centre[node, 0] - xyz[m, 0]
                    rel1 = centre[node, 1] - xyz[m, 1]
                    rel2 = centre[node, 2] - xyz[m, 2]
                    d = math.sqrt(rel0 * rel0 + rel1 * rel1 + rel2 * rel2)
                    rad = radius[node]
                    smax = chord_to_sep(d + rad, True)  # noqa: FBT003
                    smin = chord_to_sep(max(d - rad, 0.0), True)  # noqa: FBT003
                    if smax < s_lo or smin >= s_hi:
                        continue
                    accepted = False
                    if d > rad:
                        j0 = find_bin(smin, sub_edges)
                        j1 = find_bin(smax, sub_edges)
                        pe = rel0 * em[0] + rel1 * em[1] + rel2 * em[2]
                        pn = rel0 * nm[0] + rel1 * nm[1] + rel2 * nm[2]
                        rho = math.sqrt(pe * pe + pn * pn)
                        if j0 >= 0 and j0 == j1 and rho > rad:
                            bc = math.atan2(pe, pn)
                            if bc < 0.0:
                                bc += (2.0 * math.pi)
                            hw = math.asin(rad / rho)
                            # Sectors are centred on multiples of width.
                            lo_b = bc - hw + 0.5 * width
                            hi_b = bc + hw + 0.5 * width
                            s0 = int(lo_b // width)
                            s1 = int(hi_b // width)
                            if s0 == s1:
                                s0 = s0 % n_sector
                                accepted = True
                                p = node_patch[node]
                                cell = j0 * n_sector + s0
                                wn = moments[node, 0]
                                dmu = moments[node, 1] - f0
                                sep_c = chord_to_sep(d, True)  # noqa: FBT003
                                if not touched[p, cell]:
                                    touched[p, cell] = True
                                    if n_t >= t_patch.size:
                                        t_patch = _grow(t_patch)
                                        t_cell = _grow(t_cell)
                                    t_patch[n_t] = p
                                    t_cell[n_t] = cell
                                    n_t += 1
                                    if not patch_seen[p]:
                                        patch_seen[p] = True
                                        plist[n_p] = p
                                        n_p += 1
                                cells[p, cell, 0] += wn
                                cells[p, cell, 1] += wn * dmu
                                cells[p, cell, 2] += moments[node, 2] + wn * dmu * dmu
                                cells[p, cell, 3] += moments[node, 3]
                                cells[p, cell, 4] += wn * sep_c
                                cells[p, cell, 5] += moments[node, 4]
                    if accepted:
                        continue
                    if left[node] >= 0:
                        if top + 2 >= stack.size:
                            stack = _grow(stack)
                        stack[top] = left[node]
                        stack[top + 1] = left[node] + 1
                        top += 2
                        continue
                    for i in range(start[node], end[node]):
                        if i == m:
                            continue
                        q0 = xyz[i, 0] - xyz[m, 0]
                        q1 = xyz[i, 1] - xyz[m, 1]
                        q2 = xyz[i, 2] - xyz[m, 2]
                        s = chord_to_sep(math.sqrt(q0 * q0 + q1 * q1 + q2 * q2), True)  # noqa: FBT003
                        j = find_bin(s, sub_edges)
                        if j < 0:
                            continue
                        pe = q0 * em[0] + q1 * em[1] + q2 * em[2]
                        pn = q0 * nm[0] + q1 * nm[1] + q2 * nm[2]
                        bi = math.atan2(pe, pn)
                        if bi < 0.0:
                            bi += (2.0 * math.pi)
                        sec = int((bi + 0.5 * width) // width) % n_sector
                        cell = j * n_sector + sec
                        p = patch[i]
                        if not touched[p, cell]:
                            touched[p, cell] = True
                            if n_t >= t_patch.size:
                                t_patch = _grow(t_patch)
                                t_cell = _grow(t_cell)
                            t_patch[n_t] = p
                            t_cell[n_t] = cell
                            n_t += 1
                            if not patch_seen[p]:
                                patch_seen[p] = True
                                plist[n_p] = p
                                n_p += 1
                        a = f[i] - f0
                        cells[p, cell, 0] += w[i]
                        cells[p, cell, 1] += w[i] * a
                        cells[p, cell, 2] += w[i] * a * a
                        cells[p, cell, 3] += w[i] * sig2[i]
                        cells[p, cell, 4] += w[i] * s
                        cells[p, cell, 5] += 1.0
            # Pair each touched cell in the first half of sectors with the
            # opposite cell, across all touched patches.
            w0 = w[m]
            pm = patch[m]
            for t in range(n_t):
                p1 = t_patch[t]
                c1 = t_cell[t]
                sec = c1 % n_sector
                if sec >= half:
                    continue
                j = c1 // n_sector
                c2 = c1 + half
                b = sub_to_bin[j]
                x1 = cells[p1, c1]
                for q in range(n_p):
                    p2 = plist[q]
                    if not touched[p2, c2]:
                        continue
                    x2 = cells[p2, c2]
                    ww = w0 * x1[0] * x2[0]
                    if ww <= 0.0:
                        continue
                    wd = w0 * (x2[0] * x1[2] + x1[0] * x2[2] + 2.0 * x1[1] * x2[1])
                    bias = w0 * (
                        x2[0] * x1[3] + x1[0] * x2[3] + 4.0 * sig2[m] * x1[0] * x2[0]
                    )
                    wr = 0.5 * w0 * (x2[0] * x1[4] + x1[0] * x2[4])
                    add_triplet(
                        acc, slot, b, pm, p1, p2, ww, wd, bias, wr, x1[5] * x2[5]
                    )
            # Reset only what was touched.
            for t in range(n_t):
                cells[t_patch[t], t_cell[t], :] = 0.0
                touched[t_patch[t], t_cell[t]] = False
            for q in range(n_p):
                patch_seen[plist[q]] = False
    return acc.sum(axis=0)

