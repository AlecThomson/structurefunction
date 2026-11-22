"""Patch-pure ball tree with per-node value moments, and dual-tree 2-point kernel.

Every node stores the weighted mean ``mu``, the weighted sum of squared
deviations ``M2`` and the weighted noise sum ``S = sum w sigma^2`` of its
sources. For two disjoint nodes ``A`` and ``B`` the sum over all cross pairs is
then exact and free of cancellation:

    sum_{i in A, j in B} w_i w_j (f_i - f_j)^2
        = W_B M2_A + W_A M2_B + W_A W_B (mu_A - mu_B)^2

The only approximation when a node pair is accepted as a block is that every
pair is given the separation of the node centres.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numba
import numpy as np

# Exported at runtime but missing from numba's type information.
from numba import get_thread_id  # type: ignore[attr-defined]

from structurefunction._kernels.brute import pair_into
from structurefunction._kernels.common import (
    add_pair,
    chord_to_sep,
    find_bin,
    norm_cdf,
)

if TYPE_CHECKING:
    from structurefunction.types import FloatArray, IntArray


@dataclass(frozen=True, slots=True)
class Tree:
    """Flat-array ball tree over patch-sorted sources.

    Attributes:
        order: Permutation from tree order to input order.
        roots: Root node index of each patch.
        start, end: Source range ``[start, end)`` (tree order) of each node.
        left, right: Children (``-1`` for leaves).
        centre: Node centre, shape ``(n_node, 3)``.
        radius: Max distance from centre to a member.
        moments: ``(n_node, 6)``: W, mu, M2, S, n, max major-axis position var.
        patch: Patch of each node.
    """

    order: IntArray
    roots: IntArray
    start: IntArray
    end: IntArray
    left: IntArray
    right: IntArray
    centre: FloatArray
    radius: FloatArray
    moments: FloatArray
    patch: IntArray


def build_tree(
    xyz: FloatArray,
    f: FloatArray,
    sig2: FloatArray,
    w: FloatArray,
    patch: IntArray,
    n_patch: int,
    pos_var_major: FloatArray,
    leaf_size: int = 4,
) -> Tree:
    """Build one subtree per patch (so every node is patch-pure).

    Source arrays must afterwards be indexed with ``tree.order`` so that node
    ranges ``[start, end)`` address contiguous sources.
    """
    by_patch = np.argsort(patch, kind="stable").astype(np.intp)
    bounds = np.searchsorted(patch[by_patch], np.arange(n_patch + 1)).astype(np.intp)
    arrays = _build(
        xyz[by_patch],
        f[by_patch],
        sig2[by_patch],
        w[by_patch],
        pos_var_major[by_patch],
        bounds,
        leaf_size,
    )
    start, end, left, right, centre, radius, moments, roots, node_patch, perm = arrays
    return Tree(
        order=by_patch[perm],
        roots=roots,
        start=start,
        end=end,
        left=left,
        right=right,
        centre=centre,
        radius=radius,
        moments=moments,
        patch=node_patch,
    )


@numba.njit(cache=True)
def _build(
    xyz: FloatArray,
    f: FloatArray,
    sig2: FloatArray,
    w: FloatArray,
    vmaj: FloatArray,
    bounds: IntArray,
    leaf_size: int,
) -> tuple[
    IntArray,
    IntArray,
    IntArray,
    IntArray,
    FloatArray,
    FloatArray,
    FloatArray,
    IntArray,
    IntArray,
    IntArray,
]:
    """Median-split construction; returns node arrays and the tree-order permutation."""
    n = xyz.shape[0]
    n_patch = bounds.size - 1
    cap = 4 * (n // max(leaf_size // 2, 1) + 1) + 2 * n_patch + 8
    start = np.empty(cap, np.intp)
    end = np.empty(cap, np.intp)
    left = np.full(cap, -1, np.intp)
    right = np.full(cap, -1, np.intp)
    node_patch = np.empty(cap, np.intp)
    roots = np.full(n_patch, -1, np.intp)
    perm = np.arange(n)
    stack = np.empty((cap, 2), np.intp)
    n_node = 0
    for p in range(n_patch):
        if bounds[p + 1] <= bounds[p]:
            continue
        roots[p] = n_node
        start[n_node] = bounds[p]
        end[n_node] = bounds[p + 1]
        node_patch[n_node] = p
        n_node += 1
        top = 0
        stack[top, 0] = roots[p]
        top += 1
        while top > 0:
            top -= 1
            node = stack[top, 0]
            s = start[node]
            e = end[node]
            if e - s <= leaf_size:
                continue
            # Split on the axis of largest spread at the median.
            spread = np.empty(3)
            for ax in range(3):
                spread[ax] = xyz[perm[s:e], ax].max() - xyz[perm[s:e], ax].min()
            ax = int(np.argmax(spread))
            if spread[ax] <= 0.0:
                continue  # coincident points: keep as a (large) leaf
            idx = np.argsort(xyz[perm[s:e], ax])
            perm[s:e] = perm[s:e][idx]
            mid = s + (e - s) // 2
            left[node] = n_node
            start[n_node] = s
            end[n_node] = mid
            node_patch[n_node] = p
            stack[top, 0] = n_node
            top += 1
            n_node += 1
            right[node] = n_node
            start[n_node] = mid
            end[n_node] = e
            node_patch[n_node] = p
            stack[top, 0] = n_node
            top += 1
            n_node += 1

    # Node ranges index the permuted (tree-order) sources.
    xyz = xyz[perm]
    f = f[perm]
    sig2 = sig2[perm]
    w = w[perm]
    vmaj = vmaj[perm]

    centre = np.zeros((n_node, 3))
    radius = np.zeros(n_node)
    moments = np.zeros((n_node, 6))
    for node in range(n_node):
        s = start[node]
        e = end[node]
        for ax in range(3):
            centre[node, ax] = xyz[s:e, ax].mean()
        r2 = 0.0
        for i in range(s, e):
            d2 = 0.0
            for ax in range(3):
                d = xyz[i, ax] - centre[node, ax]
                d2 += d * d
            r2 = max(r2, d2)
        radius[node] = math.sqrt(r2)
        # Two-pass weighted moments (exact; nodes are small near the leaves and
        # this is O(N log N) overall).
        wsum = 0.0
        wf = 0.0
        ssum = 0.0
        vmax = 0.0
        for i in range(s, e):
            wsum += w[i]
            wf += w[i] * f[i]
            ssum += w[i] * sig2[i]
            vmax = max(vmax, vmaj[i])
        mu = wf / wsum if wsum > 0.0 else 0.0
        m2 = 0.0
        for i in range(s, e):
            d = f[i] - mu
            m2 += w[i] * d * d
        moments[node, 0] = wsum
        moments[node, 1] = mu
        moments[node, 2] = m2
        moments[node, 3] = ssum
        moments[node, 4] = e - s
        moments[node, 5] = vmax
    return (
        start[:n_node],
        end[:n_node],
        left[:n_node],
        right[:n_node],
        centre,
        radius,
        moments,
        roots,
        node_patch[:n_node],
        perm,
    )


@numba.njit(cache=True)
def _node_pair(
    acc: FloatArray,
    slot: int,
    b: int,
    frac: float,
    sep: float,
    ma: FloatArray,
    mb: FloatArray,
    pa: int,
    pb: int,
) -> None:
    """Accumulate all cross pairs of two nodes into bin ``b`` with weight ``frac``."""
    wa = ma[0]
    wb = mb[0]
    dmu = ma[1] - mb[1]
    wd = wb * ma[2] + wa * mb[2] + wa * wb * dmu * dmu
    bias = wb * ma[3] + wa * mb[3]
    ww = wa * wb
    add_pair(
        acc,
        slot,
        b,
        pa,
        pb,
        frac * ww,
        frac * wd,
        frac * bias,
        frac * ww * sep,
        frac * ma[4] * mb[4],
    )


@numba.njit(cache=True)
def _accept_block(
    acc: FloatArray,
    slot: int,
    sep: float,
    sigma: float,
    edges: FloatArray,
    ma: FloatArray,
    mb: FloatArray,
    pa: int,
    pb: int,
) -> None:
    """Accumulate a node pair at one separation, softly binned if ``sigma > 0``."""
    if sigma <= 0.0:
        b = find_bin(sep, edges)
        if b >= 0:
            _node_pair(acc, slot, b, 1.0, sep, ma, mb, pa, pb)
        return
    lo = find_bin(max(sep - 5.0 * sigma, edges[0]), edges)
    hi = find_bin(min(sep + 5.0 * sigma, np.nextafter(edges[-1], 0.0)), edges)
    if lo < 0 or hi < 0:
        return
    for k in range(lo, hi + 1):
        frac = norm_cdf((edges[k + 1] - sep) / sigma) - norm_cdf(
            (edges[k] - sep) / sigma
        )
        if frac > 0.0:
            _node_pair(acc, slot, k, frac, sep, ma, mb, pa, pb)


@numba.njit(cache=True, inline="always")
def _triangle_cdf(x: float, lo: float, mode: float, hi: float) -> float:
    """CDF of the triangular distribution on ``[lo, hi]`` peaking at ``mode``."""
    if x <= lo:
        return 0.0
    if x >= hi:
        return 1.0
    if x <= mode:
        return (x - lo) ** 2 / ((hi - lo) * (mode - lo))
    return 1.0 - (hi - x) ** 2 / ((hi - lo) * (hi - mode))


@numba.njit(cache=True)
def _accept_spread(
    acc: FloatArray,
    slot: int,
    smin: float,
    sep_c: float,
    smax: float,
    edges: FloatArray,
    ma: FloatArray,
    mb: FloatArray,
    pa: int,
    pb: int,
) -> None:
    """Share a node pair between the bins its separations span.

    Separations of the pairs are modelled as triangular on ``[smin, smax]``
    peaking at the centre separation, which is a good description for two
    compact groups and much better than assigning everything to one bin.
    """
    mode = min(max(sep_c, smin + 1e-15 * smax), smax - 1e-15 * smax)
    lo = find_bin(max(smin, edges[0]), edges)
    hi = find_bin(min(smax, np.nextafter(edges[-1], 0.0)), edges)
    if lo < 0 or hi < 0:
        return
    for k in range(lo, hi + 1):
        a = max(smin, edges[k])
        b = min(smax, edges[k + 1])
        frac = _triangle_cdf(b, smin, mode, smax) - _triangle_cdf(a, smin, mode, smax)
        if frac > 0.0:
            _node_pair(acc, slot, k, frac, 0.5 * (a + b), ma, mb, pa, pb)


@numba.njit(cache=True, parallel=True)
def dual_tree_pairs(
    start: IntArray,
    end: IntArray,
    left: IntArray,
    right: IntArray,
    centre: FloatArray,
    radius: FloatArray,
    moments: FloatArray,
    node_patch: IntArray,
    frontier: IntArray,
    xyz: FloatArray,
    f: FloatArray,
    sig2: FloatArray,
    w: FloatArray,
    patch: IntArray,
    edges: FloatArray,
    acc: FloatArray,
    angular: bool,
    bin_tolerance: float,
    east: FloatArray,
    north: FloatArray,
    pcov: FloatArray,
    use_pos: bool,
) -> FloatArray:
    """Dual-tree two-point accumulation over all patch-root pairs.

    A node pair is accepted as a block when every pair it contains (widened by
    5 sigma of positional error) falls in a single bin (exact), or when the
    spread of separations is within ``bin_tolerance`` of a bin width
    (approximate). Otherwise the larger node is split; leaf pairs are summed
    exactly.
    """
    # Every pair of frontier nodes (including each node with itself) is one
    # parallel task; the frontier partitions the sources, so each source pair
    # is visited exactly once.
    n_front = frontier.size
    n_tasks = n_front * (n_front + 1) // 2
    task_a = np.empty(n_tasks, np.intp)
    task_b = np.empty(n_tasks, np.intp)
    t = 0
    for ia in range(n_front):
        for ib in range(ia, n_front):
            task_a[t] = frontier[ia]
            task_b[t] = frontier[ib]
            t += 1
    s_lo = edges[0]
    s_hi = edges[-1]
    for t in numba.prange(n_tasks):
        slot = get_thread_id()
        stack = np.empty((256, 2), np.intp)
        top = 0
        stack[0, 0] = task_a[t]
        stack[0, 1] = task_b[t]
        top = 1
        while top > 0:
            top -= 1
            a = stack[top, 0]
            b = stack[top, 1]
            if top + 3 >= stack.shape[0]:
                grown = np.empty((stack.shape[0] * 2, 2), np.intp)
                grown[: stack.shape[0]] = stack
                stack = grown
            if a == b:
                if left[a] < 0:
                    for i in range(start[a], end[a]):
                        for j in range(i + 1, end[a]):
                            pair_into(
                                acc,
                                slot,
                                i,
                                j,
                                xyz,
                                f,
                                sig2,
                                w,
                                patch,
                                edges,
                                angular,
                                east,
                                north,
                                pcov,
                                use_pos,
                            )
                else:
                    stack[top, 0] = left[a]
                    stack[top, 1] = left[a]
                    stack[top + 1, 0] = right[a]
                    stack[top + 1, 1] = right[a]
                    stack[top + 2, 0] = left[a]
                    stack[top + 2, 1] = right[a]
                    top += 3
                continue
            d2 = 0.0
            for ax in range(3):
                dd = centre[a, ax] - centre[b, ax]
                d2 += dd * dd
            d = math.sqrt(d2)
            rr = radius[a] + radius[b]
            sep_c = chord_to_sep(d, angular)
            smin = chord_to_sep(max(d - rr, 0.0), angular)
            smax = chord_to_sep(d + rr, angular)
            pad = 0.0
            if use_pos:
                pad = 5.0 * math.sqrt(moments[a, 5] + moments[b, 5])
            if smax + pad < s_lo or smin - pad >= s_hi:
                continue  # entirely outside the binned range
            kmin = find_bin(smin - pad, edges)
            kmax = find_bin(smax + pad, edges)
            if kmin >= 0 and kmin == kmax:
                _accept_block(
                    acc,
                    slot,
                    sep_c,
                    0.0,
                    edges,
                    moments[a],
                    moments[b],
                    node_patch[a],
                    node_patch[b],
                )
                continue
            if bin_tolerance > 0.0:
                kc = find_bin(sep_c, edges)
                width = 0.0
                if kc >= 0:
                    width = edges[kc + 1] - edges[kc]
                    if not math.isfinite(width):
                        # Open top bin: use the scale of the bin below it.
                        width = edges[kc] - edges[max(kc - 1, 0)] if kc > 0 else 0.0
                if kc >= 0 and 0.5 * (smax - smin) <= bin_tolerance * width:
                    if use_pos:
                        sigma = math.sqrt(moments[a, 5] + moments[b, 5])
                        _accept_block(
                            acc,
                            slot,
                            sep_c,
                            sigma,
                            edges,
                            moments[a],
                            moments[b],
                            node_patch[a],
                            node_patch[b],
                        )
                    else:
                        _accept_spread(
                            acc,
                            slot,
                            smin,
                            sep_c,
                            smax,
                            edges,
                            moments[a],
                            moments[b],
                            node_patch[a],
                            node_patch[b],
                        )
                    continue
            a_leaf = left[a] < 0
            b_leaf = left[b] < 0
            if a_leaf and b_leaf:
                for i in range(start[a], end[a]):
                    for j in range(start[b], end[b]):
                        pair_into(
                            acc,
                            slot,
                            i,
                            j,
                            xyz,
                            f,
                            sig2,
                            w,
                            patch,
                            edges,
                            angular,
                            east,
                            north,
                            pcov,
                            use_pos,
                        )
                continue
            if b_leaf or (not a_leaf and radius[a] >= radius[b]):
                stack[top, 0] = left[a]
                stack[top, 1] = b
                stack[top + 1, 0] = right[a]
                stack[top + 1, 1] = b
            else:
                stack[top, 0] = a
                stack[top, 1] = left[b]
                stack[top + 1, 0] = a
                stack[top + 1, 1] = right[b]
            top += 2
    return acc.sum(axis=0)


def frontier(tree: Tree, min_nodes: int) -> IntArray:
    """Disjoint nodes covering all sources, at least ``min_nodes`` if possible.

    Starts from the patch roots and repeatedly splits the largest node, so
    that work can be shared between threads even with a single patch.
    """
    nodes = [int(r) for r in tree.roots if r >= 0]
    while len(nodes) < min_nodes:
        splittable = [n for n in nodes if tree.left[n] >= 0]
        if not splittable:
            break
        biggest = max(splittable, key=lambda n: tree.end[n] - tree.start[n])
        nodes.remove(biggest)
        nodes.extend((int(tree.left[biggest]), int(tree.right[biggest])))
    return np.asarray(nodes, dtype=np.intp)
