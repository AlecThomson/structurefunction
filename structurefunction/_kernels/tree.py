"""Ball tree with per-node value moments, and the two-point tree kernel.

Every node stores the weighted mean, the weighted sum of squared deviations
and the weighted error variance of its sources. For two separate groups ``A``
and ``B`` the sum over all pairs between them is then exact, without visiting
the pairs, and every term is positive (no loss of precision):

    sum_{i in A, j in B} w_i w_j (f_i - f_j)^2
        = W_B SqDev_A + W_A SqDev_B + W_A W_B (Mean_A - Mean_B)^2

The only approximation when a pair of groups is added at once is the
separation assigned to its pairs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numba
import numpy as np

# Exported at runtime but missing from numba's type information.
from numba import get_thread_id  # type: ignore[attr-defined]

from structurefunction._kernels.brute import add_source_pair
from structurefunction._kernels.common import (
    Contribution,
    add_pair,
    chord_to_separation,
    distance,
    find_bin,
    gaussian_bin_fraction,
    gaussian_bin_range,
)
from structurefunction._kernels.containers import Binning, Moment, Nodes, Sources
from structurefunction.types import FloatArray, IntArray


@dataclass(frozen=True, slots=True)
class Tree:
    """A ball tree with one subtree per patch."""

    order: IntArray
    """Input index of each source in tree order."""
    roots: IntArray
    """Root node of each patch (-1 for an empty patch)."""
    nodes: Nodes


def build_tree(
    positions: FloatArray,
    values: FloatArray,
    error_var: FloatArray,
    weights: FloatArray,
    patch: IntArray,
    n_patch: int,
    max_position_var: FloatArray,
    leaf_size: int = 4,
) -> Tree:
    """Build one subtree per patch, so no node spans two patches.

    Source arrays must afterwards be indexed with ``tree.order``, so that
    each node's sources are contiguous.
    """
    by_patch = np.argsort(patch, kind="stable").astype(np.intp)
    patch_bounds = np.searchsorted(patch[by_patch], np.arange(n_patch + 1))
    nodes, roots, permutation = _build(
        positions[by_patch],
        values[by_patch],
        error_var[by_patch],
        weights[by_patch],
        max_position_var[by_patch],
        patch_bounds.astype(np.intp),
        leaf_size,
    )
    return Tree(order=by_patch[permutation], roots=roots, nodes=nodes)


def frontier(tree: Tree, min_nodes: int) -> IntArray:
    """Non-overlapping nodes that together hold every source.

    Starts from the patch roots and splits the largest node until there are
    at least ``min_nodes``, so the work can be shared between threads even
    with a single patch.
    """
    nodes = [int(root) for root in tree.roots if root >= 0]
    while len(nodes) < min_nodes:
        splittable = [node for node in nodes if tree.nodes.left[node] >= 0]
        if not splittable:
            break
        biggest = max(splittable, key=lambda n: tree.nodes.end[n] - tree.nodes.start[n])
        nodes.remove(biggest)
        left = int(tree.nodes.left[biggest])
        nodes.extend((left, left + 1))
    return np.asarray(nodes, dtype=np.intp)


# --------------------------------------------------------------------------
# Construction


@numba.njit(cache=True)
def _split_axis(positions: FloatArray, members: IntArray) -> int:
    """Axis with the largest spread of ``positions[members]``, or -1 if none."""
    best_axis = -1
    best_spread = 0.0
    for axis in range(3):
        coords = positions[members, axis]
        spread = coords.max() - coords.min()
        if spread > best_spread:
            best_axis = axis
            best_spread = spread
    return best_axis


@numba.njit(cache=True)
def _build(
    positions: FloatArray,
    values: FloatArray,
    error_var: FloatArray,
    weights: FloatArray,
    max_position_var: FloatArray,
    patch_bounds: IntArray,
    leaf_size: int,
) -> tuple[Nodes, IntArray, IntArray]:
    """Median-split construction.

    Returns:
        The nodes, the root of each patch and the tree-order permutation.
    """
    n_sources = positions.shape[0]
    n_patch = patch_bounds.size - 1
    capacity = 4 * (n_sources // max(leaf_size // 2, 1) + 1) + 2 * n_patch + 8
    start = np.empty(capacity, np.intp)
    end = np.empty(capacity, np.intp)
    left = np.full(capacity, -1, np.intp)
    node_patch = np.empty(capacity, np.intp)
    roots = np.full(n_patch, -1, np.intp)
    permutation = np.arange(n_sources)
    to_split = np.empty(capacity, np.intp)
    n_node = 0

    for patch in range(n_patch):
        if patch_bounds[patch + 1] <= patch_bounds[patch]:
            continue
        roots[patch] = n_node
        start[n_node] = patch_bounds[patch]
        end[n_node] = patch_bounds[patch + 1]
        node_patch[n_node] = patch
        n_node += 1
        n_pending = 1
        to_split[0] = roots[patch]
        while n_pending > 0:
            n_pending -= 1
            node = to_split[n_pending]
            first, stop = start[node], end[node]
            if stop - first <= leaf_size:
                continue
            members = permutation[first:stop]
            axis = _split_axis(positions, members)
            if axis < 0:
                continue  # all sources coincide: keep as one leaf
            permutation[first:stop] = members[np.argsort(positions[members, axis])]
            middle = first + (stop - first) // 2
            left[node] = n_node
            for child_start, child_end in ((first, middle), (middle, stop)):
                start[n_node] = child_start
                end[n_node] = child_end
                node_patch[n_node] = patch
                to_split[n_pending] = n_node
                n_pending += 1
                n_node += 1

    centre, radius, moments = _node_summaries(
        start[:n_node],
        end[:n_node],
        positions[permutation],
        values[permutation],
        error_var[permutation],
        weights[permutation],
        max_position_var[permutation],
    )
    nodes = Nodes(
        start[:n_node],
        end[:n_node],
        left[:n_node],
        centre,
        radius,
        moments,
        node_patch[:n_node],
    )
    return nodes, roots, permutation


@numba.njit(cache=True)
def _node_summaries(
    start: IntArray,
    end: IntArray,
    positions: FloatArray,
    values: FloatArray,
    error_var: FloatArray,
    weights: FloatArray,
    max_position_var: FloatArray,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Centre, radius and value moments of every node (sources in tree order)."""
    n_node = start.size
    centre = np.zeros((n_node, 3))
    radius = np.zeros(n_node)
    # MAX_POSITION_VAR is the last column.
    moments = np.zeros((n_node, Moment.MAX_POSITION_VAR + 1))
    for node in range(n_node):
        first, stop = start[node], end[node]
        for axis in range(3):
            centre[node, axis] = positions[first:stop, axis].mean()
        largest = 0.0
        for i in range(first, stop):
            largest = max(largest, distance(positions, i, centre, node))
        radius[node] = largest

        node_weights = weights[first:stop]
        total_weight = node_weights.sum()
        mean = 0.0
        if total_weight > 0.0:
            mean = (node_weights * values[first:stop]).sum() / total_weight
        deviations = values[first:stop] - mean
        moments[node, Moment.WEIGHT] = total_weight
        moments[node, Moment.MEAN] = mean
        moments[node, Moment.SQUARED_DEVIATION] = (
            node_weights * deviations * deviations
        ).sum()
        moments[node, Moment.ERROR_VAR] = (node_weights * error_var[first:stop]).sum()
        moments[node, Moment.COUNT] = stop - first
        moments[node, Moment.MAX_POSITION_VAR] = max_position_var[first:stop].max()
    return centre, radius, moments


# --------------------------------------------------------------------------
# Adding a whole pair of nodes at once


@numba.njit(cache=True, inline="always")
def _group_pair_contribution(
    nodes: Nodes, node_a: int, node_b: int, separation: float
) -> Contribution:
    """Sum over all pairs between two nodes, placed at ``separation``."""
    moments = nodes.moments
    weight_a = moments[node_a, Moment.WEIGHT]
    weight_b = moments[node_b, Moment.WEIGHT]
    mean_difference = moments[node_a, Moment.MEAN] - moments[node_b, Moment.MEAN]
    weight = weight_a * weight_b
    squared_diff = (
        weight_b * moments[node_a, Moment.SQUARED_DEVIATION]
        + weight_a * moments[node_b, Moment.SQUARED_DEVIATION]
        + weight * mean_difference * mean_difference
    )
    error_bias = (
        weight_b * moments[node_a, Moment.ERROR_VAR]
        + weight_a * moments[node_b, Moment.ERROR_VAR]
    )
    count = moments[node_a, Moment.COUNT] * moments[node_b, Moment.COUNT]
    return Contribution(weight, squared_diff, error_bias, weight * separation, count)


@numba.njit(cache=True, inline="always")
def _add_node_pair_to_bin(
    acc: FloatArray,
    slot: int,
    bin_index: int,
    fraction: float,
    separation: float,
    nodes: Nodes,
    node_a: int,
    node_b: int,
) -> None:
    contribution = _group_pair_contribution(nodes, node_a, node_b, separation)
    add_pair(
        acc,
        slot,
        bin_index,
        nodes.patch[node_a],
        nodes.patch[node_b],
        contribution,
        fraction,
    )


@numba.njit(cache=True, inline="always")
def _triangle_cdf(x: float, low: float, peak: float, high: float) -> float:
    """CDF of the triangular distribution on ``[low, high]`` peaking at ``peak``."""
    if x <= low:
        return 0.0
    if x >= high:
        return 1.0
    if x <= peak:
        return (x - low) ** 2 / ((high - low) * (peak - low))
    return 1.0 - (high - x) ** 2 / ((high - low) * (high - peak))


@numba.njit(cache=True, inline="always")
def _spread_node_pair(
    acc: FloatArray,
    slot: int,
    nodes: Nodes,
    node_a: int,
    node_b: int,
    closest: float,
    centre_separation: float,
    farthest: float,
    edges: FloatArray,
) -> None:
    """Share a node pair between the bins its separations cover.

    The pairs' separations are modelled as triangular between the closest and
    farthest possible values, peaking at the centre separation.
    """
    tiny = 1e-15 * farthest
    peak = min(max(centre_separation, closest + tiny), farthest - tiny)
    first = find_bin(max(closest, edges[0]), edges)
    last = find_bin(min(farthest, np.nextafter(edges[-1], 0.0)), edges)
    if first < 0 or last < 0:
        return
    for bin_index in range(first, last + 1):
        low = max(closest, edges[bin_index])
        high = min(farthest, edges[bin_index + 1])
        fraction = _triangle_cdf(high, closest, peak, farthest) - _triangle_cdf(
            low, closest, peak, farthest
        )
        if fraction > 0.0:
            _add_node_pair_to_bin(
                acc,
                slot,
                bin_index,
                fraction,
                0.5 * (low + high),
                nodes,
                node_a,
                node_b,
            )


@numba.njit(cache=True, inline="always")
def _blur_node_pair(
    acc: FloatArray,
    slot: int,
    nodes: Nodes,
    node_a: int,
    node_b: int,
    separation: float,
    sigma: float,
    edges: FloatArray,
) -> None:
    """Spread a node pair over bins with a Gaussian of width ``sigma``."""
    first, last = gaussian_bin_range(separation, sigma, edges)
    if first < 0:
        return
    for bin_index in range(first, last + 1):
        fraction = gaussian_bin_fraction(separation, sigma, edges, bin_index)
        if fraction > 0.0:
            _add_node_pair_to_bin(
                acc, slot, bin_index, fraction, separation, nodes, node_a, node_b
            )


@numba.njit(cache=True, inline="always")
def _bin_width(edges: FloatArray, bin_index: int) -> float:
    """Width of a bin; the open top bin borrows the width of the one below."""
    width: float = edges[bin_index + 1] - edges[bin_index]
    if math.isfinite(width):
        return width
    if bin_index == 0:
        return 0.0
    below: float = edges[bin_index] - edges[bin_index - 1]
    return below


@numba.njit(cache=True)
def _try_whole_node_pair(
    acc: FloatArray,
    slot: int,
    nodes: Nodes,
    node_a: int,
    node_b: int,
    bins: Binning,
) -> bool:
    """Add all pairs between two nodes at once, if that is accurate enough.

    Returns:
        Whether the node pair was handled (added, or entirely out of range).
    """
    centre_distance = distance(nodes.centre, node_a, nodes.centre, node_b)
    radii = nodes.radius[node_a] + nodes.radius[node_b]
    centre_separation = chord_to_separation(centre_distance, bins.angular)
    closest = chord_to_separation(max(centre_distance - radii, 0.0), bins.angular)
    farthest = chord_to_separation(centre_distance + radii, bins.angular)
    position_sigma = 0.0
    if bins.use_position_errors:
        position_sigma = math.sqrt(
            nodes.moments[node_a, Moment.MAX_POSITION_VAR]
            + nodes.moments[node_b, Moment.MAX_POSITION_VAR]
        )
    margin = 5.0 * position_sigma
    edges = bins.edges
    if farthest + margin < edges[0] or closest - margin >= edges[-1]:
        return True  # no pair can land in any bin

    # Exact: every pair (allowing for position errors) is in the same bin.
    first = find_bin(closest - margin, edges)
    if first >= 0 and first == find_bin(farthest + margin, edges):
        _add_node_pair_to_bin(
            acc, slot, first, 1.0, centre_separation, nodes, node_a, node_b
        )
        return True

    # Approximate: the spread of separations is small compared with the bin.
    if bins.tolerance <= 0.0:
        return False
    centre_bin = find_bin(centre_separation, edges)
    if centre_bin < 0:
        return False
    spread = 0.5 * (farthest - closest)
    if spread > bins.tolerance * _bin_width(edges, centre_bin):
        return False
    if bins.use_position_errors:
        _blur_node_pair(
            acc, slot, nodes, node_a, node_b, centre_separation, position_sigma, edges
        )
    else:
        _spread_node_pair(
            acc,
            slot,
            nodes,
            node_a,
            node_b,
            closest,
            centre_separation,
            farthest,
            edges,
        )
    return True


# --------------------------------------------------------------------------
# Traversal


@numba.njit(cache=True)
def _add_pairs_within_leaf(
    acc: FloatArray, slot: int, node: int, nodes: Nodes, sources: Sources, bins: Binning
) -> None:
    for i in range(nodes.start[node], nodes.end[node]):
        for j in range(i + 1, nodes.end[node]):
            add_source_pair(acc, slot, i, j, sources, bins)


@numba.njit(cache=True)
def _add_pairs_between_leaves(
    acc: FloatArray,
    slot: int,
    node_a: int,
    node_b: int,
    nodes: Nodes,
    sources: Sources,
    bins: Binning,
) -> None:
    for i in range(nodes.start[node_a], nodes.end[node_a]):
        for j in range(nodes.start[node_b], nodes.end[node_b]):
            add_source_pair(acc, slot, i, j, sources, bins)


@numba.njit(cache=True)
def _push(todo: IntArray, n_todo: int, node_a: int, node_b: int) -> int:
    todo[n_todo, 0] = node_a
    todo[n_todo, 1] = node_b
    return n_todo + 1


@numba.njit(cache=True)
def _sum_node_pair(
    acc: FloatArray,
    slot: int,
    start_a: int,
    start_b: int,
    nodes: Nodes,
    sources: Sources,
    bins: Binning,
) -> None:
    """All pairs with one source under ``start_a`` and one under ``start_b``.

    With ``start_a == start_b``, all pairs within that node.
    """
    todo = np.empty((64, 2), np.intp)
    n_todo = _push(todo, 0, start_a, start_b)
    while n_todo > 0:
        if n_todo + 3 > todo.shape[0]:
            bigger = np.empty((2 * todo.shape[0], 2), np.intp)
            bigger[:n_todo] = todo[:n_todo]
            todo = bigger
        n_todo -= 1
        node_a, node_b = todo[n_todo, 0], todo[n_todo, 1]
        a_is_leaf = nodes.left[node_a] < 0
        b_is_leaf = nodes.left[node_b] < 0

        if node_a == node_b:
            if a_is_leaf:
                _add_pairs_within_leaf(acc, slot, node_a, nodes, sources, bins)
            else:
                child = nodes.left[node_a]
                n_todo = _push(todo, n_todo, child, child)
                n_todo = _push(todo, n_todo, child + 1, child + 1)
                n_todo = _push(todo, n_todo, child, child + 1)
            continue

        if _try_whole_node_pair(acc, slot, nodes, node_a, node_b, bins):
            continue
        if a_is_leaf and b_is_leaf:
            _add_pairs_between_leaves(acc, slot, node_a, node_b, nodes, sources, bins)
            continue
        # Split the larger node.
        if b_is_leaf or (
            not a_is_leaf and nodes.radius[node_a] >= nodes.radius[node_b]
        ):
            child = nodes.left[node_a]
            n_todo = _push(todo, n_todo, child, node_b)
            n_todo = _push(todo, n_todo, child + 1, node_b)
        else:
            child = nodes.left[node_b]
            n_todo = _push(todo, n_todo, node_a, child)
            n_todo = _push(todo, n_todo, node_a, child + 1)


@numba.njit(cache=True, parallel=True)
def tree_pairs(
    nodes: Nodes,
    sources: Sources,
    bins: Binning,
    start_nodes: IntArray,
    acc: FloatArray,
) -> FloatArray:
    """Two-point sums over all pairs, using the tree.

    Each pair of ``start_nodes`` (including each node with itself) is one
    parallel task. The start nodes hold every source exactly once, so every
    source pair is counted once.

    Returns:
        Accumulator summed over threads, shape ``(kind, bin, patch, stat)``.
    """
    n_start = start_nodes.size
    n_tasks = n_start * (n_start + 1) // 2
    task_a = np.empty(n_tasks, np.intp)
    task_b = np.empty(n_tasks, np.intp)
    task = 0
    for index_a in range(n_start):
        for index_b in range(index_a, n_start):
            task_a[task] = start_nodes[index_a]
            task_b[task] = start_nodes[index_b]
            task += 1
    for task in numba.prange(n_tasks):
        _sum_node_pair(
            acc, get_thread_id(), task_a[task], task_b[task], nodes, sources, bins
        )
    return acc.sum(axis=0)
