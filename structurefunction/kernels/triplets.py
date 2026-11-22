"""Three-point stencil kernels: SF3(r) = <[f(x+) - 2 f(x0) + f(x-)]^2>.

For every middle source ``x0`` the tree is searched for neighbours at the
distances of interest. Bearings are measured in the plane tangent to the sky
at ``x0``, where great circles through ``x0`` are straight lines through the
origin.

* ``exact``: every pair of neighbours on opposite sides (bearings differing
  from 180 degrees by at most ``angle_tol``) whose distances from ``x0`` agree
  to ``sep_tol``.
* ``binned``: neighbours (or whole tree nodes, when they fit) are summed into
  cells of a polar grid around ``x0``. Each cell is combined with the cells on
  the opposite side. With ``a_i = f_i - f0`` the sum over all pairs of two
  cells ``A`` and ``B`` is exact:

      sum_{i in A, j in B} w_i w_j (a_i + a_j)^2
          = W_B SumSq_A + W_A SumSq_B + 2 Sum_A Sum_B,

  where ``Sum = sum w a`` and ``SumSq = sum w a^2``.
"""

from __future__ import annotations

import math
from enum import IntEnum
from typing import NamedTuple

import numba
import numpy as np

from structurefunction.kernels.brute import position_var_towards
from structurefunction.kernels.common import (
    Contribution,
    add_triplet,
    chord_to_separation,
    distance,
    find_bin,
    gaussian_bin_fraction,
    gaussian_bin_range,
)
from structurefunction.kernels.containers import (
    Binning,
    Moment,
    Nodes,
    PolarGrid,
    Sources,
)
from structurefunction.types import BoolArray, FloatArray, IntArray


class Offset(NamedTuple):
    """A 3D vector from the middle source."""

    x: float
    """First Cartesian component."""
    y: float
    """Second Cartesian component."""
    z: float
    """Third Cartesian component."""


@numba.njit(cache=True)
def _grow(array: IntArray) -> IntArray:
    bigger = np.empty(2 * array.size, array.dtype)
    bigger[: array.size] = array
    return bigger


@numba.njit(cache=True, inline="always")
def _offset(points: FloatArray, index: int, sources: Sources, middle: int) -> Offset:
    """Vector from the middle source to row ``index`` of ``points``."""
    return Offset(
        points[index, 0] - sources.positions[middle, 0],
        points[index, 1] - sources.positions[middle, 1],
        points[index, 2] - sources.positions[middle, 2],
    )


@numba.njit(cache=True, inline="always")
def _length(offset: Offset) -> float:
    return math.sqrt(offset.x * offset.x + offset.y * offset.y + offset.z * offset.z)


@numba.njit(cache=True, inline="always")
def _tangent_components(
    offset: Offset, sources: Sources, middle: int
) -> tuple[float, float]:
    """Components of ``offset`` towards east and north at the middle source."""
    east = sources.east
    north = sources.north
    towards_east = (
        offset.x * east[middle, 0]
        + offset.y * east[middle, 1]
        + offset.z * east[middle, 2]
    )
    towards_north = (
        offset.x * north[middle, 0]
        + offset.y * north[middle, 1]
        + offset.z * north[middle, 2]
    )
    return towards_east, towards_north


@numba.njit(cache=True, inline="always")
def _bearing(offset: Offset, sources: Sources, middle: int) -> float:
    """Bearing of ``offset`` in [0, 2 pi), measured from north through east."""
    towards_east, towards_north = _tangent_components(offset, sources, middle)
    bearing = math.atan2(towards_east, towards_north)
    return bearing + 2.0 * math.pi if bearing < 0.0 else bearing


@numba.njit(cache=True)
def _node_out_of_range(
    node: int, nodes: Nodes, sources: Sources, middle: int, inner: float, outer: float
) -> bool:
    """Whether no source in ``node`` can be between ``inner`` and ``outer``."""
    to_centre = distance(nodes.centre, node, sources.positions, middle)
    radius = nodes.radius[node]
    if chord_to_separation(to_centre + radius, angular=True) < inner:
        return True
    nearest = chord_to_separation(max(to_centre - radius, 0.0), angular=True)
    return nearest >= outer


@numba.njit(cache=True)
def _neighbours_in_annulus(
    middle: int,
    nodes: Nodes,
    roots: IntArray,
    sources: Sources,
    inner: float,
    outer: float,
    found: IntArray,
) -> tuple[IntArray, int]:
    """Sources whose separation from ``middle`` is in ``[inner, outer)``."""
    n_found = 0
    todo = np.empty(64, np.intp)
    for root in roots:
        if root < 0:
            continue
        todo[0] = root
        n_todo = 1
        while n_todo > 0:
            n_todo -= 1
            node = todo[n_todo]
            if _node_out_of_range(node, nodes, sources, middle, inner, outer):
                continue
            if nodes.left[node] >= 0:
                if n_todo + 2 > todo.size:
                    todo = _grow(todo)
                todo[n_todo] = nodes.left[node]
                todo[n_todo + 1] = nodes.left[node] + 1
                n_todo += 2
                continue
            for source in range(nodes.start[node], nodes.end[node]):
                if source == middle:
                    continue
                chord = distance(sources.positions, source, sources.positions, middle)
                separation = chord_to_separation(chord, angular=True)
                if inner <= separation < outer:
                    if n_found >= found.size:
                        found = _grow(found)
                    found[n_found] = source
                    n_found += 1
    return found, n_found


# --------------------------------------------------------------------------
# Exact mode


@numba.njit(cache=True, inline="always")
def _add_triplet_to_bins(
    acc: FloatArray,
    slot: int,
    separation: float,
    sigma: float,
    edges: FloatArray,
    regions: tuple[int, int, int],
    contribution: Contribution,
) -> None:
    """Add a triplet to its bin, spreading it if its separation is uncertain."""
    region_middle, region_a, region_b = regions
    if sigma <= 0.0:
        bin_index = find_bin(separation, edges)
        if bin_index >= 0:
            add_triplet(
                acc,
                slot,
                bin_index,
                region_middle,
                region_a,
                region_b,
                contribution,
                1.0,
            )
        return
    first, last = gaussian_bin_range(separation, sigma, edges)
    if first < 0:
        return
    for bin_index in range(first, last + 1):
        fraction = gaussian_bin_fraction(separation, sigma, edges, bin_index)
        if fraction > 0.0:
            add_triplet(
                acc,
                slot,
                bin_index,
                region_middle,
                region_a,
                region_b,
                contribution,
                fraction,
            )


@numba.njit(cache=True, inline="always")
def _add_source_triplet(
    acc: FloatArray,
    slot: int,
    middle: int,
    a: int,
    b: int,
    arm_a: float,
    arm_b: float,
    sources: Sources,
    bins: Binning,
) -> None:
    """Add the triplet ``a - middle - b`` with arm lengths ``arm_a``, ``arm_b``."""
    separation = 0.5 * (arm_a + arm_b)
    sigma = 0.0
    if bins.use_position_errors:
        # The middle source's error moves both arms equally and cancels in
        # their mean; the outer sources' errors along the arms remain.
        sigma = 0.5 * math.sqrt(
            position_var_towards(a, middle, sources)
            + position_var_towards(b, middle, sources)
        )
    weight = sources.weights[middle] * sources.weights[a] * sources.weights[b]
    stencil = sources.values[a] + sources.values[b] - 2.0 * sources.values[middle]
    error_bias = (
        sources.error_var[a] + sources.error_var[b] + 4.0 * sources.error_var[middle]
    )
    contribution = Contribution(
        weight,
        weight * stencil * stencil,
        weight * error_bias,
        weight * separation,
        1.0,
    )
    regions = (sources.region[middle], sources.region[a], sources.region[b])
    _add_triplet_to_bins(
        acc, slot, separation, sigma, bins.edges, regions, contribution
    )


@numba.njit(cache=True)
def _add_triplets_around(
    acc: FloatArray,
    slot: int,
    middle: int,
    neighbours: IntArray,
    sources: Sources,
    bins: Binning,
    angle_tol: float,
    sep_tol: float,
) -> None:
    """All accepted triplets with ``middle`` as the middle source."""
    n_neighbours = neighbours.size
    bearings = np.empty(n_neighbours)
    arms = np.empty(n_neighbours)
    for k in range(n_neighbours):
        offset = _offset(sources.positions, neighbours[k], sources, middle)
        bearings[k] = _bearing(offset, sources, middle)
        arms[k] = chord_to_separation(_length(offset), angular=True)
    order = np.argsort(bearings)
    neighbours = neighbours[order]
    bearings = bearings[order]
    arms = arms[order]
    # A second lap so that windows crossing 2 pi are contiguous.
    two_laps = np.concatenate((bearings, bearings + 2.0 * math.pi))

    for k_a in range(n_neighbours):
        opposite = bearings[k_a] + math.pi
        window_start = np.searchsorted(two_laps, opposite - angle_tol, side="left")
        window_end = np.searchsorted(two_laps, opposite + angle_tol, side="right")
        for k_lap in range(window_start, window_end):
            k_b = k_lap % n_neighbours
            source_a = neighbours[k_a]
            source_b = neighbours[k_b]
            if source_a >= source_b:
                continue  # each unordered pair of outer sources once
            arm_a = arms[k_a]
            arm_b = arms[k_b]
            if abs(arm_a - arm_b) > sep_tol * 0.5 * (arm_a + arm_b):
                continue
            _add_source_triplet(
                acc, slot, middle, source_a, source_b, arm_a, arm_b, sources, bins
            )


@numba.njit(cache=True, parallel=True)
def exact_triplets(
    nodes: Nodes,
    roots: IntArray,
    sources: Sources,
    bins: Binning,
    angle_tol: float,
    sep_tol: float,
    acc: FloatArray,
) -> FloatArray:
    """Three-point sums by finding every triplet (sky positions only).

    Returns:
        Accumulator summed over threads, shape ``(kind, bin, region, stat)``.
    """
    n_sources = sources.values.size
    n_slots = acc.shape[0]
    # Arms may differ from the mean separation by up to sep_tol.
    inner = bins.edges[0] * (1.0 - sep_tol)
    outer = bins.edges[-1] * (1.0 + sep_tol)
    for slot in numba.prange(n_slots):
        found = np.empty(1024, np.intp)
        for middle in range(slot, n_sources, n_slots):
            found, n_found = _neighbours_in_annulus(
                middle, nodes, roots, sources, inner, outer, found
            )
            if n_found < 2:
                continue
            _add_triplets_around(
                acc,
                slot,
                middle,
                found[:n_found].copy(),
                sources,
                bins,
                angle_tol,
                sep_tol,
            )
    return acc.sum(axis=0)


# --------------------------------------------------------------------------
# Binned mode


class CellStat(IntEnum):
    """Per-cell sums, relative to the middle source's value ``f0``."""

    WEIGHT = 0
    """``sum w``"""
    OFFSET = 1
    """``sum w (f - f0)``"""
    OFFSET_SQ = 2
    """``sum w (f - f0)^2``"""
    ERROR_VAR = 3
    """``sum w sigma^2``"""
    SEPARATION = 4
    """``sum w * distance from the middle source``"""
    COUNT = 5
    """Number of sources."""


class Cells(NamedTuple):
    """Cell sums around one middle source, per region, plus bookkeeping.

    Only touched cells are visited when pairing and cleared afterwards, so the
    cost per middle source does not depend on the size of the grid.
    """

    sums: FloatArray
    """``(n_regions, n_cell, len(CellStat))``"""
    touched: BoolArray
    """``(n_regions, n_cell)``"""
    touched_region: IntArray
    """Region of each touched cell, in the order touched."""
    touched_cell: IntArray
    """Index of each touched cell, in the order touched."""
    region_seen: BoolArray
    """Whether each region has a touched cell, ``(n_regions,)``."""
    regions: IntArray
    """Regions touched so far, in order."""
    counts: IntArray
    """Number of touched cells, number of touched regions."""


@numba.njit(cache=True)
def _new_cells(n_regions: int, n_cell: int) -> Cells:
    capacity = n_regions * n_cell
    return Cells(
        np.zeros((n_regions, n_cell, CellStat.COUNT + 1)),  # COUNT is the last column
        np.zeros((n_regions, n_cell), np.bool_),
        np.empty(capacity, np.intp),
        np.empty(capacity, np.intp),
        np.zeros(n_regions, np.bool_),
        np.empty(n_regions, np.intp),
        np.zeros(2, np.intp),
    )


@numba.njit(cache=True, inline="always")
def _touch(cells: Cells, region: int, cell: int) -> None:
    """Record a cell (and its region) the first time it receives a source."""
    if not cells.touched[region, cell]:
        cells.touched[region, cell] = True
        n_touched = cells.counts[0]
        cells.touched_region[n_touched] = region
        cells.touched_cell[n_touched] = cell
        cells.counts[0] = n_touched + 1
        if not cells.region_seen[region]:
            cells.region_seen[region] = True
            cells.regions[cells.counts[1]] = region
            cells.counts[1] += 1


@numba.njit(cache=True, inline="always")
def _add_to_cell(
    cells: Cells,
    region: int,
    cell: int,
    weight: float,
    offset_sum: float,
    offset_sq_sum: float,
    error_var: float,
    weighted_separation: float,
    count: float,
) -> None:
    _touch(cells, region, cell)
    sums = cells.sums
    sums[region, cell, CellStat.WEIGHT] += weight
    sums[region, cell, CellStat.OFFSET] += offset_sum
    sums[region, cell, CellStat.OFFSET_SQ] += offset_sq_sum
    sums[region, cell, CellStat.ERROR_VAR] += error_var
    sums[region, cell, CellStat.SEPARATION] += weighted_separation
    sums[region, cell, CellStat.COUNT] += count


@numba.njit(cache=True, inline="always")
def _sector(bearing: float, n_sector: int) -> int:
    """Sector holding ``bearing``; sectors are centred on multiples of the width."""
    width = 2.0 * math.pi / n_sector
    return int((bearing + 0.5 * width) // width) % n_sector


@numba.njit(cache=True, inline="always")
def _node_cell(
    middle: int, node: int, nodes: Nodes, sources: Sources, grid: PolarGrid
) -> int:
    """Cell holding the whole node, or -1 if the node may span several cells."""
    offset = _offset(nodes.centre, node, sources, middle)
    to_centre = _length(offset)
    radius = nodes.radius[node]
    if to_centre <= radius:
        return -1  # the node may contain the middle source
    nearest = chord_to_separation(to_centre - radius, angular=True)
    farthest = chord_to_separation(to_centre + radius, angular=True)
    ring = find_bin(nearest, grid.ring_edges)
    if ring < 0 or ring != find_bin(farthest, grid.ring_edges):
        return -1
    towards_east, towards_north = _tangent_components(offset, sources, middle)
    in_plane = math.hypot(towards_east, towards_north)
    if in_plane <= radius:
        return -1
    bearing = _bearing(offset, sources, middle)
    half_width = math.asin(radius / in_plane)
    sector = _sector(bearing - half_width, grid.n_sector)
    if sector != _sector(bearing + half_width, grid.n_sector):
        return -1
    return ring * grid.n_sector + sector


@numba.njit(cache=True, inline="always")
def _add_node_to_cell(
    cells: Cells,
    cell: int,
    node: int,
    nodes: Nodes,
    middle_value: float,
    separation: float,
) -> None:
    moments = nodes.moments
    weight = moments[node, Moment.WEIGHT]
    mean_offset = moments[node, Moment.MEAN] - middle_value
    _add_to_cell(
        cells,
        nodes.region[node],
        cell,
        weight,
        weight * mean_offset,
        moments[node, Moment.SQUARED_DEVIATION] + weight * mean_offset**2,
        moments[node, Moment.ERROR_VAR],
        weight * separation,
        moments[node, Moment.COUNT],
    )


@numba.njit(cache=True, inline="always")
def _add_source_to_cell(
    cells: Cells, middle: int, source: int, sources: Sources, grid: PolarGrid
) -> None:
    offset = _offset(sources.positions, source, sources, middle)
    separation = chord_to_separation(_length(offset), angular=True)
    ring = find_bin(separation, grid.ring_edges)
    if ring < 0:
        return
    bearing = _bearing(offset, sources, middle)
    cell = ring * grid.n_sector + _sector(bearing, grid.n_sector)
    weight = sources.weights[source]
    value_offset = sources.values[source] - sources.values[middle]
    _add_to_cell(
        cells,
        sources.region[source],
        cell,
        weight,
        weight * value_offset,
        weight * value_offset**2,
        weight * sources.error_var[source],
        weight * separation,
        1.0,
    )


@numba.njit(cache=True)
def _fill_cells(
    middle: int,
    cells: Cells,
    nodes: Nodes,
    roots: IntArray,
    sources: Sources,
    grid: PolarGrid,
) -> None:
    """Sum every source near ``middle`` into its cell, whole nodes where possible."""
    middle_value = sources.values[middle]
    inner = grid.ring_edges[0]
    outer = grid.ring_edges[-1]
    todo = np.empty(64, np.intp)
    for root in roots:
        if root < 0:
            continue
        todo[0] = root
        n_todo = 1
        while n_todo > 0:
            n_todo -= 1
            node = todo[n_todo]
            if _node_out_of_range(node, nodes, sources, middle, inner, outer):
                continue
            cell = _node_cell(middle, node, nodes, sources, grid)
            if cell >= 0:
                to_centre = distance(nodes.centre, node, sources.positions, middle)
                separation = chord_to_separation(to_centre, angular=True)
                _add_node_to_cell(cells, cell, node, nodes, middle_value, separation)
            elif nodes.left[node] >= 0:
                if n_todo + 2 > todo.size:
                    todo = _grow(todo)
                todo[n_todo] = nodes.left[node]
                todo[n_todo + 1] = nodes.left[node] + 1
                n_todo += 2
            else:
                for source in range(nodes.start[node], nodes.end[node]):
                    if source != middle:
                        _add_source_to_cell(cells, middle, source, sources, grid)


@numba.njit(cache=True, inline="always")
def _cell_pair_contribution(
    sums: FloatArray,
    region_a: int,
    cell_a: int,
    region_b: int,
    cell_b: int,
    middle_weight: float,
    middle_error_var: float,
) -> Contribution:
    """Sum over all triplets with one outer source in each of two cells."""
    weight_a = sums[region_a, cell_a, CellStat.WEIGHT]
    weight_b = sums[region_b, cell_b, CellStat.WEIGHT]
    squared_diff = (
        weight_b * sums[region_a, cell_a, CellStat.OFFSET_SQ]
        + weight_a * sums[region_b, cell_b, CellStat.OFFSET_SQ]
        + 2.0
        * sums[region_a, cell_a, CellStat.OFFSET]
        * sums[region_b, cell_b, CellStat.OFFSET]
    )
    error_bias = (
        weight_b * sums[region_a, cell_a, CellStat.ERROR_VAR]
        + weight_a * sums[region_b, cell_b, CellStat.ERROR_VAR]
        + 4.0 * middle_error_var * weight_a * weight_b
    )
    separation = 0.5 * (
        weight_b * sums[region_a, cell_a, CellStat.SEPARATION]
        + weight_a * sums[region_b, cell_b, CellStat.SEPARATION]
    )
    count = (
        sums[region_a, cell_a, CellStat.COUNT] * sums[region_b, cell_b, CellStat.COUNT]
    )
    return Contribution(
        middle_weight * weight_a * weight_b,
        middle_weight * squared_diff,
        middle_weight * error_bias,
        middle_weight * separation,
        count,
    )


@numba.njit(cache=True, inline="always")
def _ring_pair_bin(ring_a: int, ring_b: int, grid: PolarGrid, edges: FloatArray) -> int:
    """Output bin of triplets with arms in two (equal or adjacent) rings."""
    if ring_a == ring_b:
        return int(grid.ring_to_bin[ring_a])
    return find_bin(math.sqrt(grid.ring_mid[ring_a] * grid.ring_mid[ring_b]), edges)


@numba.njit(cache=True)
def _pair_cells(
    acc: FloatArray,
    slot: int,
    middle: int,
    cells: Cells,
    sources: Sources,
    grid: PolarGrid,
    edges: FloatArray,
) -> None:
    """Combine every touched cell with the opposite cell and its neighbours.

    Cells are half a tolerance wide in bearing and distance, so a cell and the
    opposite cell's neighbours stay within the tolerances. Every unordered
    cell pair is reached from both ends; only the one whose first sector is
    smaller is kept.
    """
    half_turn = grid.n_sector // 2
    n_rings = grid.ring_mid.size
    middle_weight = sources.weights[middle]
    middle_error_var = sources.error_var[middle]
    middle_region = sources.region[middle]
    for touched in range(cells.counts[0]):
        region_a = cells.touched_region[touched]
        cell_a = cells.touched_cell[touched]
        ring_a = cell_a // grid.n_sector
        sector_a = cell_a % grid.n_sector
        for ring_b in range(max(ring_a - 1, 0), min(ring_a + 2, n_rings)):
            bin_index = _ring_pair_bin(ring_a, ring_b, grid, edges)
            if bin_index < 0:
                continue
            for step in range(-1, 2):
                sector_b = (sector_a + half_turn + step) % grid.n_sector
                if sector_a >= sector_b:
                    continue
                cell_b = ring_b * grid.n_sector + sector_b
                for k in range(cells.counts[1]):
                    region_b = cells.regions[k]
                    if not cells.touched[region_b, cell_b]:
                        continue
                    contribution = _cell_pair_contribution(
                        cells.sums,
                        region_a,
                        cell_a,
                        region_b,
                        cell_b,
                        middle_weight,
                        middle_error_var,
                    )
                    if contribution.weight > 0.0:
                        add_triplet(
                            acc,
                            slot,
                            bin_index,
                            middle_region,
                            region_a,
                            region_b,
                            contribution,
                            1.0,
                        )


@numba.njit(cache=True)
def _clear_cells(cells: Cells) -> None:
    for touched in range(cells.counts[0]):
        region = cells.touched_region[touched]
        cell = cells.touched_cell[touched]
        cells.sums[region, cell, :] = 0.0
        cells.touched[region, cell] = False
    for k in range(cells.counts[1]):
        cells.region_seen[cells.regions[k]] = False
    cells.counts[0] = 0
    cells.counts[1] = 0


@numba.njit(cache=True, parallel=True)
def binned_triplets(
    nodes: Nodes,
    roots: IntArray,
    sources: Sources,
    edges: FloatArray,
    grid: PolarGrid,
    n_regions: int,
    acc: FloatArray,
) -> FloatArray:
    """Three-point sums from a polar grid of cells around every source.

    Returns:
        Accumulator summed over threads, shape ``(kind, bin, region, stat)``.
    """
    n_sources = sources.values.size
    n_slots = acc.shape[0]
    n_cell = grid.ring_mid.size * grid.n_sector
    for slot in numba.prange(n_slots):
        cells = _new_cells(n_regions, n_cell)
        for middle in range(slot, n_sources, n_slots):
            _fill_cells(middle, cells, nodes, roots, sources, grid)
            _pair_cells(acc, slot, middle, cells, sources, grid, edges)
            _clear_cells(cells)
    return acc.sum(axis=0)
