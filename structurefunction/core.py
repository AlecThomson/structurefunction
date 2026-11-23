"""Public entry points for computing structure functions."""

from __future__ import annotations

import itertools
import math
import warnings
from typing import TYPE_CHECKING, Literal

import astropy.units as u
import numba
import numpy as np
from tqdm.auto import tqdm

from structurefunction import units
from structurefunction.estimators import estimate
from structurefunction.geometry import (
    chord_to_separation,
    log_edges,
    separation_range,
    tangent_basis,
)
from structurefunction.kernels.common import new_accumulator
from structurefunction.kernels.containers import Binning, PolarGrid, Sources
from structurefunction.kernels.tree import (
    Tree,
    build_tree,
    frontier,
    node_pair_tasks,
    tree_pairs,
)
from structurefunction.kernels.triplets import binned_triplets, exact_triplets
from structurefunction.logging import StructureFunctionWarning, logger
from structurefunction.options import StencilOptions
from structurefunction.regions import make_regions, recommend_jk_regions
from structurefunction.results import SFResult
from structurefunction.types import (
    CrossRegionWeight,
    CrossRegionWeightLike,
    ErrorMethod,
    ErrorMethodLike,
    FloatArray,
    IntArray,
    StencilMode,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from astropy.coordinates import SkyCoord
    from astropy.table import Table
    from numpy.typing import ArrayLike

    from structurefunction.options import PositionErrors


def structure_function(
    values: ArrayLike | u.Quantity,
    errors: ArrayLike | u.Quantity | None,
    coords: SkyCoord,
    n_point: Literal[2, 3] = 2,
    bins: int | u.Quantity = 10,
    min_sep: u.Quantity | None = None,
    max_sep: u.Quantity | None = None,
    weights: ArrayLike | None = None,
    jk_regions: int | ArrayLike | Literal["auto"] = "auto",
    error_method: ErrorMethodLike = ErrorMethod.JACKKNIFE,
    cross_region_weight: CrossRegionWeightLike = CrossRegionWeight.SIMPLE,
    n_boot: int = 1000,
    bin_tolerance: float = 0.05,
    pos_errors: PositionErrors | None = None,
    stencil: StencilOptions | None = None,
    min_count: int = 10,
    rng: np.random.Generator | int | None = None,
    progress: bool = True,
) -> SFResult:
    """Second-order structure function with a 2- or 3-point stencil.

    The noise bias ``<sum_i c_i^2 sigma_i^2>`` is subtracted exactly. The
    covariance between bins comes from jackknife regions (compact areas of sky
    left out in turn; see :mod:`structurefunction.regions`), so it
    includes measurement noise, sample variance and bin-to-bin correlation.

    Args:
        values: Measured value per source, plain or Quantity.
        errors: 1-sigma error per source (same unit handling as ``values``), or
            ``None`` for no bias subtraction.
        coords: Source positions in any frame. With distances, separations are
            3D (2-point stencil only).
        n_point: Stencil size: 2 (``f1 - f2``) or 3 (``f+ - 2 f0 + f-``).
        bins: Number of log-spaced bins, or explicit edges with units.
        min_sep: Lower edge when ``bins`` is a number. Default: smallest
            non-zero separation.
        max_sep: Upper edge when ``bins`` is a number. Default: largest
            separation.
        weights: Optional non-negative per-source weights. A pair is weighted
            by the product of its members' weights.
        jk_regions: Jackknife regions: ``"auto"`` (see
            :func:`structurefunction.regions.recommend_jk_regions`), a number
            (found by k-means on positions), or a region label per source.
            Use several times as many regions as bins.
        error_method: ``jackknife`` (default), ``bootstrap`` or ``none``.
        cross_region_weight: Jackknife weight of cross-region contributions.
        n_boot: Number of bootstrap resamples.
        bin_tolerance: Largest fraction of a bin width by which pairs grouped
            in the tree may be misplaced. ``0`` is exact (2-point only).
        pos_errors: Optional positional errors; pairs are then softly binned.
        stencil: Triplet-selection options for ``n_point=3``.
        min_count: Bins with fewer contributions are flagged unreliable.
        rng: Random generator or seed (regions and bootstrap).
        progress: Show a progress bar.

    Returns:
        The structure function, its covariance and per-bin diagnostics.

    Raises:
        ValueError: On invalid inputs or options.
        NotImplementedError: For combinations not yet supported.
    """
    method = ErrorMethod(error_method)
    cpw = CrossRegionWeight(cross_region_weight)
    stencil = StencilOptions() if stencil is None else stencil
    gen = np.random.default_rng(rng)
    if n_point not in {2, 3}:
        msg = f"n_point must be 2 or 3, got {n_point}"
        raise ValueError(msg)
    if bin_tolerance < 0:
        msg = f"bin_tolerance must be >= 0, got {bin_tolerance}"
        raise ValueError(msg)

    vals = units.split_values(values, errors, weights)
    pos = units.positions(coords)
    n = vals.values.size
    if pos.xyz.shape[0] != n:
        msg = f"{n} values but {pos.xyz.shape[0]} coordinates"
        raise ValueError(msg)
    if n_point == 3 and not pos.angular:
        msg = "3-point stencils are only implemented for sky (2D) positions"
        raise NotImplementedError(msg)

    edges = _edges(bins, min_sep, max_sep, pos)
    n_bins = edges.size - 1
    if isinstance(jk_regions, str):
        if jk_regions != "auto":
            msg = f"jk_regions must be a number, labels or 'auto', got {jk_regions!r}"
            raise ValueError(msg)
        jk_regions = recommend_jk_regions(pos.xyz.shape[0], n_bins)
    region_info = make_regions(pos.xyz, jk_regions, gen)
    logger.info("Using %d jackknife regions", region_info.n_regions)
    labels = region_info.labels

    error_var = vals.errors**2
    position_cov, max_position_var = _position_covariance(pos_errors, pos)
    tree = build_tree(
        pos.xyz,
        vals.values,
        error_var,
        vals.weights,
        labels,
        region_info.n_regions,
        max_position_var,
    )
    sources = _sources_in_tree_order(tree, pos, vals, error_var, labels, position_cov)
    acc = new_accumulator(numba.get_num_threads(), n_bins, region_info.n_regions)
    logger.info(
        "Computing %d-point SF: %d sources, %d bins, %d regions",
        n_point,
        n,
        n_bins,
        region_info.n_regions,
    )
    if n_point == 2:
        # With the default upper edge (the largest separation in the data) no
        # pair can lie beyond it, so the top bin is left open. Otherwise every
        # group of pairs that might reach past it is split down to single pairs.
        open_top = max_sep is None and not isinstance(bins, u.Quantity)
        kernel_edges = edges.copy()
        if open_top and np.ndim(bins) == 0:
            kernel_edges[-1] = np.inf
        binning = Binning(
            kernel_edges, pos.angular, bin_tolerance, pos_errors is not None
        )
        task_a, task_b = node_pair_tasks(frontier(tree, 32 * numba.get_num_threads()))
        # Large tasks first, so the progress bar does not stall at the end.
        order = np.argsort(
            -tree.nodes.moments[task_a, 4] * tree.nodes.moments[task_b, 4]
        )
        task_a, task_b = task_a[order], task_b[order]

        def run_pairs(first: int, stop: int) -> None:
            tree_pairs(
                tree.nodes,
                sources,
                binning,
                task_a[first:stop],
                task_b[first:stop],
                acc,
            )

        _in_chunks(task_a.size, run_pairs, progress, "Pairs")
        summed = acc.sum(axis=0)
        stencil_sq_sum = 2.0
    else:
        stencil_sq_sum = 6.0
        summed = _three_point(
            tree,
            sources,
            edges,
            stencil,
            pos_errors is not None,
            region_info.n_regions,
            acc,
            progress,
        )

    est = estimate(summed, method, cpw, n_boot, gen)
    region_size = float(
        chord_to_separation(np.array([region_info.size]), pos.angular)[0]
    )
    _check_resampling(method, region_info.n_regions, n_bins)
    reported_region_size = units.to_reported_separation(np.array([region_size]), pos)[0]

    wsum = vals.weights.sum()
    mu = np.sum(vals.weights * vals.values) / wsum
    # Intrinsic variance: the debiased SF plateaus at sum(c^2) * (Var - <sigma^2>).
    var = np.sum(vals.weights * (vals.values - mu) ** 2) / wsum
    var -= np.sum(vals.weights * error_var) / wsum
    sf_err = (
        np.sqrt(np.diag(est.cov)) if est.cov is not None else np.full(n_bins, np.nan)
    )
    sep_q = units.to_reported_separation(est.separation, pos)
    edges_q = units.to_reported_separation(edges, pos)
    common_method = method if region_info.n_regions > 1 else ErrorMethod.NONE
    reliable = (est.count >= min_count) & np.isfinite(est.sf)
    sf_unit = None if vals.unit is None else vals.unit**2
    return SFResult(
        sf=est.sf,
        sf_err=sf_err,
        noise_bias=est.noise_bias,
        saturation=float(stencil_sq_sum * var),
        sf_unit=sf_unit,
        cov=est.cov,
        separation=sep_q,
        bin_edges=edges_q,
        count=est.count,
        weight=est.weight,
        n_point=n_point,
        n_regions=region_info.n_regions,
        error_method=common_method,
        min_count=min_count,
        reliable=reliable,
        region_size=reported_region_size,
    )


def structure_function_from_table(
    table: Table,
    value: str,
    error: str | None,
    lon: str = "ra",
    lat: str = "dec",
    frame: str = "icrs",
    n_point: Literal[2, 3] = 2,
    bins: int | u.Quantity = 10,
    jk_regions: int | Literal["auto"] = "auto",
    error_method: ErrorMethodLike = ErrorMethod.JACKKNIFE,
    pos_errors: PositionErrors | None = None,
    stencil: StencilOptions | None = None,
    rng: np.random.Generator | int | None = None,
    progress: bool = True,
) -> SFResult:
    """Convenience wrapper taking column names of an astropy Table.

    Columns without units are used as plain arrays; coordinate columns
    without units are taken to be in degrees.
    """
    from astropy.coordinates import SkyCoord

    def col(name: str) -> u.Quantity | FloatArray:
        c = table[name]
        if getattr(c, "unit", None) is not None:
            return u.Quantity(c)
        return np.asarray(c, dtype=np.float64)

    def angle(name: str) -> u.Quantity:
        c = col(name)
        return c if isinstance(c, u.Quantity) else u.Quantity(c, u.deg)

    coords = SkyCoord(angle(lon), angle(lat), frame=frame)
    return structure_function(
        values=col(value),
        errors=None if error is None else col(error),
        coords=coords,
        n_point=n_point,
        bins=bins,
        jk_regions=jk_regions,
        error_method=error_method,
        pos_errors=pos_errors,
        stencil=stencil,
        rng=rng,
        progress=progress,
    )


def _position_covariance(
    pos_errors: PositionErrors | None, pos: units.Positions
) -> tuple[FloatArray, FloatArray]:
    """Per-source positional covariance and its largest eigenvalue."""
    n_sources = pos.xyz.shape[0]
    if pos_errors is None:
        return np.zeros((n_sources, 3)), np.zeros(n_sources)
    var_east, var_north, cov = units.position_covariance(pos_errors, pos)
    half_sum = 0.5 * (var_east + var_north)
    largest = half_sum + np.sqrt((0.5 * (var_east - var_north)) ** 2 + cov**2)
    return np.column_stack((var_east, var_north, cov)), largest


def _sources_in_tree_order(
    tree: Tree,
    pos: units.Positions,
    vals: units.Values,
    error_var: FloatArray,
    labels: IntArray,
    position_cov: FloatArray,
) -> Sources:
    order = tree.order
    if pos.angular:
        east, north = tangent_basis(pos.lon, pos.lat)
    else:
        east, north = np.zeros_like(pos.xyz), np.zeros_like(pos.xyz)
    return Sources(
        pos.xyz[order],
        vals.values[order],
        error_var[order],
        vals.weights[order],
        labels[order],
        east[order],
        north[order],
        position_cov[order],
    )


def _three_point(
    tree: Tree,
    sources: Sources,
    edges: FloatArray,
    stencil: StencilOptions,
    use_position_errors: bool,
    n_regions: int,
    acc: FloatArray,
    progress: bool,
) -> FloatArray:
    angle_tol = math.radians(stencil.angle_tol_deg)
    n_sources = sources.values.size
    if StencilMode(stencil.mode) == StencilMode.EXACT:
        binning = Binning(edges, True, 0.0, use_position_errors)

        def run_exact(first: int, stop: int) -> None:
            exact_triplets(
                tree.nodes, tree.roots, sources, binning, angle_tol, stencil.sep_tol,
                first, stop, acc,
            )  # fmt: skip

        _in_chunks(n_sources, run_exact, progress, "Triplets")
        return acc.sum(axis=0)
    if use_position_errors:
        msg = (
            "positional errors with 3-point stencils need StencilOptions(mode='exact')"
        )
        raise ValueError(msg)
    # Cells are half a tolerance wide; the kernel also pairs neighbouring cells.
    ring_edges, ring_to_bin = _ring_edges(edges, math.sqrt(1.0 + stencil.sep_tol) - 1.0)
    grid = PolarGrid(
        ring_edges,
        ring_to_bin,
        np.sqrt(ring_edges[:-1] * ring_edges[1:]),
        2 * math.ceil(2.0 * math.pi / angle_tol),
    )

    def run_binned(first: int, stop: int) -> None:
        binned_triplets(
            tree.nodes, tree.roots, sources, edges, grid, n_regions, first, stop, acc
        )

    _in_chunks(n_sources, run_binned, progress, "Triplets")
    return acc.sum(axis=0)


def _in_chunks(
    n_items: int,
    run: Callable[[int, int], None],
    progress: bool,
    label: str,
    n_chunks: int = 40,
) -> None:
    """Call ``run(first, stop)`` over ``n_items`` in chunks, with a progress bar.

    Each chunk is one parallel kernel call, so chunks must hold many more
    items than there are threads or threads sit idle at the end of each.
    """
    bounds = np.unique(np.linspace(0, n_items, n_chunks + 1).astype(int))
    with tqdm(
        total=n_items, desc=label, unit="task" if label == "Pairs" else "source",
        disable=not progress, leave=False,
    ) as bar:  # fmt: skip
        for first, stop in itertools.pairwise(bounds):
            run(int(first), int(stop))
            bar.update(int(stop - first))


def _edges(
    bins: int | u.Quantity,
    min_sep: u.Quantity | None,
    max_sep: u.Quantity | None,
    pos: units.Positions,
) -> FloatArray:
    if isinstance(bins, u.Quantity) or np.ndim(bins) > 0:
        edges = units.separation_array(bins, pos)
        if edges.ndim != 1 or edges.size < 2 or np.any(np.diff(edges) <= 0):
            msg = "bin edges must be a strictly increasing 1D array"
            raise ValueError(msg)
        return edges
    n_bins = int(bins)
    if n_bins < 1:
        msg = f"need at least one bin, got {n_bins}"
        raise ValueError(msg)
    if min_sep is None or max_sep is None:
        smin, smax = separation_range(pos.xyz, pos.angular)
    smin = smin if min_sep is None else units.separation_value(min_sep, pos)
    smax = smax if max_sep is None else units.separation_value(max_sep, pos)
    return log_edges(smin, smax, n_bins)


def _ring_edges(edges: FloatArray, sep_tol: float) -> tuple[FloatArray, IntArray]:
    """Geometric radial sub-bins of ratio ``<= 1 + sep_tol`` nested in ``edges``."""
    if edges[0] <= 0.0:
        msg = "binned 3-point stencils need a positive minimum separation"
        raise ValueError(msg)
    parts = []
    owner = []
    for k in range(edges.size - 1):
        n_sub = max(
            1, math.ceil(math.log(edges[k + 1] / edges[k]) / math.log1p(sep_tol))
        )
        sub = np.geomspace(edges[k], edges[k + 1], n_sub + 1)
        parts.append(sub[:-1])
        owner.extend([k] * n_sub)
    parts.append(edges[-1:])
    return np.concatenate(parts), np.asarray(owner, dtype=np.intp)


def _check_resampling(
    method: ErrorMethod,
    n_regions: int,
    n_bins: int,
) -> None:
    if method == ErrorMethod.NONE:
        return
    if n_regions < 3 * n_bins:
        warnings.warn(
            f"{n_regions} regions for {n_bins} bins: the covariance will be noisy "
            "and possibly singular (aim for n_regions >= 3 * n_bins)",
            StructureFunctionWarning,
            stacklevel=3,
        )
