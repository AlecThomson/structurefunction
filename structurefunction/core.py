"""Public entry points for computing structure functions."""

from __future__ import annotations

import math
import warnings
from typing import TYPE_CHECKING, Literal, overload

import astropy.units as u
import numba
import numpy as np

from structurefunction import _units
from structurefunction._kernels.brute import empty_pos_arrays
from structurefunction._kernels.common import new_accumulator
from structurefunction._kernels.tree import build_tree, dual_tree_pairs
from structurefunction._kernels.triplets import binned_triplets, exact_triplets
from structurefunction.estimators import estimate
from structurefunction.geometry import (
    chord_to_separation,
    log_edges,
    separation_range,
    tangent_basis,
)
from structurefunction.logging import StructureFunctionWarning, logger
from structurefunction.options import StencilOptions
from structurefunction.patches import make_patches
from structurefunction.results import SFResult
from structurefunction.types import (
    CrossPatchWeight,
    CrossPatchWeightLike,
    ErrorMethod,
    ErrorMethodLike,
    FloatArray,
    IntArray,
    StencilMode,
)

if TYPE_CHECKING:
    from astropy.coordinates import SkyCoord
    from astropy.table import Table
    from numpy.typing import ArrayLike

    from structurefunction.options import PositionErrors


@overload
def structure_function(
    values: u.Quantity,
    errors: u.Quantity | None,
    coords: SkyCoord,
    n_point: Literal[2, 3] = 2,
    bins: int | u.Quantity = 10,
    min_sep: u.Quantity | None = None,
    max_sep: u.Quantity | None = None,
    weights: ArrayLike | None = None,
    patches: int | ArrayLike = 64,
    error_method: ErrorMethodLike = ErrorMethod.JACKKNIFE,
    cross_patch_weight: CrossPatchWeightLike = CrossPatchWeight.SIMPLE,
    n_boot: int = 1000,
    bin_tolerance: float = 0.05,
    pos_errors: PositionErrors | None = None,
    stencil: StencilOptions | None = None,
    min_count: int = 10,
    rng: np.random.Generator | int | None = None,
) -> SFResult[u.Quantity]: ...


@overload
def structure_function(
    values: ArrayLike,
    errors: ArrayLike | None,
    coords: SkyCoord,
    n_point: Literal[2, 3] = 2,
    bins: int | u.Quantity = 10,
    min_sep: u.Quantity | None = None,
    max_sep: u.Quantity | None = None,
    weights: ArrayLike | None = None,
    patches: int | ArrayLike = 64,
    error_method: ErrorMethodLike = ErrorMethod.JACKKNIFE,
    cross_patch_weight: CrossPatchWeightLike = CrossPatchWeight.SIMPLE,
    n_boot: int = 1000,
    bin_tolerance: float = 0.05,
    pos_errors: PositionErrors | None = None,
    stencil: StencilOptions | None = None,
    min_count: int = 10,
    rng: np.random.Generator | int | None = None,
) -> SFResult[FloatArray]: ...


@overload
def structure_function(
    values: ArrayLike | u.Quantity,
    errors: ArrayLike | u.Quantity | None,
    coords: SkyCoord,
    n_point: Literal[2, 3] = 2,
    bins: int | u.Quantity = 10,
    min_sep: u.Quantity | None = None,
    max_sep: u.Quantity | None = None,
    weights: ArrayLike | None = None,
    patches: int | ArrayLike = 64,
    error_method: ErrorMethodLike = ErrorMethod.JACKKNIFE,
    cross_patch_weight: CrossPatchWeightLike = CrossPatchWeight.SIMPLE,
    n_boot: int = 1000,
    bin_tolerance: float = 0.05,
    pos_errors: PositionErrors | None = None,
    stencil: StencilOptions | None = None,
    min_count: int = 10,
    rng: np.random.Generator | int | None = None,
) -> SFResult[u.Quantity] | SFResult[FloatArray]: ...


def structure_function(
    values: ArrayLike | u.Quantity,
    errors: ArrayLike | u.Quantity | None,
    coords: SkyCoord,
    n_point: Literal[2, 3] = 2,
    bins: int | u.Quantity = 10,
    min_sep: u.Quantity | None = None,
    max_sep: u.Quantity | None = None,
    weights: ArrayLike | None = None,
    patches: int | ArrayLike = 64,
    error_method: ErrorMethodLike = ErrorMethod.JACKKNIFE,
    cross_patch_weight: CrossPatchWeightLike = CrossPatchWeight.SIMPLE,
    n_boot: int = 1000,
    bin_tolerance: float = 0.05,
    pos_errors: PositionErrors | None = None,
    stencil: StencilOptions | None = None,
    min_count: int = 10,
    rng: np.random.Generator | int | None = None,
) -> SFResult[u.Quantity] | SFResult[FloatArray]:
    """Second-order structure function with a 2- or 3-point stencil.

    The noise bias ``<sum_i c_i^2 sigma_i^2>`` is subtracted exactly. The
    covariance between bins comes from resampling spatial patches, so it
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
        patches: Number of k-means patches, or a patch label per source.
        error_method: ``jackknife`` (default), ``bootstrap`` or ``none``.
        cross_patch_weight: Jackknife weight of cross-patch contributions.
        n_boot: Number of bootstrap resamples.
        bin_tolerance: Largest fraction of a bin width by which pairs grouped
            in the tree may be misplaced. ``0`` is exact (2-point only).
        pos_errors: Optional positional errors; pairs are then softly binned.
        stencil: Triplet-selection options for ``n_point=3``.
        min_count: Bins with fewer contributions are flagged unreliable.
        rng: Random generator or seed (patches and bootstrap).

    Returns:
        The structure function, its covariance and per-bin diagnostics.

    Raises:
        ValueError: On invalid inputs or options.
        NotImplementedError: For combinations not yet supported.
    """
    method = ErrorMethod(error_method)
    cpw = CrossPatchWeight(cross_patch_weight)
    stencil = StencilOptions() if stencil is None else stencil
    gen = np.random.default_rng(rng)
    if n_point not in {2, 3}:
        msg = f"n_point must be 2 or 3, got {n_point}"
        raise ValueError(msg)
    if bin_tolerance < 0:
        msg = f"bin_tolerance must be >= 0, got {bin_tolerance}"
        raise ValueError(msg)

    vals = _units.split_values(values, errors, weights)
    pos = _units.positions(coords)
    n = vals.values.size
    if pos.xyz.shape[0] != n:
        msg = f"{n} values but {pos.xyz.shape[0]} coordinates"
        raise ValueError(msg)
    if n_point == 3 and not pos.angular:
        msg = "3-point stencils are only implemented for sky (2D) positions"
        raise NotImplementedError(msg)

    edges = _edges(bins, min_sep, max_sep, pos)
    n_bins = edges.size - 1
    patch_info = make_patches(pos.xyz, patches, gen)
    labels = patch_info.labels

    east, north = tangent_basis(pos.lon, pos.lat)
    use_pos = pos_errors is not None
    if use_pos and pos_errors is not None:
        ve, vn, cen = _units.position_covariance(pos_errors, pos)
        pcov = np.column_stack((ve, vn, cen))
        vmaj = 0.5 * (ve + vn) + np.sqrt((0.5 * (ve - vn)) ** 2 + cen**2)
    else:
        _, _, pcov = empty_pos_arrays(n)
        vmaj = np.zeros(n)
    if not pos.angular:
        east, north, _ = empty_pos_arrays(n)

    sig2 = vals.errors**2
    tree = build_tree(
        pos.xyz, vals.values, sig2, vals.weights, labels, patch_info.n_patch, vmaj
    )
    o = tree.order
    acc = new_accumulator(numba.get_num_threads(), n_bins, patch_info.n_patch)
    logger.info(
        "Computing %d-point SF: %d sources, %d bins, %d patches",
        n_point,
        n,
        n_bins,
        patch_info.n_patch,
    )
    if n_point == 2:
        summed = dual_tree_pairs(
            tree.start,
            tree.end,
            tree.left,
            tree.right,
            tree.centre,
            tree.radius,
            tree.moments,
            tree.patch,
            tree.roots,
            pos.xyz[o],
            vals.values[o],
            sig2[o],
            vals.weights[o],
            labels[o],
            edges,
            acc,
            pos.angular,
            bin_tolerance,
            east[o],
            north[o],
            pcov[o],
            use_pos,
        )
        coeff_sq = 2.0
    else:
        coeff_sq = 6.0
        mode = StencilMode(stencil.mode)
        if mode == StencilMode.EXACT:
            summed = exact_triplets(
                tree.start,
                tree.end,
                tree.left,
                tree.centre,
                tree.radius,
                tree.roots,
                pos.xyz[o],
                vals.values[o],
                sig2[o],
                vals.weights[o],
                labels[o],
                edges,
                acc,
                math.radians(stencil.angle_tol_deg),
                stencil.sep_tol,
                east[o],
                north[o],
                pcov[o],
                use_pos,
            )
        else:
            if use_pos:
                msg = (
                    "positional errors with 3-point stencils need "
                    "StencilOptions(mode='exact')"
                )
                raise ValueError(msg)
            sub_edges, sub_to_bin = _sub_edges(edges, stencil.sep_tol)
            n_sector = 2 * math.ceil(math.pi / math.radians(stencil.angle_tol_deg))
            summed = binned_triplets(
                tree.start,
                tree.end,
                tree.left,
                tree.centre,
                tree.radius,
                tree.moments,
                tree.patch,
                tree.roots,
                pos.xyz[o],
                vals.values[o],
                sig2[o],
                vals.weights[o],
                labels[o],
                sub_edges,
                sub_to_bin,
                n_sector,
                patch_info.n_patch,
                acc,
                east[o],
                north[o],
            )

    est = estimate(summed, method, cpw, n_boot, gen)
    _check_resampling(method, patch_info.n_patch, n_bins, edges, patch_info.size, pos)

    wsum = vals.weights.sum()
    mu = np.sum(vals.weights * vals.values) / wsum
    # Intrinsic variance: the debiased SF plateaus at sum(c^2) * (Var - <sigma^2>).
    var = np.sum(vals.weights * (vals.values - mu) ** 2) / wsum
    var -= np.sum(vals.weights * sig2) / wsum
    sf_err = (
        np.sqrt(np.diag(est.cov)) if est.cov is not None else np.full(n_bins, np.nan)
    )
    sep_q = _units.to_reported_separation(est.separation, pos)
    edges_q = _units.to_reported_separation(edges, pos)
    common_method = method if patch_info.n_patch > 1 else ErrorMethod.NONE
    unit = vals.unit
    if unit is None:
        return SFResult[FloatArray](
            sf=est.sf,
            sf_err=sf_err,
            noise_bias=est.noise_bias,
            saturation=np.asarray(coeff_sq * var),
            cov=est.cov,
            separation=sep_q,
            bin_edges=edges_q,
            count=est.count,
            weight=est.weight,
            n_point=n_point,
            n_patch=patch_info.n_patch,
            error_method=common_method,
            min_count=min_count,
        )
    unit2 = unit**2
    return SFResult[u.Quantity](
        sf=u.Quantity(est.sf, unit2),
        sf_err=u.Quantity(sf_err, unit2),
        noise_bias=u.Quantity(est.noise_bias, unit2),
        saturation=u.Quantity(coeff_sq * var, unit2),
        cov=est.cov,
        separation=sep_q,
        bin_edges=edges_q,
        count=est.count,
        weight=est.weight,
        n_point=n_point,
        n_patch=patch_info.n_patch,
        error_method=common_method,
        min_count=min_count,
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
    patches: int = 64,
    error_method: ErrorMethodLike = ErrorMethod.JACKKNIFE,
    pos_errors: PositionErrors | None = None,
    stencil: StencilOptions | None = None,
    rng: np.random.Generator | int | None = None,
) -> SFResult[u.Quantity] | SFResult[FloatArray]:
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
        patches=patches,
        error_method=error_method,
        pos_errors=pos_errors,
        stencil=stencil,
        rng=rng,
    )


def _edges(
    bins: int | u.Quantity,
    min_sep: u.Quantity | None,
    max_sep: u.Quantity | None,
    pos: _units.Positions,
) -> FloatArray:
    if isinstance(bins, u.Quantity) or np.ndim(bins) > 0:
        edges = _units.separation_array(bins, pos)
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
    smin = smin if min_sep is None else _units.separation_value(min_sep, pos)
    smax = smax if max_sep is None else _units.separation_value(max_sep, pos)
    return log_edges(smin, smax, n_bins)


def _sub_edges(edges: FloatArray, sep_tol: float) -> tuple[FloatArray, IntArray]:
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
    n_patch: int,
    n_bins: int,
    edges: FloatArray,
    patch_size: float,
    pos: _units.Positions,
) -> None:
    if method == ErrorMethod.NONE:
        return
    if n_patch < 3 * n_bins:
        warnings.warn(
            f"{n_patch} patches for {n_bins} bins: the covariance will be noisy "
            "and possibly singular (aim for n_patch >= 3 * n_bins)",
            StructureFunctionWarning,
            stacklevel=3,
        )
    size = float(chord_to_separation(np.array([patch_size]), pos.angular)[0])
    if edges[-1] > size:
        warnings.warn(
            "largest separations exceed the typical patch size; patch resampling "
            "underestimates errors there",
            StructureFunctionWarning,
            stacklevel=3,
        )
