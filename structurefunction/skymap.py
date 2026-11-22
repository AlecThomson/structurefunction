"""Structure-function sky maps: apodised structure functions on a HEALPix grid.

For each HEALPix cell, sources around the cell centre are weighted by a taper
(cosine or Gaussian) in angular distance. The structure function is then
computed with those weights, so each pair is weighted by the product of its
members' taper values. Optionally a model is fitted per cell, giving maps of
amplitude, break scale and slopes.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, assert_never

import astropy.units as u
import numpy as np
from astropy_healpix import HEALPix
from scipy.spatial import KDTree
from tqdm.auto import tqdm

from structurefunction import units
from structurefunction.core import structure_function
from structurefunction.fitting import fit_sf
from structurefunction.geometry import separation_range
from structurefunction.logging import StructureFunctionWarning, logger
from structurefunction.options import AxisPositionErrors, CircularPositionError
from structurefunction.types import (
    ErrorMethod,
    FitModel,
    FitModelLike,
    Sampler,
    Taper,
    TaperLike,
)

if TYPE_CHECKING:
    from astropy.coordinates import SkyCoord
    from numpy.typing import ArrayLike

    from structurefunction.options import PositionErrors, StencilOptions
    from structurefunction.types import FloatArray, IntArray


def taper_weights(
    distance: FloatArray, radius: float, taper: TaperLike = Taper.COSINE
) -> FloatArray:
    """Taper value for angular ``distance`` (same unit as ``radius``).

    Args:
        distance: Angular distances from the cell centre.
        radius: Cosine: radius where the weight reaches zero. Gaussian: FWHM.
        taper: Window shape.

    Returns:
        Weights in ``[0, 1]``, exactly 1 at the centre.
    """
    taper = Taper(taper)
    d = np.asarray(distance, dtype=np.float64)
    if taper == Taper.COSINE:
        return np.where(d < radius, 0.5 * (1.0 + np.cos(np.pi * d / radius)), 0.0)
    sigma = radius / (2.0 * math.sqrt(2.0 * math.log(2.0)))
    return np.where(d < 3.0 * sigma, np.exp(-0.5 * (d / sigma) ** 2), 0.0)


def taper_support(radius: float, taper: TaperLike = Taper.COSINE) -> float:
    """Distance beyond which the taper is zero."""
    if Taper(taper) == Taper.COSINE:
        return radius
    return 3.0 * radius / (2.0 * math.sqrt(2.0 * math.log(2.0)))


def pixel_size(nside: int) -> u.Quantity:
    """Typical HEALPix pixel size, the square root of the pixel area."""
    return u.Quantity(math.sqrt(4.0 * math.pi / (12 * nside**2)), u.rad).to(u.deg)


def recommend_nside(
    radius: u.Quantity, taper: TaperLike = Taper.COSINE, oversample: float = 3.0
) -> int:
    """Smallest HEALPix nside that samples the taper finely enough.

    The taper's full width at half maximum is ``radius`` for both shapes (the
    cosine taper falls to one half at ``radius / 2``; for the Gaussian
    ``radius`` is the FWHM). Pixels are at most ``FWHM / oversample``; the
    default of three pixels per FWHM is the usual choice for smoothed sky
    maps (two is the bare Nyquist limit and looks blocky). Finer grids only add
    strongly correlated cells.

    Args:
        radius: Taper scale, as passed to :func:`structure_function_skymap`.
        taper: Taper shape.
        oversample: Pixels per FWHM.

    Returns:
        A power of two.
    """
    Taper(taper)  # validate
    target = radius.to(u.deg) / oversample
    nside = 1
    while pixel_size(nside) > target:
        nside *= 2
    return nside


@dataclass(frozen=True, slots=True)
class SkyMapResult:
    """Per-cell structure functions and fitted parameters on a HEALPix grid.

    Arrays have one row per computed cell (``pixels``); :func:`healpix_map`
    expands a quantity to a full-sky HEALPix array, NaN for skipped cells.
    """

    nside: int
    """HEALPix resolution used."""
    order: Literal["ring", "nested"]
    """HEALPix pixel ordering."""
    pixels: IntArray
    """HEALPix indices of the computed cells."""
    n_eff: FloatArray
    """Effective number of sources per cell, ``(sum w)^2 / sum w^2``."""
    sf: FloatArray
    """Structure function per cell and bin, shape ``(n_cell, n_bin)``."""
    sf_err: FloatArray
    """Structure function errors per cell and bin."""
    count: FloatArray
    """Pairs or triplets per cell and bin."""
    bin_edges: u.Quantity
    """Separation bin edges, shared by all cells."""
    separation: FloatArray
    """Mean separation per cell and bin (unit of ``bin_edges``)."""
    sf_unit: u.UnitBase | None
    """Unit of ``sf`` (``value_unit**2``), or ``None`` for plain arrays."""
    param_names: tuple[str, ...]
    """Fitted parameter names (empty without a fit)."""
    params: FloatArray
    """MAP parameters per cell, shape ``(n_cell, n_param)``."""
    params_lower: FloatArray
    """16th percentile of each fitted parameter per cell."""
    params_upper: FloatArray
    """84th percentile of each fitted parameter per cell."""
    chi2: FloatArray
    """Chi-squared of the fit in each cell (NaN without a fit)."""
    n_point: int
    """Number of points in the stencil."""
    taper: Taper
    """Taper shape."""
    radius: u.Quantity
    """Taper scale."""


def healpix_map(
    result: SkyMapResult, name: str, bin_index: int | None = None
) -> FloatArray:
    """Full-sky HEALPix array of one sky-map quantity, NaN where not computed.

    Args:
        result: The sky map.
        name: ``sf``, ``sf_err``, ``count``, ``separation``, ``n_eff``,
            ``chi2``, a parameter name, or a parameter name suffixed with
            ``_lower`` / ``_upper``.
        bin_index: Bin for per-bin quantities.

    Raises:
        KeyError: For an unknown quantity.
        ValueError: If ``bin_index`` is missing for a per-bin quantity.
    """
    out = np.full(12 * result.nside**2, np.nan)
    per_bin = {
        "sf": result.sf,
        "sf_err": result.sf_err,
        "count": result.count,
        "separation": result.separation,
    }
    if name in per_bin:
        if bin_index is None:
            msg = f"{name!r} is per bin: pass bin_index"
            raise ValueError(msg)
        out[result.pixels] = per_bin[name][:, bin_index]
        return out
    per_cell = {"n_eff": result.n_eff, "chi2": result.chi2}
    for index, parameter in enumerate(result.param_names):
        per_cell[parameter] = result.params[:, index]
        per_cell[f"{parameter}_lower"] = result.params_lower[:, index]
        per_cell[f"{parameter}_upper"] = result.params_upper[:, index]
    if name not in per_cell:
        msg = f"unknown quantity {name!r}; choose from {sorted(per_bin | per_cell)}"
        raise KeyError(msg)
    out[result.pixels] = per_cell[name]
    return out


def structure_function_skymap(
    values: ArrayLike | u.Quantity,
    errors: ArrayLike | u.Quantity | None,
    coords: SkyCoord,
    radius: u.Quantity,
    nside: int | Literal["auto"] = "auto",
    bins: int | u.Quantity = 8,
    min_sep: u.Quantity | None = None,
    max_sep: u.Quantity | None = None,
    taper: TaperLike = Taper.COSINE,
    order: Literal["ring", "nested"] = "ring",
    weights: ArrayLike | None = None,
    n_point: Literal[2, 3] = 2,
    jk_regions: int = 16,
    min_sources: float = 30.0,
    fit_model: FitModelLike | None = FitModel.BROKEN_POWER_LAW,
    pixels: ArrayLike | None = None,
    bin_tolerance: float = 0.05,
    pos_errors: PositionErrors | None = None,
    stencil: StencilOptions | None = None,
    rng: np.random.Generator | int | None = None,
    progress: bool = True,
) -> SkyMapResult:
    """Apodised structure functions (and fits) for every HEALPix cell.

    The grid uses the longitude/latitude of ``coords``' own frame, so pass
    Galactic coordinates for a Galactic map.

    Args:
        values: Measured values, plain or Quantity.
        errors: 1-sigma errors (same unit handling as ``values``) or ``None``.
        coords: Source positions (2D sky positions).
        radius: Taper scale: zero point (cosine) or FWHM (Gaussian).
        nside: HEALPix resolution, or ``"auto"`` for :func:`recommend_nside`.
        bins: Number of log bins, or explicit edges shared by all cells.
        min_sep: Lower edge for numeric ``bins`` (default: smallest separation
            in the catalogue).
        max_sep: Upper edge for numeric ``bins`` (default: ``radius``).
        taper: ``cosine`` or ``gaussian``.
        order: HEALPix pixel ordering.
        weights: Optional per-source weights, multiplied by the taper.
        n_point: Stencil size.
        jk_regions: Jackknife regions per cell (k-means on the cell's sources).
        min_sources: Skip cells with fewer effective sources.
        fit_model: Model to fit per cell (MAP), or ``None``.
        pixels: Only compute these cells (default: all with enough sources).
        bin_tolerance: See :func:`structurefunction.structure_function`.
        pos_errors: Optional positional errors.
        stencil: Three-point stencil options.
        rng: Random generator or seed.
        progress: Show a progress bar over cells.

    Returns:
        Per-cell structure functions and parameters.

    Raises:
        ValueError: For coordinates with distances.
    """
    taper = Taper(taper)
    gen = np.random.default_rng(rng)
    vals = units.split_values(values, errors, weights)
    pos = units.positions(coords)
    if not pos.angular:
        msg = "sky maps need 2D sky positions"
        raise ValueError(msg)
    radius_rad = float(radius.to_value(u.rad))
    support = taper_support(radius_rad, taper)

    if isinstance(bins, u.Quantity):
        edges_q = bins
    else:
        if min_sep is None:
            lo, _ = separation_range(pos.xyz, angular=True)
            min_sep = u.Quantity(lo, u.rad).to(u.deg)
        hi = radius if max_sep is None else max_sep
        edges_q = u.Quantity(
            np.geomspace(min_sep.to_value(u.deg), hi.to_value(u.deg), int(bins) + 1),
            u.deg,
        )

    if nside == "auto":
        nside = recommend_nside(radius, taper)
    logger.info("Sky map: nside %d (%.2f deg pixels)", nside, pixel_size(nside).value)
    hp = HEALPix(nside=nside, order=order)
    targets = (
        np.arange(hp.npix) if pixels is None else np.asarray(pixels, dtype=np.intp)
    )
    lon_c, lat_c = hp.healpix_to_lonlat(targets)
    centres = np.column_stack(
        (
            np.cos(lat_c.to_value(u.rad)) * np.cos(lon_c.to_value(u.rad)),
            np.cos(lat_c.to_value(u.rad)) * np.sin(lon_c.to_value(u.rad)),
            np.sin(lat_c.to_value(u.rad)),
        )
    )
    tree = KDTree(pos.xyz)
    chord = 2.0 * math.sin(min(support, math.pi) / 2.0)
    members = tree.query_ball_point(centres, chord)

    n_bin = edges_q.size - 1
    model = None if fit_model is None else FitModel(fit_model)
    names: tuple[str, ...] = ()
    if model == FitModel.POWER_LAW:
        names = ("log10_amplitude", "slope")
    elif model is not None:
        names = ("log10_amplitude", "log10_break", "slope_1", "slope_2")

    rows: list[int] = []
    out: dict[str, list[FloatArray]] = {
        k: []
        for k in ("n_eff", "sf", "sf_err", "count", "sep", "par", "lo", "hi", "chi2")
    }
    sf_unit: u.UnitBase | None = None
    n_eff_occupied: list[float] = []
    cells = tqdm(
        enumerate(zip(targets, members, strict=True)),
        total=targets.size,
        desc="Cells",
        unit="cell",
        disable=not progress,
        leave=False,
    )
    for k, (pix, idx_list) in cells:
        idx = np.asarray(idx_list, dtype=np.intp)
        if idx.size < 3:
            continue
        chord = np.linalg.norm(pos.xyz[idx] - centres[k], axis=1)
        distance = 2.0 * np.arcsin(np.clip(chord / 2.0, 0, 1))
        w = taper_weights(distance, radius_rad, taper) * vals.weights[idx]
        keep = w > 0
        idx, w = idx[keep], w[keep]
        n_eff = float(w.sum() ** 2 / np.sum(w**2)) if w.size else 0.0
        n_eff_occupied.append(n_eff)
        if n_eff < min_sources:
            continue
        v = units.attach_unit(vals.values[idx], vals.unit)
        e = units.attach_unit(vals.errors[idx], vals.unit)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", StructureFunctionWarning)
            res = structure_function(
                values=v,
                errors=e if errors is not None else None,
                coords=coords[idx],
                n_point=n_point,
                bins=edges_q,
                weights=w,
                jk_regions=min(jk_regions, max(int(n_eff) // 10, 2)),
                bin_tolerance=bin_tolerance,
                pos_errors=_subset_pos_errors(pos_errors, idx),
                stencil=stencil,
                rng=gen,
                progress=False,
            )
        sf_unit = res.sf_unit
        rows.append(int(pix))
        out["n_eff"].append(np.array([n_eff]))
        out["sf"].append(res.sf)
        out["sf_err"].append(res.sf_err)
        out["count"].append(res.count)
        out["sep"].append(np.asarray(res.separation.to_value(edges_q.unit)))
        par = np.full(len(names), np.nan)
        plo = par.copy()
        phi = par.copy()
        chi2 = np.array([np.nan])
        if model is not None and res.error_method != ErrorMethod.NONE:
            try:
                fit = fit_sf(res, model=model, sampler=Sampler.MAP, rng=gen)
            except (ValueError, np.linalg.LinAlgError) as err:
                logger.debug("Cell %d: fit skipped (%s)", pix, err)
            else:
                par, plo, phi = fit.map, fit.lower, fit.upper
                chi2 = np.array([fit.chi2])
        out["par"].append(par)
        out["lo"].append(plo)
        out["hi"].append(phi)
        out["chi2"].append(chi2)

    logger.info("Sky map: %d of %d cells computed", len(rows), targets.size)
    if n_eff_occupied and np.median(n_eff_occupied) < min_sources:
        warnings.warn(
            f"the typical cell has {np.median(n_eff_occupied):.0f} effective sources, "
            f"fewer than min_sources={min_sources:g}, so most cells were skipped. "
            "Use a larger radius.",
            StructureFunctionWarning,
            stacklevel=2,
        )

    def stack(key: str, width: int) -> FloatArray:
        if not out[key]:
            return np.empty((0, width))
        return np.vstack(out[key]).reshape(len(rows), width)

    return SkyMapResult(
        nside=nside,
        order=order,
        pixels=np.asarray(rows, dtype=np.intp),
        n_eff=stack("n_eff", 1)[:, 0],
        sf=stack("sf", n_bin),
        sf_err=stack("sf_err", n_bin),
        count=stack("count", n_bin),
        bin_edges=edges_q,
        separation=stack("sep", n_bin),
        sf_unit=sf_unit,
        param_names=names,
        params=stack("par", len(names)),
        params_lower=stack("lo", len(names)),
        params_upper=stack("hi", len(names)),
        chi2=stack("chi2", 1)[:, 0],
        n_point=n_point,
        taper=taper,
        radius=radius,
    )


def _subset_pos_errors(
    pos_errors: PositionErrors | None, idx: IntArray
) -> PositionErrors | None:
    """Restrict per-source positional errors to a subset of sources."""
    if pos_errors is None:
        return None

    def sub(q: u.Quantity) -> u.Quantity:
        return q if q.ndim == 0 else q[idx]

    if isinstance(pos_errors, CircularPositionError):
        return pos_errors.model_copy(update={"sigma": sub(pos_errors.sigma)})
    if isinstance(pos_errors, AxisPositionErrors):  # pragma: no branch
        return pos_errors.model_copy(
            update={
                "sigma_ra": sub(pos_errors.sigma_ra),
                "sigma_dec": sub(pos_errors.sigma_dec),
            }
        )
    assert_never(pos_errors)
