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
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Literal, assert_never

import astropy.units as u
import numpy as np
from astropy.table import QTable, Table
from astropy_healpix import HEALPix
from scipy.spatial import KDTree

from structurefunction import _units
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
    from pathlib import Path

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


@dataclass(frozen=True, slots=True)
class SkyMapResult:
    """Per-cell structure functions and fitted parameters on a HEALPix grid.

    Arrays have one row per computed cell (``pixels``); use :meth:`map` to
    expand to a full-sky HEALPix array with NaN for cells that were skipped.
    """

    nside: int
    order: Literal["ring", "nested"]
    pixels: IntArray
    """HEALPix indices of the computed cells."""
    n_eff: FloatArray
    """Effective number of sources per cell, ``(sum w)^2 / sum w^2``."""
    sf: FloatArray
    """Structure function per cell and bin, shape ``(n_cell, n_bin)``."""
    sf_err: FloatArray
    count: FloatArray
    bin_edges: u.Quantity
    separation: FloatArray
    """Mean separation per cell and bin (unit of ``bin_edges``)."""
    sf_unit: u.UnitBase | None
    """Unit of ``sf`` (``value_unit**2``), or ``None`` for plain arrays."""
    param_names: tuple[str, ...]
    """Fitted parameter names (empty without a fit)."""
    params: FloatArray
    """MAP parameters per cell, shape ``(n_cell, n_param)``."""
    params_lower: FloatArray
    params_upper: FloatArray
    chi2: FloatArray
    n_point: int
    taper: Taper
    radius: u.Quantity

    @property
    def npix(self) -> int:
        """Number of pixels on the full sky."""
        return 12 * self.nside**2

    def map(self, name: str, bin_index: int | None = None) -> FloatArray:
        """Full-sky HEALPix array of a quantity, NaN where not computed.

        Args:
            name: ``sf``, ``sf_err``, ``count``, ``separation``, ``n_eff``,
                ``chi2``, a parameter name, or a parameter name suffixed with
                ``_lower`` / ``_upper``.
            bin_index: Bin for per-bin quantities.

        Raises:
            KeyError: For an unknown quantity.
            ValueError: If ``bin_index`` is missing for a per-bin quantity.
        """
        out = np.full(self.npix, np.nan)
        per_bin = {
            "sf": self.sf,
            "sf_err": self.sf_err,
            "count": self.count,
            "separation": self.separation,
        }
        if name in per_bin:
            if bin_index is None:
                msg = f"{name!r} is per bin: pass bin_index"
                raise ValueError(msg)
            out[self.pixels] = per_bin[name][:, bin_index]
            return out
        scalar = {"n_eff": self.n_eff, "chi2": self.chi2}
        for i, p in enumerate(self.param_names):
            scalar[p] = self.params[:, i]
            scalar[f"{p}_lower"] = self.params_lower[:, i]
            scalar[f"{p}_upper"] = self.params_upper[:, i]
        if name not in scalar:
            msg = f"unknown quantity {name!r}; choose from {sorted(per_bin | scalar)}"
            raise KeyError(msg)
        out[self.pixels] = scalar[name]
        return out

    def to_table(self) -> QTable:
        """One row per computed cell; per-bin quantities as vector columns."""
        t = QTable()
        t["pixel"] = self.pixels
        t["n_eff"] = self.n_eff
        t["sf"] = self.sf if self.sf_unit is None else u.Quantity(self.sf, self.sf_unit)
        t["sf_err"] = (
            self.sf_err
            if self.sf_unit is None
            else u.Quantity(self.sf_err, self.sf_unit)
        )
        t["count"] = self.count
        t["separation"] = u.Quantity(self.separation, self.bin_edges.unit)
        t["chi2"] = self.chi2
        for i, p in enumerate(self.param_names):
            t[p] = self.params[:, i]
            t[f"{p}_lower"] = self.params_lower[:, i]
            t[f"{p}_upper"] = self.params_upper[:, i]
        # Bin edges as (identical) vector columns and short header keys keep
        # the table valid in both ECSV and FITS.
        unit = self.bin_edges.unit
        edges = self.bin_edges.to_value(unit)
        t["bin_lo"] = u.Quantity(np.broadcast_to(edges[:-1], self.sf.shape), unit)
        t["bin_hi"] = u.Quantity(np.broadcast_to(edges[1:], self.sf.shape), unit)
        t.meta.update(
            NSIDE=self.nside,
            ORDER=self.order,
            NPOINT=self.n_point,
            TAPER=str(self.taper),
            RADIUS=float(self.radius.to_value(u.deg)),
            SFUNIT="" if self.sf_unit is None else str(self.sf_unit),
        )
        return t

    def write(self, path: str | Path, overwrite: bool = False) -> None:
        """Write to any astropy table format (ECSV or FITS)."""
        self.to_table().write(path, overwrite=overwrite)

    @classmethod
    def read(cls, path: str | Path) -> SkyMapResult:
        """Read a result written by :meth:`write`."""
        t = QTable(Table.read(path))
        m = t.meta
        names = tuple(
            c.removesuffix("_lower") for c in t.colnames if c.endswith("_lower")
        )

        def plain(col: str) -> FloatArray:
            c = t[col]
            return np.asarray(getattr(c, "value", c), dtype=np.float64)

        def stack(suffix: str) -> FloatArray:
            if not names:
                return np.empty((len(t), 0))
            return np.column_stack([plain(f"{p}{suffix}") for p in names])

        lo = t["bin_lo"][0]
        hi = t["bin_hi"][0]
        edges = u.Quantity(np.append(lo.value, hi.value[-1]), lo.unit)
        return cls(
            nside=int(m["NSIDE"]),
            order=m["ORDER"],
            pixels=np.asarray(t["pixel"], dtype=np.intp),
            n_eff=plain("n_eff"),
            sf=plain("sf"),
            sf_err=plain("sf_err"),
            count=plain("count"),
            bin_edges=edges,
            separation=plain("separation"),
            sf_unit=u.Unit(m["SFUNIT"]) if m["SFUNIT"] else None,
            param_names=names,
            params=stack(""),
            params_lower=stack("_lower"),
            params_upper=stack("_upper"),
            chi2=plain("chi2"),
            n_point=int(m["NPOINT"]),
            taper=Taper(m["TAPER"]),
            radius=u.Quantity(m["RADIUS"], u.deg),
        )


def structure_function_skymap(
    values: ArrayLike | u.Quantity,
    errors: ArrayLike | u.Quantity | None,
    coords: SkyCoord,
    nside: int,
    radius: u.Quantity,
    bins: int | u.Quantity = 8,
    min_sep: u.Quantity | None = None,
    max_sep: u.Quantity | None = None,
    taper: TaperLike = Taper.COSINE,
    order: Literal["ring", "nested"] = "ring",
    weights: ArrayLike | None = None,
    n_point: Literal[2, 3] = 2,
    patches: int = 16,
    min_sources: float = 30.0,
    fit_model: FitModelLike | None = FitModel.BROKEN_POWER_LAW,
    pixels: ArrayLike | None = None,
    bin_tolerance: float = 0.05,
    pos_errors: PositionErrors | None = None,
    stencil: StencilOptions | None = None,
    rng: np.random.Generator | int | None = None,
) -> SkyMapResult:
    """Apodised structure functions (and fits) for every HEALPix cell.

    The grid uses the longitude/latitude of ``coords``' own frame, so pass
    Galactic coordinates for a Galactic map.

    Args:
        values: Measured values, plain or Quantity.
        errors: 1-sigma errors (same unit handling as ``values``) or ``None``.
        coords: Source positions (2D sky positions).
        nside: HEALPix resolution.
        radius: Taper scale: zero point (cosine) or FWHM (Gaussian).
        bins: Number of log bins, or explicit edges shared by all cells.
        min_sep: Lower edge for numeric ``bins`` (default: smallest separation
            in the catalogue).
        max_sep: Upper edge for numeric ``bins`` (default: ``radius``).
        taper: ``cosine`` or ``gaussian``.
        order: HEALPix pixel ordering.
        weights: Optional per-source weights, multiplied by the taper.
        n_point: Stencil size.
        patches: k-means patches per cell for the jackknife.
        min_sources: Skip cells with fewer effective sources.
        fit_model: Model to fit per cell (MAP), or ``None``.
        pixels: Only compute these cells (default: all with enough sources).
        bin_tolerance: See :func:`structurefunction.structure_function`.
        pos_errors: Optional positional errors.
        stencil: Three-point stencil options.
        rng: Random generator or seed.

    Returns:
        Per-cell structure functions and parameters.

    Raises:
        ValueError: For coordinates with distances.
    """
    taper = Taper(taper)
    gen = np.random.default_rng(rng)
    vals = _units.split_values(values, errors, weights)
    pos = _units.positions(coords)
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
    for k, (pix, idx_list) in enumerate(zip(targets, members, strict=True)):
        idx = np.asarray(idx_list, dtype=np.intp)
        if idx.size < 3:
            continue
        d = 2.0 * np.arcsin(
            np.clip(np.linalg.norm(pos.xyz[idx] - centres[k], axis=1) / 2.0, 0, 1)
        )
        w = taper_weights(d, radius_rad, taper) * vals.weights[idx]
        keep = w > 0
        idx, w = idx[keep], w[keep]
        n_eff = float(w.sum() ** 2 / np.sum(w**2)) if w.size else 0.0
        if n_eff < min_sources:
            continue
        v = _units.attach_unit(vals.values[idx], vals.unit)
        e = _units.attach_unit(vals.errors[idx], vals.unit)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", StructureFunctionWarning)
            res = structure_function(
                values=v,
                errors=e if errors is not None else None,
                coords=coords[idx],
                n_point=n_point,
                bins=edges_q,
                weights=w,
                patches=min(patches, max(int(n_eff) // 10, 2)),
                bin_tolerance=bin_tolerance,
                pos_errors=_subset_pos_errors(pos_errors, idx),
                stencil=stencil,
                rng=gen,
            )
        sf_q = res.sf
        if isinstance(sf_q, u.Quantity):
            sf_unit = sf_q.unit
        rows.append(int(pix))
        out["n_eff"].append(np.array([n_eff]))
        out["sf"].append(np.asarray(getattr(sf_q, "value", sf_q), dtype=np.float64))
        out["sf_err"].append(
            np.asarray(getattr(res.sf_err, "value", res.sf_err), dtype=np.float64)
        )
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
        return replace(pos_errors, sigma=sub(pos_errors.sigma))
    if isinstance(pos_errors, AxisPositionErrors):  # pragma: no branch
        return replace(
            pos_errors,
            sigma_ra=sub(pos_errors.sigma_ra),
            sigma_dec=sub(pos_errors.sigma_dec),
        )
    assert_never(pos_errors)
