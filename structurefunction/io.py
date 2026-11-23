"""Convert results to and from astropy tables and files.

``to_table`` and ``write`` accept either result type. ECSV keeps everything;
FITS works too (header keys are kept short for it).
"""

from __future__ import annotations

from functools import singledispatch
from typing import TYPE_CHECKING

import astropy.units as u
import numpy as np
from astropy.table import QTable, Table

from structurefunction.results import SFResult
from structurefunction.skymap import SkyMapResult
from structurefunction.types import ErrorMethod, FloatArray, Taper

if TYPE_CHECKING:
    from pathlib import Path


@singledispatch
def to_table(result: object) -> QTable:
    """One row per bin (``SFResult``) or per cell (``SkyMapResult``).

    Raises:
        TypeError: For anything else.
    """
    msg = f"cannot convert {type(result).__name__} to a table"
    raise TypeError(msg)


def write(
    result: SFResult | SkyMapResult,
    path: str | Path,
    overwrite: bool = False,
) -> None:
    """Write a result to any astropy table format (ECSV recommended)."""
    to_table(result).write(path, overwrite=overwrite)


def _column_values(table: QTable, name: str) -> FloatArray:
    column = table[name]
    return np.asarray(getattr(column, "value", column), dtype=np.float64)


# --------------------------------------------------------------------------
# Structure functions


@to_table.register
def _sf_to_table(result: SFResult) -> QTable:
    def with_unit(values: FloatArray) -> FloatArray | u.Quantity:
        return values if result.sf_unit is None else u.Quantity(values, result.sf_unit)

    table = QTable()
    table["separation"] = result.separation
    table["bin_lo"] = result.bin_edges[:-1]
    table["bin_hi"] = result.bin_edges[1:]
    table["sf"] = with_unit(result.sf)
    table["sf_err"] = with_unit(result.sf_err)
    table["noise_bias"] = with_unit(result.noise_bias)
    table["count"] = result.count
    table["weight"] = result.weight
    table["reliable"] = result.reliable
    if result.cov is not None:
        table["cov"] = result.cov
    table.meta.update(
        NPOINT=result.n_point,
        NREGIONS=result.n_regions,
        ERRMETH=str(result.error_method),
        MINCOUNT=result.min_count,
        SATURATE=result.saturation,
        SFUNIT="" if result.sf_unit is None else str(result.sf_unit),
        REGSIZE=(
            float("nan")
            if result.region_size is None
            else float(result.region_size.value)
        ),
        REGUNIT="" if result.region_size is None else str(result.region_size.unit),
    )
    return table


def sf_from_table(table: Table) -> SFResult:
    """Rebuild a structure function from :func:`to_table` output."""
    qtable = QTable(table)
    meta = qtable.meta
    lo, hi = qtable["bin_lo"], qtable["bin_hi"]
    region_size = (
        u.Quantity(meta["REGSIZE"], u.Unit(meta["REGUNIT"]))
        if meta["REGUNIT"]
        else None
    )
    return SFResult(
        sf=_column_values(qtable, "sf"),
        sf_err=_column_values(qtable, "sf_err"),
        noise_bias=_column_values(qtable, "noise_bias"),
        saturation=float(meta["SATURATE"]),
        sf_unit=u.Unit(meta["SFUNIT"]) if meta["SFUNIT"] else None,
        cov=np.asarray(qtable["cov"]) if "cov" in qtable.colnames else None,
        separation=qtable["separation"],
        bin_edges=u.Quantity(np.append(lo.value, hi.value[-1]), lo.unit),
        count=_column_values(qtable, "count"),
        weight=_column_values(qtable, "weight"),
        n_point=int(meta["NPOINT"]),
        n_regions=int(meta["NREGIONS"]),
        error_method=ErrorMethod(meta["ERRMETH"]),
        min_count=int(meta["MINCOUNT"]),
        reliable=np.asarray(qtable["reliable"], dtype=bool),
        region_size=region_size,
    )


def read_sf(path: str | Path) -> SFResult:
    """Read a structure function written by :func:`write`."""
    return sf_from_table(Table.read(path))


# --------------------------------------------------------------------------
# Sky maps


@to_table.register
def _skymap_to_table(result: SkyMapResult) -> QTable:
    table = QTable()
    table["pixel"] = result.pixels
    table["n_eff"] = result.n_eff

    def with_unit(values: FloatArray) -> FloatArray | u.Quantity:
        return values if result.sf_unit is None else u.Quantity(values, result.sf_unit)

    table["sf"] = with_unit(result.sf)
    table["sf_err"] = with_unit(result.sf_err)
    table["count"] = result.count
    table["separation"] = u.Quantity(result.separation, result.bin_edges.unit)
    table["chi2"] = result.chi2
    for index, name in enumerate(result.param_names):
        table[name] = result.params[:, index]
        table[f"{name}_lower"] = result.params_lower[:, index]
        table[f"{name}_upper"] = result.params_upper[:, index]
    # Bin edges as (identical) vector columns keep the table valid in FITS.
    unit = result.bin_edges.unit
    edges = result.bin_edges.to_value(unit)
    table["bin_lo"] = u.Quantity(np.broadcast_to(edges[:-1], result.sf.shape), unit)
    table["bin_hi"] = u.Quantity(np.broadcast_to(edges[1:], result.sf.shape), unit)
    table.meta.update(
        NSIDE=result.nside,
        ORDER=result.order,
        NPOINT=result.n_point,
        TAPER=str(result.taper),
        RADIUS=float(result.radius.to_value(u.deg)),
        SFUNIT="" if result.sf_unit is None else str(result.sf_unit),
    )
    return table


def skymap_from_table(table: Table) -> SkyMapResult:
    """Rebuild a sky map from :func:`to_table` output."""
    qtable = QTable(table)
    meta = qtable.meta
    names = tuple(
        column.removesuffix("_lower")
        for column in qtable.colnames
        if column.endswith("_lower")
    )

    def stacked(suffix: str) -> FloatArray:
        if not names:
            return np.empty((len(qtable), 0))
        return np.column_stack([_column_values(qtable, f"{n}{suffix}") for n in names])

    lo, hi = qtable["bin_lo"][0], qtable["bin_hi"][0]
    return SkyMapResult(
        nside=int(meta["NSIDE"]),
        order=meta["ORDER"],
        pixels=np.asarray(qtable["pixel"], dtype=np.intp),
        n_eff=_column_values(qtable, "n_eff"),
        sf=_column_values(qtable, "sf"),
        sf_err=_column_values(qtable, "sf_err"),
        count=_column_values(qtable, "count"),
        bin_edges=u.Quantity(np.append(lo.value, hi.value[-1]), lo.unit),
        separation=_column_values(qtable, "separation"),
        sf_unit=u.Unit(meta["SFUNIT"]) if meta["SFUNIT"] else None,
        param_names=names,
        params=stacked(""),
        params_lower=stacked("_lower"),
        params_upper=stacked("_upper"),
        chi2=_column_values(qtable, "chi2"),
        n_point=int(meta["NPOINT"]),
        taper=Taper(meta["TAPER"]),
        radius=u.Quantity(meta["RADIUS"], u.deg),
    )


def read_skymap(path: str | Path) -> SkyMapResult:
    """Read a sky map written by :func:`write`."""
    return skymap_from_table(Table.read(path))
