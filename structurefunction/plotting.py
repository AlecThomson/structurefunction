"""Quick-look plots."""

from __future__ import annotations

from typing import TYPE_CHECKING

import astropy.units as u
import numpy as np

from structurefunction.fitting import evaluate
from structurefunction.skymap import healpix_map

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from structurefunction.fitting import FitResult
    from structurefunction.results import SFResult
    from structurefunction.skymap import SkyMapResult


def _readable_log_ticks(ax: Axes) -> None:
    """Label log axes at 1, 2 and 5 times powers of ten, as plain numbers.

    Matplotlib's default labels every minor tick when an axis spans less than
    about a decade, and those labels overlap.
    """
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
        axis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
        axis.set_minor_formatter(NullFormatter())


def plot_sf(
    result: SFResult,
    fit: FitResult | None = None,
    ax: Axes | None = None,
    n_draws: int = 200,
    rng: np.random.Generator | int | None = None,
) -> Figure:
    """Structure function with errors, saturation level and optional model band.

    Unreliable bins (``count < min_count``) are drawn hollow.
    """
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure

    if ax is None:
        fig, ax = plt.subplots()
    else:
        root = ax.get_figure(root=True)
        if not isinstance(root, Figure):
            msg = "ax must belong to a figure"
            raise TypeError(msg)
        fig = root
    unit = result.sf_unit
    y = result.sf
    yerr = result.sf_err
    r = result.separation.value
    good = result.reliable
    ax.errorbar(r[good], y[good], yerr[good], fmt="o", color="k", label="SF")
    ax.errorbar(
        r[~good],
        y[~good],
        yerr[~good],
        fmt="o",
        mfc="none",
        color="k",
        alpha=0.5,
        label=f"< {result.min_count} {'pairs' if result.n_point == 2 else 'triplets'}",
    )
    if result.region_size is not None and np.any(~result.resampling_reliable):
        ax.axvspan(
            result.region_size.to_value(result.separation.unit),
            np.nanmax(r) * 1.5,
            color="0.9",
            zorder=0,
            label="errors underestimated (larger than jackknife regions)",
        )
    sat = result.saturation
    ax.axhline(sat, ls="--", color="0.5", label=f"{result.n_point}-point plateau")
    if fit is not None:
        gen = np.random.default_rng(rng)
        grid = np.geomspace(r[good].min(), r[good].max(), 200)
        idx = gen.integers(0, fit.samples.shape[0], n_draws)
        draws = np.array([evaluate(fit, grid, fit.samples[i]) for i in idx])
        lo, hi = np.percentile(draws, [16, 84], axis=0)
        ax.fill_between(grid, lo, hi, color="C0", alpha=0.3)
        ax.plot(grid, evaluate(fit, grid), color="C0", label=str(fit.model))
    ax.set_xscale("log")
    ax.set_yscale("log")
    _readable_log_ticks(ax)
    ax.set_xlabel(f"Separation [{result.separation.unit:latex_inline}]")
    label = "SF" if unit is None else f"SF [{unit:latex_inline}]"
    ax.set_ylabel(label)
    ax.legend()
    return fig


def plot_skymap(
    result: SkyMapResult,
    name: str,
    bin_index: int | None = None,
    ax: Axes | None = None,
    resolution_deg: float = 0.5,
    cmap: str = "viridis",
) -> Figure:
    """Mollweide view of one sky-map quantity (see :meth:`SkyMapResult.map`).

    Longitude increases to the left, as on the sky.
    """
    import matplotlib.pyplot as plt
    from astropy_healpix import HEALPix
    from matplotlib.figure import Figure

    values = healpix_map(result, name, bin_index)
    hp = HEALPix(nside=result.nside, order=result.order)
    lon = np.arange(-180.0, 180.0 + resolution_deg, resolution_deg)
    lat = np.arange(-90.0, 90.0 + resolution_deg, resolution_deg)
    lon_g, lat_g = np.meshgrid(lon, lat)
    pix = hp.lonlat_to_healpix(lon_g * u.deg, lat_g * u.deg)
    img = values[pix]
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(projection="mollweide")
    else:
        root = ax.get_figure(root=True)
        if not isinstance(root, Figure):
            msg = "ax must belong to a figure"
            raise TypeError(msg)
        fig = root
    mesh = ax.pcolormesh(
        -np.radians(lon_g), np.radians(lat_g), img, cmap=cmap, shading="auto"
    )
    ax.set_xticklabels([])
    label = name
    if bin_index is not None:
        lo, hi = result.bin_edges[bin_index], result.bin_edges[bin_index + 1]
        unit = f"{result.bin_edges.unit:latex_inline}"
        label = f"{name}, separations {lo.value:.3g} to {hi.value:.3g} {unit}"
    fig.colorbar(mesh, ax=ax, orientation="horizontal", label=label)
    ax.grid(visible=True)
    return fig
