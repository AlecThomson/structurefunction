"""Quick-look plots (needs the ``plot`` extra)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import astropy.units as u
import numpy as np

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from structurefunction.fitting import FitResult
    from structurefunction.results import SFResult
    from structurefunction.skymap import SkyMapResult
    from structurefunction.types import FloatArray


def plot_sf(
    result: SFResult[FloatArray] | SFResult[u.Quantity],
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
    unit = result.sf.unit if isinstance(result.sf, u.Quantity) else None
    y = np.asarray(u.Quantity(result.sf).value if unit else result.sf)
    yerr = np.asarray(u.Quantity(result.sf_err).value if unit else result.sf_err)
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
    sat = float(u.Quantity(result.saturation).value if unit else result.saturation)
    ax.axhline(sat, ls="--", color="0.5", label=f"{result.n_point}-point plateau")
    if fit is not None:
        gen = np.random.default_rng(rng)
        grid = np.geomspace(r[good].min(), r[good].max(), 200)
        idx = gen.integers(0, fit.samples.shape[0], n_draws)
        draws = np.array([fit.evaluate(grid, fit.samples[i]) for i in idx])
        lo, hi = np.percentile(draws, [16, 84], axis=0)
        ax.fill_between(grid, lo, hi, color="C0", alpha=0.3)
        ax.plot(grid, fit.evaluate(grid), color="C0", label=str(fit.model))
    ax.set_xscale("log")
    ax.set_yscale("log")
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

    values = result.map(name, bin_index)
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
    label = name if bin_index is None else f"{name} (bin {bin_index})"
    fig.colorbar(mesh, ax=ax, orientation="horizontal", label=label)
    ax.grid(visible=True)
    return fig
