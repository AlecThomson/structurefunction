"""Command-line interface.

``structurefunction sf``: one structure function from a table.
``structurefunction skymap``: apodised structure functions on a HEALPix grid.
"""

from __future__ import annotations

from argparse import ArgumentParser, Namespace
from pathlib import Path

import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table
from capn_crunch import add_options_to_parser, create_options_from_parser

from structurefunction.core import structure_function_from_table
from structurefunction.io import write
from structurefunction.options import StencilOptions
from structurefunction.skymap import structure_function_skymap
from structurefunction.types import ErrorMethod, FitModel, Taper


def _add_common(parser: ArgumentParser) -> None:
    parser.add_argument("table", type=Path, help="Input table (ECSV, FITS, ...)")
    parser.add_argument("output", type=Path, help="Output table (ECSV or FITS)")
    parser.add_argument("--value", required=True, help="Value column")
    parser.add_argument("--error", default=None, help="1-sigma error column")
    parser.add_argument(
        "--lon", default="ra", help="Longitude column (deg if unitless)"
    )
    parser.add_argument(
        "--lat", default="dec", help="Latitude column (deg if unitless)"
    )
    parser.add_argument("--frame", default="icrs", help="Frame of lon/lat")
    parser.add_argument("--n-point", type=int, choices=(2, 3), default=2)
    # Three-point stencil flags (--angle-tol-deg, --sep-tol, --mode) come from
    # the StencilOptions fields and their docstrings.
    add_options_to_parser(parser, StencilOptions)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--plot", type=Path, default=None, help="Save a quick-look plot"
    )
    parser.add_argument("--overwrite", action="store_true")


def get_parser(parser: ArgumentParser | None = None) -> ArgumentParser:
    """Build (or extend) the argument parser."""
    if parser is None:
        parser = ArgumentParser(description="Structure functions of sky data.")
    sub = parser.add_subparsers(dest="command", required=True)

    sf = sub.add_parser("sf", help="One structure function for the whole table")
    _add_common(sf)
    sf.add_argument("--bins", type=int, default=10, help="Number of log bins")
    sf.add_argument(
        "--jk-regions", type=int, default=64, help="Number of jackknife regions"
    )
    sf.add_argument(
        "--error-method", choices=[m.value for m in ErrorMethod], default="jackknife"
    )

    sky = sub.add_parser("skymap", help="Apodised structure functions per HEALPix cell")
    _add_common(sky)
    sky.add_argument(
        "--nside",
        default="auto",
        help="HEALPix nside, or 'auto' to sample the taper at twice per FWHM",
    )
    sky.add_argument("--radius-deg", type=float, required=True, help="Taper scale")
    sky.add_argument("--taper", choices=[t.value for t in Taper], default="cosine")
    sky.add_argument("--bins", type=int, default=8)
    sky.add_argument("--min-sep-deg", type=float, default=None)
    sky.add_argument("--max-sep-deg", type=float, default=None)
    sky.add_argument(
        "--jk-regions", type=int, default=16, help="Jackknife regions per cell"
    )
    sky.add_argument("--min-sources", type=float, default=30.0)
    sky.add_argument(
        "--fit-model",
        choices=[*(m.value for m in FitModel), "none"],
        default="broken_power_law",
    )
    sky.add_argument("--plot-quantity", default="slope_1", help="Quantity for --plot")
    return parser


def _column(table: Table, name: str) -> u.Quantity:
    col = table[name]
    return u.Quantity(col) if getattr(col, "unit", None) is not None else col


def cli(args: Namespace | None = None) -> None:
    """Run the command-line interface."""
    if args is None:
        args = get_parser().parse_args()
    table = Table.read(args.table)
    stencil = create_options_from_parser(args, StencilOptions)
    if args.command == "sf":
        res = structure_function_from_table(
            table=table,
            value=args.value,
            error=args.error,
            lon=args.lon,
            lat=args.lat,
            frame=args.frame,
            n_point=args.n_point,
            bins=args.bins,
            jk_regions=args.jk_regions,
            error_method=args.error_method,
            stencil=stencil,
            rng=args.seed,
        )
        write(res, args.output, overwrite=args.overwrite)
        if args.plot is not None:
            from structurefunction.plotting import plot_sf

            plot_sf(res).savefig(args.plot)
        return

    def angle(name: str) -> u.Quantity:
        c = _column(table, name)
        return c if isinstance(c, u.Quantity) else u.Quantity(c, u.deg)

    sky = structure_function_skymap(
        values=_column(table, args.value),
        errors=None if args.error is None else _column(table, args.error),
        coords=SkyCoord(angle(args.lon), angle(args.lat), frame=args.frame),
        nside="auto" if args.nside == "auto" else int(args.nside),
        radius=args.radius_deg * u.deg,
        bins=args.bins,
        min_sep=None if args.min_sep_deg is None else args.min_sep_deg * u.deg,
        max_sep=None if args.max_sep_deg is None else args.max_sep_deg * u.deg,
        taper=args.taper,
        n_point=args.n_point,
        jk_regions=args.jk_regions,
        min_sources=args.min_sources,
        fit_model=None if args.fit_model == "none" else args.fit_model,
        stencil=stencil,
        rng=args.seed,
    )
    write(sky, args.output, overwrite=args.overwrite)
    if args.plot is not None:
        from structurefunction.plotting import plot_skymap

        plot_skymap(sky, args.plot_quantity).savefig(args.plot)


if __name__ == "__main__":
    cli()
