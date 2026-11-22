"""Timing and accuracy of the kernels against catalogue size.

Usage: uv run python benchmarks/bench_sf.py --n 1000 10000 100000 1000000
"""

from __future__ import annotations

import argparse
import time
import warnings
from functools import partial
from typing import TYPE_CHECKING

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord

from structurefunction import (
    StencilOptions,
    StructureFunctionWarning,
    structure_function,
)

if TYPE_CHECKING:
    from collections.abc import Callable


def catalogue(n: int, seed: int) -> tuple[SkyCoord, np.ndarray, np.ndarray]:
    """Uniform all-sky sources with a smooth field plus noise."""
    rng = np.random.default_rng(seed)
    lon = rng.uniform(0, 360, n)
    lat = np.degrees(np.arcsin(rng.uniform(-1, 1, n)))
    f = np.sin(np.radians(lon) * 5) * np.cos(np.radians(lat) * 3) * 20
    sigma = np.full(n, 2.0)
    return SkyCoord(lon * u.deg, lat * u.deg), f + rng.normal(0, sigma), sigma


def timed[T](label: str, n: int, call: Callable[[], T]) -> T:
    """Run ``call`` once and print its wall time."""
    t0 = time.perf_counter()
    res = call()
    print(f"{label:<28} N={n:>9,d}  {time.perf_counter() - t0:8.2f} s")
    return res


def main() -> None:
    """Run the benchmark for each requested catalogue size."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, nargs="+", default=[1_000, 10_000, 100_000])
    parser.add_argument("--three-point-max", type=int, default=100_000)
    args = parser.parse_args()
    warnings.simplefilter("ignore", StructureFunctionWarning)

    c, f, s = catalogue(2000, 0)  # compile everything once
    structure_function(f, s, c, bins=4)
    structure_function(f, s, c, n_point=3, bins=4, min_sep=1 * u.deg, max_sep=5 * u.deg)

    for n in args.n:
        c, f, s = catalogue(n, 1)
        exact = timed(
            "2-point, all seps, exact",
            n,
            partial(structure_function, f, s, c, bins=15, bin_tolerance=0.0),
        )
        approx = timed(
            "2-point, all seps, tol 0.05",
            n,
            partial(structure_function, f, s, c, bins=15, bin_tolerance=0.05),
        )
        with np.errstate(invalid="ignore", divide="ignore"):  # empty bins
            dev = np.nanmax(np.abs(approx.sf - exact.sf) / exact.sf_err)
        print(f"{'':<28} max |approx - exact| / sigma = {dev:.3f}")
        if n <= args.three_point_max:
            timed(
                "3-point binned, 0.1-5 deg",
                n,
                partial(
                    structure_function,
                    f,
                    s,
                    c,
                    n_point=3,
                    bins=8,
                    min_sep=0.1 * u.deg,
                    max_sep=5 * u.deg,
                    stencil=StencilOptions(mode="binned"),
                ),
            )


if __name__ == "__main__":
    main()
