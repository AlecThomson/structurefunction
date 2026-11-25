"""How the three-point stencil tolerances affect the measured slope.

Gaussian random fields with shell-averaged power ``k^-beta`` are sampled at
random positions. The three-point SF slope over 0.1 to 0.5 degrees is measured
for each pair of tolerances and compared with the tightest setting. Loose
tolerances admit triplets that are not quite straight or not quite symmetric,
which lets the field's gradient leak in and flattens steep slopes.

Usage: uv run python benchmarks/stencil_tolerances.py
"""

from __future__ import annotations

import warnings
from itertools import product

import astropy.units as u
import numpy as np

from structurefunction import (
    StencilOptions,
    StructureFunctionWarning,
    structure_function,
)
from structurefunction.simulate import gaussian_random_field, sample_field


def slope(
    coords: object, values: np.ndarray, angle_tol: float, sep_tol: float, mode: str
) -> tuple[float, float]:
    """Fitted log-log slope and the total number of triplets."""
    edges = np.geomspace(0.1, 0.5, 5) * u.deg
    res = structure_function(
        values,
        None,
        coords,  # type: ignore[arg-type]
        n_point=3,
        bins=edges,
        jk_regions=1,
        error_method="none",
        stencil=StencilOptions(angle_tol_deg=angle_tol, sep_tol=sep_tol, mode=mode),  # type: ignore[arg-type]
    )
    fit = np.polyfit(np.log(res.separation.value), np.log(res.sf), 1)[0]
    return float(fit), float(res.count.sum())


def main() -> None:
    """Print slope bias against the tightest tolerances for each setting."""
    warnings.simplefilter("ignore", StructureFunctionWarning)
    angles = (2.0, 5.0, 10.0, 15.0, 20.0)
    seps = (0.02, 0.05, 0.1, 0.2)
    print("beta mode   angle  sep   slope   bias  triplets")
    for beta in (2.0, 3.0, 4.0):
        # Average over a few realisations so the bias is not just noise.
        mocks = [
            sample_field(
                gaussian_random_field(beta, 1024, rng=seed),
                10 * u.deg,
                30000,
                rng=seed + 10,
            )
            for seed in range(3)
        ]
        for mode in ("exact", "binned"):
            results = {}
            for angle, sep in product(angles, seps):
                fits = [slope(c, v, angle, sep, mode) for c, v in mocks]
                results[angle, sep] = (
                    np.mean([f[0] for f in fits]),
                    np.mean([f[1] for f in fits]),
                )
            reference = results[angles[0], seps[0]][0]
            for (angle, sep), (value, count) in results.items():
                print(
                    f"{beta:4.1f} {mode:6s} {angle:5.1f} {sep:5.2f} {value:6.3f} "
                    f"{value - reference:+6.3f} {count:9.0f}"
                )


if __name__ == "__main__":
    main()
