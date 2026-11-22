from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import astropy.units as u
import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from astropy_healpix import HEALPix

from structurefunction import (
    SkyMapResult,
    StructureFunctionWarning,
    Taper,
    structure_function,
    structure_function_skymap,
)
from structurefunction.skymap import pixel_size, recommend_nside, taper_weights
from structurefunction.types import TaperLike

if TYPE_CHECKING:
    from pathlib import Path


def sky(n: int, seed: int) -> tuple[SkyCoord, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    lon = rng.uniform(0, 360, n)
    lat = np.degrees(np.arcsin(rng.uniform(-1, 1, n)))
    sigma = np.full(n, 0.5)
    values = rng.normal(0, 2.0, n) + rng.normal(0, sigma)
    return SkyCoord(lon * u.deg, lat * u.deg), values, sigma


def test_tapers() -> None:
    d = np.array([0.0, 5.0, 10.0, 20.0])
    cos = taper_weights(d, 10.0, Taper.COSINE)
    np.testing.assert_allclose(cos, [1.0, 0.5, 0.0, 0.0], atol=1e-12)
    gauss = taper_weights(d, 10.0, "gaussian")  # FWHM 10
    np.testing.assert_allclose(gauss[:2], [1.0, 0.5], rtol=1e-12)


def test_cell_matches_direct_weighted_call() -> None:
    c, f, s = sky(20000, 1)
    edges = np.geomspace(0.5, 8, 5) * u.deg
    res = structure_function_skymap(
        f, s, c, nside=2, radius=15 * u.deg, bins=edges, patches=8,
        fit_model=None, pixels=[5], bin_tolerance=0.0, rng=0,
    )  # fmt: skip
    lon, lat = HEALPix(nside=2).healpix_to_lonlat([5])
    centre = SkyCoord(lon, lat)
    d = c.separation(centre).to_value(u.deg)
    w = taper_weights(d, 15.0)
    keep = w > 0
    direct = structure_function(
        f[keep], s[keep], c[keep], bins=edges, weights=w[keep], patches=8,
        bin_tolerance=0.0, rng=0,
    )  # fmt: skip
    np.testing.assert_allclose(res.sf[0], direct.sf, rtol=1e-10)
    np.testing.assert_allclose(res.count[0], direct.count)


def test_white_noise_map_is_flat(tmp_path: Path) -> None:
    c, f, s = sky(30000, 2)
    res = structure_function_skymap(
        f, s, c, nside=1, radius=40 * u.deg, bins=4, min_sep=2 * u.deg,
        patches=16, fit_model="power_law", rng=1,
    )  # fmt: skip
    assert res.pixels.size == 12
    # Debiased SF of white noise with variance 4: 2 * 4 = 8 everywhere.
    pull = (res.sf - 8.0) / res.sf_err
    assert np.all(np.abs(pull) < 5)
    slope = res.map("slope")
    assert np.all(np.abs(slope - 0.0) < 0.5)
    out = tmp_path / "map.ecsv"
    res.write(out)
    back = SkyMapResult.read(out)
    np.testing.assert_allclose(back.map("sf", 1), res.map("sf", 1))
    np.testing.assert_allclose(back.map("slope"), slope)


def test_sparse_cells_are_skipped() -> None:
    c, f, s = sky(500, 3)
    res = structure_function_skymap(
        f, s, c, nside=4, radius=5 * u.deg, bins=3, min_sep=0.5 * u.deg,
        min_sources=30, fit_model=None,
    )  # fmt: skip
    m = res.map("sf", 0)
    assert np.isnan(m).sum() == m.size - res.pixels.size
    with pytest.raises(ValueError, match="bin_index"):
        res.map("sf")
    with pytest.raises(KeyError):
        res.map("nope")


def test_cli_and_plot(tmp_path: Path) -> None:
    import matplotlib as mpl

    mpl.use("Agg")
    from astropy.table import Table

    from structurefunction.cli import cli, get_parser

    c, f, s = sky(20000, 4)
    t = Table({"ra": c.ra, "dec": c.dec, "rm": f, "rm_err": s})
    src = tmp_path / "in.fits"
    t.write(src)
    out = tmp_path / "map.fits"
    png = tmp_path / "map.png"
    args = get_parser().parse_args(
        [
            "skymap", str(src), str(out), "--value", "rm", "--error", "rm_err",
            "--nside", "1", "--radius-deg", "40", "--bins", "4",
            "--min-sep-deg", "2", "--fit-model", "power_law", "--plot", str(png),
            "--plot-quantity", "slope", "--seed", "0",
        ]
    )  # fmt: skip
    cli(args)
    assert SkyMapResult.read(out).pixels.size == 12
    assert png.stat().st_size > 0


@pytest.mark.parametrize("radius_deg", [0.5, 3.0, 10.0, 40.0])
@pytest.mark.parametrize("taper", ["cosine", "gaussian"])
def test_recommend_nside_samples_the_taper(radius_deg: float, taper: TaperLike) -> None:
    radius = radius_deg * u.deg
    nside = recommend_nside(radius, taper)
    assert nside & (nside - 1) == 0  # power of two
    assert pixel_size(nside) <= radius / 2
    if nside > 1:
        assert pixel_size(nside // 2) > radius / 2


def test_auto_nside_matches_explicit() -> None:
    c, f, s = sky(20000, 5)
    radius = 40 * u.deg
    auto = structure_function_skymap(
        f, s, c, radius=radius, bins=3, min_sep=2 * u.deg, fit_model=None, rng=0
    )
    explicit = structure_function_skymap(
        f, s, c, radius=radius, nside=recommend_nside(radius), bins=3,
        min_sep=2 * u.deg, fit_model=None, rng=0,
    )  # fmt: skip
    assert auto.nside == explicit.nside == recommend_nside(radius)
    np.testing.assert_allclose(auto.map("sf", 1), explicit.map("sf", 1))


def test_sparse_catalogue_warns() -> None:
    c, f, s = sky(300, 6)
    with warnings.catch_warnings():
        warnings.simplefilter("error", StructureFunctionWarning)
        with pytest.raises(StructureFunctionWarning, match="effective sources"):
            structure_function_skymap(
                f, s, c, radius=5 * u.deg, bins=3, min_sep=0.5 * u.deg, fit_model=None
            )
