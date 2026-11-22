from __future__ import annotations

import typing
from importlib import resources

import astropy.units as u
import numpy as np
import pytest
from astropy.table import Table

from structurefunction import structure_function_from_table, types
from structurefunction.cli import cli, get_parser
from structurefunction.io import read_sf, write

if typing.TYPE_CHECKING:
    from enum import StrEnum
    from pathlib import Path


def mao2010() -> Table:
    path = resources.files("structurefunction.data") / "mao2010_ngp.ecsv"
    with resources.as_file(path) as p:
        return Table.read(p)


def test_result_round_trips_through_ecsv(tmp_path: Path) -> None:
    res = structure_function_from_table(
        mao2010(), "rm", "rm_err", bins=6, regions=20, rng=0
    )
    out = tmp_path / "sf.ecsv"
    write(res, out)
    back = read_sf(out)
    assert isinstance(back.sf, u.Quantity)
    assert back.sf.unit == (u.rad / u.m**2) ** 2
    np.testing.assert_allclose(back.sf.value, u.Quantity(res.sf).value)
    assert back.cov is not None
    assert res.cov is not None
    np.testing.assert_allclose(back.cov, res.cov)
    np.testing.assert_array_equal(back.reliable, res.reliable)
    assert back.region_size == res.region_size


def test_cli_writes_table(tmp_path: Path) -> None:
    src = tmp_path / "in.ecsv"
    mao2010().write(src)
    out = tmp_path / "out.ecsv"
    args = get_parser().parse_args(
        [
            "sf",
            str(src),
            str(out),
            "--value",
            "rm",
            "--error",
            "rm_err",
            "--bins",
            "5",
            "--regions",
            "16",
            "--seed",
            "1",
        ]
    )
    cli(args)
    t = Table.read(out)
    assert len(t) == 5
    assert t.meta["NPOINT"] == 2


@pytest.mark.parametrize(
    ("enum", "alias"),
    [
        (types.ErrorMethod, types.ErrorMethodLike),
        (types.CrossRegionWeight, types.CrossRegionWeightLike),
        (types.StencilMode, types.StencilModeLike),
        (types.FitModel, types.FitModelLike),
        (types.Sampler, types.SamplerLike),
        (types.Likelihood, types.LikelihoodLike),
        (types.RAErrorConvention, types.RAErrorConventionLike),
    ],
)
def test_string_aliases_match_enums(
    enum: type[StrEnum], alias: typing.TypeAliasType
) -> None:
    literal = next(a for a in typing.get_args(alias.__value__) if a is not enum)
    assert set(typing.get_args(literal)) == {m.value for m in enum}
