from __future__ import annotations

import warnings

import astropy.units as u
import numpy as np
import pytest

from structurefunction import StructureFunctionWarning, structure_function

from conftest import make_catalogue

EDGES = np.geomspace(0.1, 4.0, 6) * u.deg


def test_jackknife_samples_are_leave_one_out() -> None:
    cat = make_catalogue(500, seed=21)
    labels = (cat.lon // 2).astype(int)  # 4 vertical strips
    full = structure_function(
        cat.values,
        cat.errors,
        cat.coords,
        bins=EDGES,
        weights=cat.weights,
        jk_regions=labels,
        bin_tolerance=0.0,
    )
    loo = []
    for p in np.unique(labels):
        keep = labels != p
        r = structure_function(
            cat.values[keep],
            cat.errors[keep],
            cat.coords[keep],
            bins=EDGES,
            weights=cat.weights[keep],
            jk_regions=labels[keep],
            bin_tolerance=0.0,
            error_method="none",
        )
        loo.append(r.sf)
    samples = np.array(loo)
    n = samples.shape[0]
    expected = (n - 1) / n * np.cov(samples.T, bias=True) * n
    assert full.cov is not None
    np.testing.assert_allclose(full.cov, expected, rtol=1e-9)


def test_bootstrap_is_reproducible() -> None:
    cat = make_catalogue(500, seed=22)
    a, b = (
        structure_function(
            cat.values,
            cat.errors,
            cat.coords,
            bins=EDGES,
            jk_regions=16,
            error_method="bootstrap",
            n_boot=200,
            rng=5,
        )
        for _ in range(2)
    )
    np.testing.assert_array_equal(a.cov, b.cov)


def test_guard_rail_warns_for_few_regions() -> None:
    cat = make_catalogue(300, seed=23)
    with warnings.catch_warnings():
        warnings.simplefilter("error", StructureFunctionWarning)
        with pytest.raises(StructureFunctionWarning, match="regions"):
            structure_function(
                cat.values, cat.errors, cat.coords, bins=10, jk_regions=4
            )


def test_reliable_flags_low_counts() -> None:
    cat = make_catalogue(200, seed=24)
    res = structure_function(
        cat.values, cat.errors, cat.coords, bins=EDGES, jk_regions=4, min_count=50
    )
    np.testing.assert_array_equal(res.reliable, res.count >= 50)
