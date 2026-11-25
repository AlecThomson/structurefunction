from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import astropy.units as u
import numpy as np
import pytest
from astropy.coordinates import SkyCoord

if TYPE_CHECKING:
    from structurefunction.types import FloatArray


@dataclass(frozen=True)
class Catalogue:
    lon: FloatArray
    lat: FloatArray
    values: FloatArray
    errors: FloatArray
    weights: FloatArray

    @property
    def coords(self) -> SkyCoord:
        return SkyCoord(self.lon * u.deg, self.lat * u.deg)


def make_catalogue(n: int, seed: int, size_deg: float = 8.0) -> Catalogue:
    rng = np.random.default_rng(seed)
    lon = rng.uniform(0, size_deg, n)
    lat = rng.uniform(-size_deg / 2, size_deg / 2, n)
    values = np.sin(np.radians(lon) * 40) + 0.3 * rng.normal(size=n)
    errors = rng.uniform(0.05, 0.2, n)
    weights = rng.uniform(0.5, 1.5, n)
    return Catalogue(lon, lat, values, errors, weights)


@pytest.fixture
def small() -> Catalogue:
    return make_catalogue(400, seed=1)
