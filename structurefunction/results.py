"""Result container for structure functions, with astropy Table round-trips."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import astropy.units as u
import numpy as np
from astropy.table import QTable, Table

from structurefunction.types import ErrorMethod, FloatArray

if TYPE_CHECKING:
    from pathlib import Path


@dataclass(frozen=True, slots=True)
class SFResult[V: (FloatArray, u.Quantity)]:
    """A binned, noise-debiased structure function and its uncertainties.

    ``V`` is ``Quantity`` when the input values had units (the structure
    function then carries ``unit**2``) and a plain float array otherwise.
    """

    sf: V
    """Noise-debiased structure function per bin."""
    sf_err: V
    """Square root of the covariance diagonal (NaN without resampling)."""
    noise_bias: V
    """Noise bias subtracted per bin, ``<sum_i c_i^2 sigma_i^2>``."""
    saturation: V
    """Large-scale plateau, ``sum c^2 (Var(f) - <sigma^2>)`` (noise removed)."""
    cov: FloatArray | None
    """Bin-to-bin covariance of ``sf`` (in ``unit**4``), or ``None``."""
    separation: u.Quantity
    """Weighted mean separation of the contributing pairs or triplets."""
    bin_edges: u.Quantity
    """Separation bin edges."""
    count: FloatArray
    """Number of contributing pairs or triplets per bin."""
    weight: FloatArray
    """Sum of stencil weights per bin."""
    n_point: int
    """Number of points in the stencil."""
    n_patch: int
    """Number of spatial patches used for resampling."""
    error_method: ErrorMethod
    """How ``cov`` was estimated."""
    min_count: int
    """Bins with fewer contributions are flagged as unreliable."""

    @property
    def reliable(self) -> np.ndarray[tuple[int], np.dtype[np.bool_]]:
        """Bins with at least ``min_count`` contributions and a finite estimate."""
        sf = np.asarray(u.Quantity(self.sf).value if self.has_units else self.sf)
        return (self.count >= self.min_count) & np.isfinite(sf)

    @property
    def has_units(self) -> bool:
        """Whether the input values carried units."""
        return isinstance(self.sf, u.Quantity)

    def to_table(self) -> QTable:
        """One row per bin; covariance as a vector column; settings in ``meta``."""
        t = QTable()
        t["separation"] = self.separation
        t["bin_lo"] = self.bin_edges[:-1]
        t["bin_hi"] = self.bin_edges[1:]
        t["sf"] = self.sf
        t["sf_err"] = self.sf_err
        t["noise_bias"] = self.noise_bias
        t["count"] = self.count
        t["weight"] = self.weight
        t["reliable"] = self.reliable
        if self.cov is not None:
            t["cov"] = self.cov
        t.meta.update(
            n_point=self.n_point,
            n_patch=self.n_patch,
            error_method=str(self.error_method),
            min_count=self.min_count,
            saturation=(
                float(u.Quantity(self.saturation).value)
                if self.has_units
                else float(np.asarray(self.saturation))
            ),
            unit=str(u.Quantity(self.sf).unit) if self.has_units else None,
        )
        return t

    def write(self, path: str | Path, overwrite: bool = False) -> None:
        """Write to any astropy-writable table format (ECSV recommended)."""
        self.to_table().write(path, overwrite=overwrite)

    @classmethod
    def from_table(cls, table: Table) -> SFResult[FloatArray] | SFResult[u.Quantity]:
        """Rebuild a result written by :meth:`to_table`."""
        t = QTable(table)
        meta = t.meta
        unit = u.Unit(meta["unit"]) if meta.get("unit") else None
        edges = np.append(t["bin_lo"].value, t["bin_hi"].value[-1]) * t["bin_lo"].unit
        cov = np.asarray(t["cov"]) if "cov" in t.colnames else None

        def values(name: str) -> FloatArray:
            col = t[name]
            return np.asarray(col.value if hasattr(col, "unit") else col, float)

        sf = values("sf")
        sf_err = values("sf_err")
        bias = values("noise_bias")
        sat = np.asarray(meta["saturation"], dtype=np.float64)
        sep = t["separation"]
        count = values("count")
        weight = values("weight")
        n_point = int(meta["n_point"])
        n_patch = int(meta["n_patch"])
        method = ErrorMethod(meta["error_method"])
        min_count = int(meta["min_count"])
        if unit is None:
            return SFResult[FloatArray](
                sf=sf,
                sf_err=sf_err,
                noise_bias=bias,
                saturation=sat,
                cov=cov,
                separation=sep,
                bin_edges=edges,
                count=count,
                weight=weight,
                n_point=n_point,
                n_patch=n_patch,
                error_method=method,
                min_count=min_count,
            )
        return SFResult[u.Quantity](
            sf=u.Quantity(sf, unit),
            sf_err=u.Quantity(sf_err, unit),
            noise_bias=u.Quantity(bias, unit),
            saturation=u.Quantity(sat, unit),
            cov=cov,
            separation=sep,
            bin_edges=edges,
            count=count,
            weight=weight,
            n_point=n_point,
            n_patch=n_patch,
            error_method=method,
            min_count=min_count,
        )

    @classmethod
    def read(cls, path: str | Path) -> SFResult[FloatArray] | SFResult[u.Quantity]:
        """Read a result written by :meth:`write`."""
        return cls.from_table(Table.read(path))
