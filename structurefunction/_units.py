"""The single boundary between astropy objects and plain float64 arrays.

Everything downstream of this module works on ``NDArray[np.float64]``. Keeping
astropy here also keeps static-checker friction with dynamically generated
units (``u.deg`` etc.) in one place.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, assert_never

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord, UnitSphericalRepresentation

from structurefunction.options import (
    AxisPositionErrors,
    CircularPositionError,
    PositionErrors,
)
from structurefunction.typedefs import RAErrorConvention

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

    from structurefunction.typedefs import FloatArray


@dataclass(frozen=True, slots=True)
class Values:
    """Unitless values, errors and weights plus the unit they came with."""

    values: FloatArray
    errors: FloatArray
    weights: FloatArray
    unit: u.UnitBase | None


@dataclass(frozen=True, slots=True)
class Positions:
    """Source positions in the internal metric.

    Angular data are unit vectors (separations in radians); data with distances
    are 3D Cartesian (separations in ``unit``).
    """

    xyz: FloatArray
    lon: FloatArray
    lat: FloatArray
    angular: bool
    unit: u.UnitBase


def _is_quantity(x: object) -> bool:
    return isinstance(x, u.Quantity)


def split_values(
    values: ArrayLike | u.Quantity,
    errors: ArrayLike | u.Quantity | None,
    weights: ArrayLike | None,
) -> Values:
    """Strip units, broadcast and validate values, errors and weights.

    Args:
        values: Measured values, plain or Quantity.
        errors: 1-sigma errors, matching ``values`` in having units or not.
            ``None`` means noise-free (no bias subtraction).
        weights: Optional non-negative per-source weights (dimensionless).

    Returns:
        The unitless arrays and the unit of ``values`` (or ``None``).

    Raises:
        TypeError: If ``values`` and ``errors`` mix plain arrays and Quantities.
        ValueError: On shape mismatch, non-finite values or negative weights.
    """
    unit: u.UnitBase | None = None
    if _is_quantity(values):
        q = u.Quantity(values)
        unit = q.unit
        f = np.asarray(q.value, dtype=np.float64)
    else:
        f = np.asarray(values, dtype=np.float64)
    if f.ndim != 1:
        msg = f"values must be 1D, got shape {f.shape}"
        raise ValueError(msg)

    if errors is None:
        sigma = np.zeros_like(f)
    elif _is_quantity(errors) != (unit is not None):
        msg = "values and errors must both be Quantities or both be plain arrays"
        raise TypeError(msg)
    elif unit is not None:
        sigma = np.asarray(u.Quantity(errors).to_value(unit), dtype=np.float64)
    else:
        sigma = np.asarray(errors, dtype=np.float64)
    sigma = np.broadcast_to(sigma, f.shape).astype(np.float64)

    w = (
        np.ones_like(f)
        if weights is None
        else np.broadcast_to(np.asarray(weights, dtype=np.float64), f.shape).astype(
            np.float64
        )
    )

    if not (np.all(np.isfinite(f)) and np.all(np.isfinite(sigma))):
        msg = "values and errors must be finite; drop or mask bad sources first"
        raise ValueError(msg)
    if np.any(sigma < 0) or np.any(w < 0) or not np.all(np.isfinite(w)):
        msg = "errors and weights must be finite and non-negative"
        raise ValueError(msg)
    return Values(values=f, errors=sigma, weights=w, unit=unit)


def attach_unit(
    array: FloatArray, unit: u.UnitBase | None, power: int = 1
) -> FloatArray | u.Quantity:
    """Re-attach ``unit**power`` to an array, or return it unchanged."""
    if unit is None:
        return array
    return u.Quantity(array, unit**power)


def positions(coords: SkyCoord) -> Positions:
    """Convert a SkyCoord in any frame to internal positions.

    Coordinates with a distance give 3D Cartesian positions; otherwise unit
    vectors on the sphere. Longitude/latitude are those of ``coords``' own frame.
    """
    sph = coords.spherical
    lon = np.asarray(sph.lon.to_value(u.rad), dtype=np.float64).ravel()
    lat = np.asarray(sph.lat.to_value(u.rad), dtype=np.float64).ravel()
    has_distance = not isinstance(coords.data, UnitSphericalRepresentation)
    if has_distance:
        dist = sph.distance
        unit = dist.unit
        r = np.asarray(dist.to_value(unit), dtype=np.float64).ravel()
        angular = False
    else:
        unit = u.rad
        r = np.ones_like(lon)
        angular = True
    cos_lat = np.cos(lat)
    xyz = np.column_stack(
        (r * cos_lat * np.cos(lon), r * cos_lat * np.sin(lon), r * np.sin(lat))
    )
    return Positions(xyz=xyz, lon=lon, lat=lat, angular=angular, unit=unit)


def separation_value(sep: u.Quantity | float, pos: Positions) -> float:
    """Express a separation in the internal unit (radians or distance unit)."""
    if _is_quantity(sep):
        return float(u.Quantity(sep).to_value(pos.unit))
    return float(sep)


def separation_array(seps: u.Quantity | ArrayLike, pos: Positions) -> FloatArray:
    """Express separations (e.g. bin edges) in the internal unit."""
    if _is_quantity(seps):
        return np.asarray(u.Quantity(seps).to_value(pos.unit), dtype=np.float64)
    return np.asarray(seps, dtype=np.float64)


def separation_unit(pos: Positions) -> u.UnitBase:
    """Unit reported for separations: degrees for angular data."""
    return u.deg if pos.angular else pos.unit


def to_reported_separation(seps: FloatArray, pos: Positions) -> u.Quantity:
    """Convert internal separations to the reported unit."""
    return u.Quantity(seps, pos.unit).to(separation_unit(pos))


def position_covariance(
    pos_errors: PositionErrors, pos: Positions
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Per-source tangent-plane positional covariance in radians squared.

    Returns:
        ``(var_east, var_north, cov_east_north)`` arrays, one entry per source.

    Raises:
        NotImplementedError: For coordinates with distances.
    """
    if not pos.angular:
        msg = "positional errors are only supported for angular (2D sky) data"
        raise NotImplementedError(msg)
    n = pos.lon.size
    if isinstance(pos_errors, CircularPositionError):
        s = np.broadcast_to(
            np.asarray(pos_errors.sigma.to_value(u.rad), dtype=np.float64), (n,)
        )
        var = s**2
        return var.copy(), var.copy(), np.zeros(n)
    if isinstance(pos_errors, AxisPositionErrors):  # pragma: no branch
        ra = pos_errors.sigma_ra
        # Time units of RA (h, min, s) convert to angle at 15 deg per hour.
        if ra.unit.physical_type == "time":
            ra_rad = (ra.to_value(u.hour) * 15.0 * np.pi / 180.0).astype(np.float64)
        else:
            ra_rad = np.asarray(ra.to_value(u.rad), dtype=np.float64)
        ra_rad = np.broadcast_to(ra_rad, (n,)).astype(np.float64)
        if pos_errors.ra_convention == RAErrorConvention.COORDINATE:
            ra_rad = ra_rad * np.cos(pos.lat)
        dec_rad = np.broadcast_to(
            np.asarray(pos_errors.sigma_dec.to_value(u.rad), dtype=np.float64), (n,)
        ).astype(np.float64)
        rho = 0.0 if pos_errors.corr is None else pos_errors.corr
        return ra_rad**2, dec_rad**2, rho * ra_rad * dec_rad
    assert_never(pos_errors)
