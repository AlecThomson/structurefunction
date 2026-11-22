"""Frozen option and input containers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from structurefunction.types import (
    RAErrorConvention,
    RAErrorConventionLike,
    StencilMode,
    StencilModeLike,
)

if TYPE_CHECKING:
    from astropy.table import Table
    from astropy.units import Quantity


@dataclass(frozen=True, slots=True)
class StencilOptions:
    """Options controlling how multi-point stencils are gathered on irregular data.

    Seta et al. (2023) define stencils on regular grids. On irregular data we
    accept a triplet when the outer points lie on opposite sides of the middle
    point within ``angle_tol`` of exact collinearity, and their separations from
    the middle point agree within ``sep_tol`` (fractional).
    """

    angle_tol_deg: float = 10.0
    """Allowed deviation from collinearity, in degrees (tangent-plane bearing)."""
    sep_tol: float = 0.1
    """Allowed fractional mismatch of the two arm lengths, ``|d+ - d-| / mean``."""
    mode: StencilModeLike = StencilMode.BINNED
    """``exact`` triplet search, or ``binned`` polar-grid moment aggregation."""

    def __post_init__(self) -> None:
        """Validate tolerances and normalise the mode."""
        if not 0.0 < self.angle_tol_deg < 90.0:  # noqa: PLR2004
            msg = f"angle_tol_deg must be in (0, 90), got {self.angle_tol_deg}"
            raise ValueError(msg)
        if not 0.0 < self.sep_tol < 1.0:
            msg = f"sep_tol must be in (0, 1), got {self.sep_tol}"
            raise ValueError(msg)
        object.__setattr__(self, "mode", StencilMode(self.mode))


@dataclass(frozen=True, slots=True)
class CircularPositionError:
    """A circular (radial) 1-sigma positional error per source.

    Args:
        sigma: Angular error, scalar or one per source.
    """

    sigma: Quantity

    @classmethod
    def from_table(cls, table: Table, column: str) -> CircularPositionError:
        """Build from a table column carrying angular units."""
        return cls(sigma=table[column].quantity)


@dataclass(frozen=True, slots=True)
class AxisPositionErrors:
    """Independent 1-sigma positional errors in longitude (RA) and latitude (Dec).

    Args:
        sigma_ra: Longitude error. Angular or time units (e.g. ``s`` of RA).
        sigma_dec: Latitude error, angular units.
        ra_convention: Whether ``sigma_ra`` already includes ``cos(dec)``.
        corr: Optional RA/Dec error correlation coefficient(s), in [-1, 1].
    """

    sigma_ra: Quantity
    sigma_dec: Quantity
    ra_convention: RAErrorConventionLike = RAErrorConvention.ON_SKY
    corr: float | None = field(default=None)

    def __post_init__(self) -> None:
        """Normalise the RA convention."""
        object.__setattr__(
            self, "ra_convention", RAErrorConvention(self.ra_convention)
        )

    @classmethod
    def from_table(
        cls,
        table: Table,
        ra_column: str,
        dec_column: str,
        ra_convention: RAErrorConventionLike = RAErrorConvention.ON_SKY,
    ) -> AxisPositionErrors:
        """Build from two table columns carrying units."""
        return cls(
            sigma_ra=table[ra_column].quantity,
            sigma_dec=table[dec_column].quantity,
            ra_convention=ra_convention,
        )


type PositionErrors = CircularPositionError | AxisPositionErrors
