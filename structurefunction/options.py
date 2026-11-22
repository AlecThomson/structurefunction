"""Validated, immutable option containers (pydantic, via capn-crunch).

Validation lives in the field types: ranges in ``Field``, and string modes are
turned into their enums by an ``AfterValidator``. Option classes also generate
command-line arguments with ``capn_crunch.add_options_to_parser``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated

from astropy.units import Quantity
from capn_crunch import BaseOptions
from pydantic import AfterValidator, Field

from structurefunction.types import (
    RAErrorConvention,
    RAErrorConventionLike,
    StencilMode,
    StencilModeLike,
)

if TYPE_CHECKING:
    from astropy.table import Table


def _stencil_mode(value: StencilModeLike) -> StencilMode:
    return StencilMode(value)


def _ra_convention(value: RAErrorConventionLike) -> RAErrorConvention:
    return RAErrorConvention(value)


class StencilOptions(BaseOptions):
    """How three-point stencils are gathered on irregular data.

    A triplet is accepted when the outer sources are within ``angle_tol_deg``
    of a straight line through the middle source, and their distances from it
    agree to within ``sep_tol``.
    """

    angle_tol_deg: Annotated[float, Field(gt=0.0, lt=90.0)] = 10.0
    """Allowed deviation from a straight line, in degrees."""
    sep_tol: Annotated[float, Field(gt=0.0, lt=1.0)] = 0.1
    """Allowed fractional mismatch of the two arm lengths, ``|d+ - d-| / mean``."""
    mode: Annotated[StencilModeLike, AfterValidator(_stencil_mode)] = StencilMode.BINNED
    """``exact`` finds every triplet; ``binned`` sums cells and is faster."""


class CircularPositionError(BaseOptions):
    """A circular (radial) 1-sigma positional error per source."""

    sigma: Quantity
    """Angular 1-sigma error, scalar or one per source."""


class AxisPositionErrors(BaseOptions):
    """Independent 1-sigma positional errors in longitude (RA) and latitude (Dec)."""

    sigma_ra: Quantity
    """Longitude (RA) error, angular or time units (e.g. ``s`` of RA)."""
    sigma_dec: Quantity
    """Latitude (Dec) error, angular units."""
    ra_convention: Annotated[RAErrorConventionLike, AfterValidator(_ra_convention)] = (
        RAErrorConvention.ON_SKY
    )
    """Whether ``sigma_ra`` already includes ``cos(dec)``."""
    corr: Annotated[float, Field(ge=-1.0, le=1.0)] | None = None
    """RA/Dec error correlation coefficient, or ``None`` for uncorrelated."""


type PositionErrors = CircularPositionError | AxisPositionErrors


def circular_position_error_from_table(
    table: Table, column: str
) -> CircularPositionError:
    """Circular positional errors from a table column with angular units."""
    return CircularPositionError(sigma=table[column].quantity)


def axis_position_errors_from_table(
    table: Table,
    ra_column: str,
    dec_column: str,
    ra_convention: RAErrorConventionLike = RAErrorConvention.ON_SKY,
) -> AxisPositionErrors:
    """RA and Dec positional errors from two table columns with units."""
    return AxisPositionErrors(
        sigma_ra=table[ra_column].quantity,
        sigma_dec=table[dec_column].quantity,
        ra_convention=ra_convention,
    )
