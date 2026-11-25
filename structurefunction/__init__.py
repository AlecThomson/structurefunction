"""Fast, robust structure functions of irregularly sampled sky data."""

from __future__ import annotations

from structurefunction._version import version as __version__
from structurefunction.core import structure_function, structure_function_from_table
from structurefunction.fitting import (
    FitPriors,
    FitResult,
    compare_models,
    evaluate,
    fit_sf,
)
from structurefunction.io import (
    read_sf,
    read_skymap,
    sf_from_table,
    skymap_from_table,
    to_table,
    write,
)
from structurefunction.logging import StructureFunctionWarning
from structurefunction.options import (
    AxisPositionErrors,
    CircularPositionError,
    StencilOptions,
    axis_position_errors_from_table,
    circular_position_error_from_table,
)
from structurefunction.results import SFResult
from structurefunction.skymap import (
    SkyMapResult,
    healpix_map,
    recommend_nside,
    structure_function_skymap,
)
from structurefunction.types import (
    CrossRegionWeight,
    ErrorMethod,
    FitModel,
    Likelihood,
    RAErrorConvention,
    Sampler,
    StencilMode,
    Taper,
)

__all__ = [
    "AxisPositionErrors",
    "CircularPositionError",
    "CrossRegionWeight",
    "ErrorMethod",
    "FitModel",
    "FitPriors",
    "FitResult",
    "Likelihood",
    "RAErrorConvention",
    "SFResult",
    "Sampler",
    "SkyMapResult",
    "StencilMode",
    "StencilOptions",
    "StructureFunctionWarning",
    "Taper",
    "__version__",
    "axis_position_errors_from_table",
    "circular_position_error_from_table",
    "compare_models",
    "evaluate",
    "fit_sf",
    "healpix_map",
    "read_sf",
    "read_skymap",
    "recommend_nside",
    "sf_from_table",
    "skymap_from_table",
    "structure_function",
    "structure_function_from_table",
    "structure_function_skymap",
    "to_table",
    "write",
]
