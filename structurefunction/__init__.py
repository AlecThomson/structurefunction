"""Fast, robust structure functions of irregularly sampled sky data."""

from __future__ import annotations

from structurefunction._version import version as __version__
from structurefunction.core import structure_function, structure_function_from_table
from structurefunction.logging import StructureFunctionWarning
from structurefunction.options import (
    AxisPositionErrors,
    CircularPositionError,
    StencilOptions,
)
from structurefunction.results import SFResult
from structurefunction.types import (
    CrossPatchWeight,
    ErrorMethod,
    FitModel,
    Likelihood,
    RAErrorConvention,
    Sampler,
    StencilMode,
)

__all__ = [
    "AxisPositionErrors",
    "CircularPositionError",
    "CrossPatchWeight",
    "ErrorMethod",
    "FitModel",
    "Likelihood",
    "RAErrorConvention",
    "SFResult",
    "Sampler",
    "StencilMode",
    "StencilOptions",
    "StructureFunctionWarning",
    "__version__",
    "structure_function",
    "structure_function_from_table",
]
