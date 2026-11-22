"""NamedTuples that group the arrays passed into the numba kernels."""

from __future__ import annotations

from enum import IntEnum
from typing import NamedTuple

from structurefunction.types import FloatArray, IntArray


class Moment(IntEnum):
    """Columns of ``Nodes.moments``."""

    WEIGHT = 0
    """Sum of weights."""
    MEAN = 1
    """Weighted mean value."""
    SQUARED_DEVIATION = 2
    """Weighted sum of squared deviations from ``MEAN``."""
    ERROR_VAR = 3
    """Weighted sum of error variances."""
    COUNT = 4
    """Number of sources."""
    MAX_POSITION_VAR = 5
    """Largest positional variance of any source, in any direction."""


class Nodes(NamedTuple):
    """Ball-tree nodes. The two children of a node are stored consecutively."""

    start: IntArray
    """First source (tree order) in each node."""
    end: IntArray
    """One past the last source in each node."""
    left: IntArray
    """Left child, or -1 for a leaf. The right child is ``left + 1``."""
    centre: FloatArray
    """Mean position, shape ``(n_node, 3)``."""
    radius: FloatArray
    """Largest distance from the centre to a member."""
    moments: FloatArray
    """Value moments, columns as in :class:`Moment`."""
    patch: IntArray
    """Patch of each node (nodes never span patches)."""


class Sources(NamedTuple):
    """Per-source arrays, in tree order."""

    positions: FloatArray
    """Unit vectors, or 3D positions with distances, shape ``(n, 3)``."""
    values: FloatArray
    error_var: FloatArray
    """Measurement error variances."""
    weights: FloatArray
    patch: IntArray
    east: FloatArray
    """Unit east vector at each source (angular data only)."""
    north: FloatArray
    """Unit north vector at each source (angular data only)."""
    position_cov: FloatArray
    """Positional covariance: east variance, north variance, covariance."""


class Binning(NamedTuple):
    """Separation bins and how pairs are placed in them."""

    edges: FloatArray
    angular: bool
    """Separations are great-circle angles (else 3D distances)."""
    tolerance: float
    """``bin_tolerance``: allowed misplacement as a fraction of a bin width."""
    use_position_errors: bool


class PolarGrid(NamedTuple):
    """Cells around a middle source for the binned three-point stencil."""

    ring_edges: FloatArray
    """Distance edges of the rings, nested inside the output bins."""
    ring_to_bin: IntArray
    """Output bin of each ring."""
    ring_mid: FloatArray
    """Geometric centre of each ring."""
    n_sector: int
    """Number of bearing sectors (even)."""
