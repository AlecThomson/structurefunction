"""Package logger and warning category. Nothing is configured on import."""

from __future__ import annotations

import logging

logger = logging.getLogger("structurefunction")


class StructureFunctionWarning(UserWarning):
    """Warns about estimates that may be unreliable (e.g. too few regions)."""
