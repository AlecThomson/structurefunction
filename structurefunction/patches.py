"""Spatial patches for jackknife and bootstrap resampling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy.cluster.vq import kmeans2

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

    from structurefunction.types import FloatArray, IntArray


@dataclass(frozen=True, slots=True)
class Patches:
    """Patch labels and summary geometry.

    Attributes:
        labels: Patch index of each source, in ``[0, n_patch)``.
        n_patch: Number of patches.
        size: Typical patch diameter (internal separation units): twice the
            median over patches of the RMS member distance from the centre.
    """

    labels: IntArray
    n_patch: int
    size: float


def make_patches(
    xyz: FloatArray,
    patches: int | ArrayLike,
    rng: np.random.Generator,
) -> Patches:
    """K-means patches on positions, or validate user-supplied labels.

    Args:
        xyz: Internal positions, shape ``(n, 3)``.
        patches: Number of patches, or an integer label per source.
        rng: Random generator used to seed k-means.

    Returns:
        The patch labels (relabelled to be contiguous) and their size.
    """
    n = xyz.shape[0]
    if np.ndim(patches) == 0:
        k = int(np.asarray(patches))
        if k < 1:
            msg = f"number of patches must be >= 1, got {k}"
            raise ValueError(msg)
        k = min(k, n)
        if k == 1:
            labels = np.zeros(n, dtype=np.intp)
        else:
            seed = int(rng.integers(2**31 - 1))
            _, raw = kmeans2(xyz, k, minit="++", seed=seed)
            labels = np.asarray(raw, dtype=np.intp)
    else:
        labels = np.asarray(patches, dtype=np.intp)
        if labels.shape != (n,):
            msg = f"patch labels must have shape ({n},), got {labels.shape}"
            raise ValueError(msg)
    # Relabel to 0..n_patch-1 (k-means can leave empty clusters).
    uniq, labels = np.unique(labels, return_inverse=True)
    labels = labels.astype(np.intp)
    n_patch = int(uniq.size)
    sizes = np.empty(n_patch)
    for p in range(n_patch):
        member = xyz[labels == p]
        centre = member.mean(axis=0)
        sizes[p] = 2.0 * np.sqrt(np.mean(np.sum((member - centre) ** 2, axis=1)))
    return Patches(labels=labels, n_patch=n_patch, size=float(np.median(sizes)))
