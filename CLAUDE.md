# structurefunction

- Use `uv run` for everything. Gate before committing:
  `uv run prek run --all-files` and `uv run pytest -n auto`.
- Kernels must also pass with JIT off:
  `NUMBA_DISABLE_JIT=1 uv run pytest -m "not slow and not slow_uncompiled" tests/test_pairs.py tests/test_triplets.py tests/test_errors.py tests/test_position_errors.py`.
- Google docstrings, terse. No em/en dashes.
- No bare `*`, `*args` or `**kwargs` in signatures. No module-level constants
  unless genuinely global (logger, `__all__`, type aliases). No import-time side
  effects.
- Modes are `StrEnum`s in `types.py`, each with a `*Like` alias accepting the
  string; convert once at the boundary.
- astropy (`Quantity`, `Unit`, `SkyCoord`) is handled only in `_units.py`;
  everything downstream is `NDArray[np.float64]`.
- Kernel accumulators: `(slot, Kind, bin, patch, Stat)`; see
  `_kernels/common.py`. New kernels must credit AUTO/CROSS/MARK so the jackknife
  and bootstrap stay single-pass.
- Every kernel change needs a comparison against `tests/reference.py` (numpy,
  obviously correct) at `bin_tolerance=0`.
- Kernel arrays travel in the NamedTuples in `_kernels/containers.py`. In hot
  loops, index rows directly (`positions[i, 0]`) rather than slicing
  (`positions[i]`): numba builds a new array object for every slice.
- `inline="always"` is for pure helpers and the accumulation path only. numba
  has miscompiled inlined helpers that mutate arrays after an early return
  (binned triplets silently lost most triplets); write such helpers without
  early returns and check them with `NUMBA_DISABLE_JIT=1` as well.
- numba's on-disk cache does not notice changes in functions called from other
  modules; clear `__pycache__/*.nbi|*.nbc` if results look stale.

## Type checkers

mypy (strict) is the authoritative gate. ty is pinned exactly in `pyproject.toml`
(beta: rules change between releases; bump deliberately). If ty false positives
spread beyond `_units.py`, make its CI step advisory rather than bending code.
