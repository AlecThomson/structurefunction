# Usage

```python
import astropy.units as u
from astropy.coordinates import SkyCoord

from structurefunction import StencilOptions, structure_function
from structurefunction.fitting import fit_sf

coords = SkyCoord(ra, dec)
sf2 = structure_function(rm, rm_err, coords, bins=12, patches=64)
sf3 = structure_function(rm, rm_err, coords, n_point=3, bins=12, patches=64)
fit = fit_sf(sf2, model="broken_power_law", sampler="emcee")
sf2.write("sf2.ecsv")
```

`values` and `errors` may be plain arrays or `Quantity` (the structure function
then carries `unit**2`). `coords` may be in any frame; with distances, 3D
separations are used.

From the command line:

```bash
structurefunction sf catalogue.fits sf.ecsv --value rm --error rm_err --n-point 3 --plot sf.png
```

## Options worth knowing

- `bin_tolerance`: largest fraction of a bin width by which tree-grouped pairs
  may be misplaced. `0` is exact.
- `patches`: number of k-means sky patches (or labels). Use at least three
  times the number of bins, and patches larger than the separations you care
  about.
- `pos_errors`: `CircularPositionError` or `AxisPositionErrors`. Pairs are
  softly binned using each pair's separation uncertainty.
- `stencil`: `StencilOptions(angle_tol_deg, sep_tol, mode)` for three-point
  stencils. `exact` enumerates triplets; `binned` aggregates around each middle
  source and scales to large catalogues.

## Sky maps

`structure_function_skymap` computes an apodised structure function for every
HEALPix cell. Sources around each cell centre are weighted by a cosine or
Gaussian taper (times any user `weights`), so each pair is weighted by the
product of its members' taper values. With `fit_model`, a model is fitted per
cell and its parameters become maps.

```python
import astropy.units as u

from structurefunction import structure_function_skymap
from structurefunction.plotting import plot_skymap

sky = structure_function_skymap(
    rm,
    rm_err,
    coords.galactic,
    nside=8,
    radius=10 * u.deg,
    taper="gaussian",
    bins=8,
    fit_model="broken_power_law",
)
plot_skymap(sky, "slope_1")
slope_map = sky.map("slope_1")  # full-sky HEALPix array, NaN where skipped
sky.write("skymap.fits")
```

```bash
structurefunction skymap catalogue.fits skymap.fits --value rm --error rm_err --nside 8 --radius-deg 10 --plot slope.png
```

The grid uses the frame of `coords`, so pass Galactic coordinates for a
Galactic map. Cells with fewer than `min_sources` effective sources
(`(sum w)^2 / sum w^2`) are skipped.
