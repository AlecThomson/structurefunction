[![Documentation Status](https://img.shields.io/readthedocs/structurefunction/latest)](https://structurefunction.readthedocs.io/en/latest/)
[![DOI](https://img.shields.io/badge/doi-10.25919/dkwn--mg50-blue.svg)](https://doi.org/10.25919/dkwn-mg50)

# structurefunction

Second-order structure functions of irregularly sampled sky data, such as
Faraday rotation measures of background sources.

- Two-point and three-point stencils
  ([Seta et al. 2023](https://arxiv.org/abs/2206.13798)).
- Exact noise-bias subtraction, per pair or triplet.
- Covariance between bins from a jackknife or bootstrap over sky regions.
- Optional positional errors (circular, or separate RA/Dec).
- Tree-based numba kernels that scale to millions of sources.
- Power-law and broken-power-law fits with the full covariance, and model
  comparison by BIC and Bayesian evidence.
- Sky maps: tapered structure functions and fitted parameters per HEALPix cell.
- Works with plain arrays or astropy `Quantity`, any `SkyCoord` frame, and
  astropy tables.

## Installation

```bash
pip install structurefunction
```

Nested-sampling evidence (nautilus) needs the `evidence` extra.

## Usage

```python
from astropy.coordinates import SkyCoord

from structurefunction import structure_function
from structurefunction.fitting import fit_sf

sf = structure_function(rm, rm_err, SkyCoord(ra, dec), n_point=3, bins=12)
fit = fit_sf(sf, model="broken_power_law")
```

See the [documentation](https://structurefunction.readthedocs.io/) for the
methods and a worked example reproducing
[Mao et al. 2010](https://ui.adsabs.harvard.edu/abs/2010ApJ...714.1170M).

Version 2 is a rewrite and is not backwards compatible with 1.x; see the
[migration guide](https://structurefunction.readthedocs.io/en/latest/examples/migration.html).
A [comparison with TreeCorr](https://structurefunction.readthedocs.io/en/latest/examples/treecorr_comparison.html)
shows where each is faster and what each offers.

## Acknowledgment

If you use this software in your research, please cite the associated CSIRO
software collection:

> Thomson, Alec (2024): StructureFunction. v1. CSIRO. Software Collection.
> https://doi.org/10.25919/dkwn-mg50

## License

MIT.
