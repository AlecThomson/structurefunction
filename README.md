[![Documentation Status](https://readthedocs.org/projects/structurefunction/badge/?version=latest)](https://structurefunction.readthedocs.io/en/latest/?badge=latest)

# StructureFunction

Efficient computation of structure functions for astronomical data with errors.

## Structure functions

I follow the definitions laid out by [Haverkorn et al. 2004](https://ui.adsabs.harvard.edu/abs/2004ApJ...609..776H). Whilst structure functions can be computed for any value on a sparse grid, here I focus on rotation meaures (RM) from astronomical sources. As such, data points are distributed on a spherical surface.

The second-order structure function of RM is given by:

```math
SF_{\text{RM},\text{obs}}(\delta\theta) = \langle[\text{RM}{(\theta)} - \text{RM}{(\theta + \delta\theta)}]^2\rangle
```

That is, the ensemble average of the squared-difference in RM for sources with angular seperation $\delta\theta$. We also need to correct for the impact of errors by:

```math
SF_{\text{RM}}(\delta\theta) = SF_{\text{RM},\text{obs}}(\delta\theta) - SF_{\sigma_\text{RM}}(\delta\theta)
```

Computing the error on the structure function is difficult. Here I use Monte-Carlo error propagation to compute the errors numerically.

I provide the ability to fit a broken power-law to the data using both standard least-squares, and full-blown MCMC fitting powered by [bilby](https://lscsoft.docs.ligo.org/bilby/).

## Documentation

The documentation is available [here](https://structurefunction.readthedocs.io/en/latest/).

## Installation

To get the latest version from this repo
```
pip install git+https://github.com/AlecThomson/structurefunction
```
Or, install from PyPi
```
pip install structurefunction
```

## Usage

See the notebook included in the examples. There I repoduce the results of [Mao et al. 2010](https://ui.adsabs.harvard.edu/abs/2010ApJ...714.1170M).

## Acknowledgment
If you use this software in your research, please cite the associated CSIRO software collection:
> Thomson, Alec (2024): StructureFunction. v1. CSIRO. Software Collection. https://doi.org/10.25919/dkwn-mg50

## License
MIT.
