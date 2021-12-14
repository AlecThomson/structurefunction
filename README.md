# StructureFunction

Efficient computation of structure functions for astronomical data with errors.

## Structure functions

I follow the definitions laid out by [Haverkorn et al. 2004](https://ui.adsabs.harvard.edu/abs/2004ApJ...609..776H). Whilst structure functions can be computed for any value on a sparse grid, here I focus on rotation meaures (RM) from astronomical sources. As such, data points are distributed on a spherical surface.

The second-order structure function of RM is given by:

$$ SF_{\text{RM},\text{obs}}(\delta\theta) = \langle[\text{RM}{\theta} - \text{RM}(\theta+\delta\theta)]\rangle$$

That is, the ensemble average of the squared-difference in RM for sources with angular seperation $\delta\theta$. We also need to correct for the impact of errors by:

$$ SF_{\text{RM}}(\delta\theta) = SF_{\text{RM},\text{obs}}(\delta\theta) - SF_{\sigma_\text{RM}}(\delta\theta) $$

Computing the error on the structure is diffifcult. Here I use Monte-Carlo error propagation to compute the errors numerically.

## Installation

To get the latest version from this repo
```
pip install githttps://github.com/AlecThomson/structurefunction
```
Or, install from PyPi
```
pip install structurefunction
```

## Usage

See the notebook included in the examples. There I repoduce the results of [Mao et al. 2010](https://ui.adsabs.harvard.edu/abs/2010ApJ...714.1170M).