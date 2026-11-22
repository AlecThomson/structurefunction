# Methods

## Structure functions

For a stencil with coefficients $c_i$ (two-point: $[1, -1]$; three-point:
$[1, -2, 1]$, Seta et al. 2022) and source weights $w_i$,

$$
\mathrm{SF}(r) = \frac{\sum w \,(\sum_i c_i \mathrm{RM}_i)^2}{\sum w}
 - \frac{\sum w \,\sum_i c_i^2 \sigma_i^2}{\sum w},
$$

where the outer sums run over the pairs or triplets in a separation bin and $w$
is the product of the weights of the sources involved. The second term removes
the contribution of the measurement errors. At separations well beyond the
correlation length, the two-point structure function tends to twice the variance
of the RMs (with the error variance removed), and the three-point one to six
times it.

## Three-point structure functions on irregular data

A triplet $(x_-, x_0, x_+)$ is used when $x_-$ and $x_+$ lie on opposite sides
of $x_0$, within `angle_tol_deg` of a straight line (great circle) through it,
and their distances from $x_0$ agree to within a fraction `sep_tol`. The
triplet's separation is the mean of the two distances.

In `binned` mode, the neighbours of each middle source are summed into cells by
distance and direction. Cells are half a tolerance wide, and each is combined
with the opposite cell and its neighbours, so every triplet used is within the
tolerances. This is much faster on large catalogues but finds roughly half the
triplets of `exact` mode.

### Choosing the tolerances

Looser tolerances use more triplets, but admit ones that are slightly bent or
lopsided, which lets the large-scale gradient leak in and flattens steep
structure functions. `benchmarks/stencil_tolerances.py` measures this on
Gaussian random fields with power spectra $k^{-\beta}$, sampled at 30 000
random positions, fitting the slope between 0.1 and 0.5 degrees. Below is the
slope change relative to the tightest setting (2 degrees, 0.02), averaged over
three fields, for the steepest case $\beta = 4$ (true slope 3):

| `angle_tol_deg` | `sep_tol` | exact | binned |
| --------------: | --------: | ----: | -----: |
|               2 |      0.02 |  0.00 |   0.00 |
|               5 |      0.05 |  0.00 |  -0.01 |
|               5 |      0.10 | -0.01 |  -0.01 |
|              10 |      0.10 | -0.04 |  -0.03 |
|              10 |      0.20 | -0.08 |  -0.05 |
|              15 |      0.10 | -0.09 |  -0.06 |
|              20 |      0.20 | -0.18 |  -0.12 |

For $\beta = 2$ and $3$ every setting tested is within 0.02. The defaults
(10 degrees, 0.1) keep the bias below 0.05 while using about 25 times as many
triplets as the tightest setting. Tighten them for very steep spectra with
plenty of sources.

## Speed

Sources are stored in a tree, one subtree per patch. For two well separated
groups of sources, the sum of $(\mathrm{RM}_i - \mathrm{RM}_j)^2$ over every pair
between them follows from each group's mean and spread, without visiting the
pairs. A group pair is handled this way when all its pairs fall in the same bin.
With `bin_tolerance > 0`, it is also handled when the spread of its separations
is less than that fraction of the bin width; it is then shared between the bins
it overlaps, assuming the separations follow a triangular distribution between
the closest and farthest possible values.

When the upper edge is left at its default (the largest separation in the
catalogue), the top bin is treated as open, since no pair can lie beyond it.

## Errors

The sky is split into patches (k-means). The jackknife recomputes the structure
function with each patch left out in turn; the spread of those estimates gives
the covariance between bins. This includes the measurement errors and the
variation between different parts of the sky. It misses variance on scales
larger than a patch, so errors of bins beyond the typical patch size
(`SFResult.patch_size`, `SFResult.resampling_reliable`) are underestimated. The
bootstrap option resamples patches instead.

## Fitting and choosing a model

Models are fitted using the full covariance. Because the covariance is itself
estimated from a finite number of patches, the default likelihood is that of
Sellentin & Heavens (2016). Slopes are limited to be non-negative by default,
since a structure function does not fall with separation.

`compare_models` fits a constant (no structure on the scales fitted), a power
law and broken power laws, and ranks them by BIC. With `sampler="nautilus"` it
also gives the Bayesian evidence. A BIC difference above about 6, or an
evidence ratio $\Delta \ln Z$ above about 3, is strong support for the better
model.

## Sky maps

`structure_function_skymap` weights the sources around each HEALPix cell centre
by a cosine or Gaussian taper of scale `radius` and computes the structure
function with those weights. With `nside="auto"`, `recommend_nside` chooses the
coarsest grid whose pixels are at most half the taper's half-power width
(`radius` for both tapers), so neighbouring cells overlap and the map is fully
sampled.
