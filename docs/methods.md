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

In `binned` mode, the neighbours of each middle source are sorted into cells by
distance and direction, and each cell is combined with the cells on the
opposite side. This is much faster on large catalogues. It uses the same
tolerances but finds somewhat fewer triplets than `exact` mode.

## Speed

Sources are stored in a tree. For two well separated groups of sources, the sum
of $(\mathrm{RM}_i - \mathrm{RM}_j)^2$ over every pair between them follows
from each group's mean and spread, without visiting the pairs. A group pair is
handled this way when all its pairs fall in the same bin, or, with
`bin_tolerance > 0`, when their separations differ by less than that fraction
of the bin width.

## Errors

The sky is split into patches (k-means). The jackknife recomputes the structure
function with each patch left out in turn; the spread of those estimates gives
the covariance between bins. This includes the measurement errors and the
variation between different parts of the sky. Patches should be larger than the
separations of interest. The bootstrap option resamples patches instead.

## Fitting

Models are fitted using the full covariance. Because the covariance is itself
estimated from a finite number of patches, the default likelihood is that of
Sellentin & Heavens (2016).
