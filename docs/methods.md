# Methods

## Estimators

For a stencil with coefficients $c_i$ (two-point: $[1, -1]$; three-point:
$[1, -2, 1]$, Seta et al. 2023) and source weights $w_i$,

$$
\mathrm{SF}(r) = \frac{\sum w \,(\sum_i c_i f_i)^2}{\sum w}
 - \frac{\sum w \,\sum_i c_i^2 \sigma_i^2}{\sum w},
$$

where the outer sums run over pairs or triplets in the separation bin and
$w = \prod_i w_i$. The second term is the noise bias; it is exact for
independent zero-mean noise. For uncorrelated data the structure function
plateaus at $\sum_i c_i^2$ times the intrinsic variance: 2 for two points,
6 for three.

## Three-point stencils on irregular data

A triplet $(x_-, x_0, x_+)$ is accepted when the bearings of $x_\pm$ from $x_0$,
measured in the tangent plane at $x_0$ (where great circles through $x_0$ are
straight lines), differ from $\pi$ by at most `angle_tol_deg`, and the arm
lengths satisfy $|d_+ - d_-| \le$ `sep_tol` $\bar d$. The separation is
$\bar d$.

`binned` mode places neighbours in a polar grid of bearing sectors and radial
sub-bins of ratio $1 + $ `sep_tol` around each middle source, and pairs each
cell with the opposite sector in the same sub-bin. Every accepted triplet
satisfies the same tolerances as in `exact` mode, but triplets straddling cell
boundaries are not counted, so it uses fewer triplets in exchange for cost
linear in the number of neighbours.

## Tree aggregation

Sources are held in a ball tree with one subtree per patch. Each node stores
its weighted mean $\mu$, sum of squared deviations $M_2$ and noise sum
$S = \sum w\sigma^2$. For disjoint nodes $A$, $B$,

$$
\sum_{i\in A, j\in B} w_i w_j (f_i - f_j)^2
= W_B M_{2,A} + W_A M_{2,B} + W_A W_B (\mu_A - \mu_B)^2,
$$

with every term non-negative. A node pair is added as a block when all its
pairs fall in one bin (exact), or when its spread of separations is within
`bin_tolerance` of the bin width.

## Covariance

Each pair or triplet is credited to the patches its members occupy. Deleting a
patch removes everything touching it, which gives the delete-one jackknife in a
single pass. The marked bootstrap resamples patches with each contribution
split by member fraction.

## Fitting

Models are fitted with the full covariance. Because the covariance is estimated
from $n_s$ patches, the default likelihood is that of Sellentin & Heavens
(2016), $\ln L = -\tfrac{n_s}{2}\ln\left(1 + \chi^2/(n_s - 1)\right)$.
