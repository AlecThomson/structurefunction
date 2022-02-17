from typing import Tuple
import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import itertools
import warnings
import bilby
from sigfig import round
import corner

warnings.filterwarnings("ignore", message="All-NaN slice encountered")
warnings.filterwarnings("ignore", message="All-NaN axis encountered")
warnings.filterwarnings("ignore", message="Mean of empty slice")


def model(x, amplitude, x_break, alpha_1, alpha_2):
    alpha = np.where(x < x_break, alpha_1, alpha_2)
    xx = x / x_break
    return amplitude * np.power(xx, -alpha)


def structure_function(
    data: u.Quantity,
    errors: u.Quantity,
    coords: SkyCoord,
    samples: int,
    bins: u.Quantity,
    show_plots=False,
    verbose=False,
    fit=False,
    outdir=None,
    **kwargs,
) -> Tuple[u.Quantity, u.Quantity, Tuple[u.Quantity, u.Quantity], np.ndarray]:

    """Compute the second order structure function with Monte-Carlo error propagation.

    Args:
        data (u.Quantity): 1D array of data values.
        errors (u.Quantity): 1D array of errors.
        coords (SkyCoord): 1D array of coordinates.
        samples (int): Number of samples to use for Monte-Carlo error propagation.
        bins (u.Quantity): Bin edges of the structure function.
        show_plots (bool, optional): Show plots. Defaults to False.
        verbose (bool, optional): Print progress. Defaults to False.
        fit (bool, optional): Fit the structure function. Defaults to False.
        outdir (str, optional): Output directory for bilby. Defaults to None.
        **kwargs: Additional keyword arguments to pass to the bilby.core.run_sampler function.

    Returns:
        Tuple[u.Quantity, u.Quantity, Tuple[u.Quantity, u.Quantity], np.ndarray]: 
            cbins: center of bins.
            medians: median of the structure function.
            errors: upper and lower error of the structure function.
            counts: number of source pairs in each bin.
    """

    # Sample the errors assuming a Gaussian distribution
    if verbose:
        print("Sampling errors...")
    rm_dist = []
    d_rm_dist = []
    for i in tqdm(range(data.shape[0]), "Sample Gaussian", disable=not verbose):
        rm_dist.append(np.random.normal(loc=data[i], scale=errors[i], size=samples))
        d_rm_dist.append(np.random.normal(loc=errors[i], scale=errors[i], size=samples))
    rm_dist = np.array(rm_dist)
    d_rm_dist = np.array(d_rm_dist)

    # Get all combinations of sources and compute the difference
    if verbose:
        print("Getting data differences...")
    F_dist = np.array(list(itertools.combinations(rm_dist, r=2)))

    diffs_dist = ((F_dist[:, 0] - F_dist[:, 1]) ** 2).T

    # Get all combinations of data_errs sources and compute the difference
    if verbose:
        print("Getting data error differences...")
    dF_dist = np.array(list(itertools.combinations(d_rm_dist, r=2)))

    d_diffs_dist = ((dF_dist[:, 0] - dF_dist[:, 1]) ** 2).T

    # Get the angular separation of the source paris
    if verbose:
        print("Getting angular separations...")
    cx_ra_perm, cy_ra_perm = np.array(
        list(itertools.combinations(coords.ra.to(u.deg).value, r=2))
    ).T
    cx_dec_perm, cy_dec_perm = np.array(
        list(itertools.combinations(coords.dec.to(u.deg).value, r=2))
    ).T
    coords_x = SkyCoord(cx_ra_perm * u.deg, cx_dec_perm * u.deg)
    coords_y = SkyCoord(cy_ra_perm * u.deg, cy_dec_perm * u.deg)
    dtheta = coords_x.separation(coords_y)
    dtheta = coords_x.separation(coords_y)

    # Compute the SF
    if verbose:
        print("Computing SF...")
    sf_dists = np.zeros((len(bins) - 1, samples)) * np.nan
    d_sf_dists = np.zeros((len(bins) - 1, samples)) * np.nan
    count = np.zeros((len(bins) - 1)) * np.nan
    cbins = np.zeros((len(bins) - 1)) * np.nan * u.deg
    for i, b in enumerate(tqdm(bins[:-1], disable=not verbose)):
        bin_idx = (bins[i] <= dtheta) & (dtheta < bins[i + 1])
        centre = (bins[i] + bins[i + 1]) / 2

        cbins[i] = centre
        count[i] = np.sum(bin_idx)
        sf_dist = np.nanmean(diffs_dist[:, bin_idx], axis=1)
        d_sf_dist = np.nanmean(d_diffs_dist[:, bin_idx], axis=1)

        sf_dists[i] = sf_dist
        d_sf_dists[i] = d_sf_dist

    # Get the final SF correcting for the errors
    medians = np.nanmedian(sf_dists - d_sf_dists, axis=1)
    per16 = np.nanpercentile(sf_dists - d_sf_dists, 16, axis=1)
    per84 = np.nanpercentile(sf_dists - d_sf_dists, 84, axis=1)
    err_low = medians - per16
    err_high = per84 - medians
    err = [err_low.astype(float), err_high.astype(float)]

    if fit:
        if verbose:
            print("Fitting SF with a broken power law...")
        # A few simple setup steps
        label = "linear_regression"
        if outdir is None:
            outdir = "outdir"
        bilby.utils.check_directory_exists_and_if_not_mkdir(outdir)

        # initialize a linear model
        injection_parameters = dict(amplitude=1.0, x_break=1.0, alpha_1=1.0, alpha_2=1)
        # Only use bins with at least 10 sources
        cut = (
            (count >= 10)
            & np.isfinite(cbins)
            & np.isfinite(medians)
            & np.isfinite(err[0])
            & np.isfinite(err[1])
        )
        x = np.array(cbins[cut].value)
        y = medians[cut]
        y_err = (per84 - per16)[cut]
        likelihood = bilby.likelihood.GaussianLikelihood(x, y, model, y_err)
        priors = dict()
        priors["amplitude"] = bilby.core.prior.Uniform(
            y.min() - y_err.max(), y.max() + y_err.max(), "amplitude"
        )
        priors["x_break"] = bilby.core.prior.Uniform(x.min(), x.max(), "x_break")
        priors["alpha_1"] = bilby.core.prior.Uniform(-2, 2, "alpha_1")
        priors["alpha_2"] = bilby.core.prior.Uniform(-2, 2, "alpha_2")
        result = bilby.run_sampler(
            likelihood=likelihood,
            priors=priors,
            sample="unif",
            injection_parameters=injection_parameters,
            outdir=outdir,
            label=label,
            **kwargs,
        )
        if show_plots:
            samps = result.samples
            labels = result.parameter_labels
            fig = plt.figure(figsize=(10, 10), facecolor="w")
            fig = corner.corner(samps, labels=labels, fig=fig)
        if verbose:
            amp_ps = np.nanpercentile(result.posterior["amplitude"], [16, 50, 84])
            break_ps = np.nanpercentile(result.posterior["x_break"], [16, 50, 84])
            a1_ps = np.nanpercentile(result.posterior["alpha_1"], [16, 50, 84])
            a2_ps = np.nanpercentile(result.posterior["alpha_2"], [16, 50, 84])

            amplitude = round(amp_ps[1], uncertainty=amp_ps[2] - amp_ps[1])
            x_break = round(break_ps[1], uncertainty=break_ps[2] - break_ps[1])
            alpha_1 = round(a1_ps[1], uncertainty=a1_ps[2] - a1_ps[1])
            alpha_2 = round(a2_ps[1], uncertainty=a2_ps[2] - a2_ps[1])

            print("Fitting results:")
            print(f"    Amplitude: {amplitude} [{data.unit}]")
            print(f"    Break point: {x_break} [{u.deg}]")
            print(f"    alpha 1 (theta < break): {alpha_1}")
            print(f"    alpha 2 (theta > break): {alpha_2}")
    else:
        result = None

    ##############################################################################

    ##############################################################################
    if show_plots:
        plt.figure(figsize=(6, 6), facecolor="w")
        plt.plot(cbins, medians, ".", label="Median from MC")
        plt.errorbar(
            cbins.value, medians, yerr=err, color="tab:blue", marker=None, fmt=" "
        )
        if fit:
            cbins_hi = np.logspace(
                np.log10(cbins.value.min()), np.log10(cbins.value.max()), 1000
            )
            errmodel = []
            # Sample the posterior randomly 1000 times
            for i in range(1000):
                idx = np.random.choice(np.arange(result.posterior.shape[0]))
                _mod = model(
                    x=cbins_hi,
                    amplitude=result.posterior["amplitude"][idx],
                    x_break=result.posterior["x_break"][idx],
                    alpha_1=result.posterior["alpha_1"][idx],
                    alpha_2=result.posterior["alpha_2"][idx],
                )
                # errDict[name] = model_dict['posterior'][name][idx]
                errmodel.append(_mod)
            errmodel = np.array(errmodel)
            low, med, high = np.percentile(errmodel, [16, 50, 84], axis=0)
            plt.plot(cbins_hi, med, "-", color="tab:orange", label="Best fit")
            plt.fill_between(cbins_hi, low, high, color="tab:orange", alpha=0.5)

        saturate = np.var(data) * 2
        plt.hlines(
            saturate,
            cbins.value.min(),
            cbins.value.max(),
            linestyle="--",
            color="tab:red",
            label="Expected saturation ($2\sigma^2$)",
        )
        plt.xscale("log")
        plt.yscale("log")
        # plt.legend()
        plt.xlabel(rf"$\Delta\theta$ [{cbins.unit:latex_inline}]")
        plt.ylabel(rf"SF [{data.unit**2:latex_inline}]")
        plt.xlim(bins[0].value, bins[-1].value)
        plt.ylim(np.nanmin(medians) / 10, np.nanmax(medians) * 10)
        plt.legend()

        plt.figure(figsize=(6, 6), facecolor="w")
        plt.plot(cbins, count, ".", color="tab:red", label="Median from MC")
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel(rf"$\Delta\theta$ [{cbins.unit:latex_inline}]")
        plt.ylabel(r"Number of source pairs")
        plt.xlim(bins[0].value, bins[-1].value)

        counts = []
        cor_dists = sf_dists - d_sf_dists
        plt.figure()
        for dist in tqdm(cor_dists, disable=not verbose):
            n, hbins, _ = plt.hist(
                dist, range=(np.nanmin(cor_dists), np.nanmax(cor_dists)), bins=100
            )
            plt.clf()
            counts.append(n)
            c_hbins = []
            for i in range(len(hbins) - 1):
                c = (hbins[i] + hbins[i + 1]) / 2
                c_hbins.append(c)
        counts = np.array(counts)
        c_hbins = np.array(c_hbins)

        x = bins.value
        y = c_hbins
        X, Y = np.meshgrid(x, y)
        plt.figure(figsize=(7, 6), facecolor="w")
        plt.pcolormesh(X, Y, counts.T, cmap=plt.cm.cubehelix_r, shading="auto")
        plt.colorbar()
        plt.xticks(x)
        plt.yticks(y)
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel(rf"$\Delta\theta$ [{cbins.unit:latex_inline}]")
        plt.ylabel(rf"SF [{data.unit**2:latex_inline}]")
        plt.xlim(bins[0].value, bins[-1].value)
        plt.ylim(abs(np.nanmin(medians) / 10), np.nanmax(medians) * 10)
        if fit:
            plt.plot(cbins_hi, med, "-", color="tab:red", label="Best fit")
            plt.fill_between(cbins_hi, low, high, color="tab:red", alpha=0.5)
        plt.legend()
    ##############################################################################

    return cbins, medians, err, count, result
