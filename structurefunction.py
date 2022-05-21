import os
from typing import Tuple, Union
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
from astropy.visualization import quantity_support
from astropy.modeling import models, fitting
import pandas as pd

quantity_support()
warnings.filterwarnings("ignore")
# warnings.filterwarnings("ignore", message="All-NaN slice encountered")
# warnings.filterwarnings("ignore", message="All-NaN axis encountered")
# warnings.filterwarnings("ignore", message="Mean of empty slice")
# warnings.filterwarnings("ignore", message="converting a masked element to nan")
# warnings.filterwarnings("ignore", message="invalid value encountered in power")
# warnings.filterwarnings("ignore", message="overflow encountered in power")


def model(x, amplitude, x_break, alpha_1, alpha_2):
    alpha = np.where(x < x_break, alpha_1, alpha_2)
    xx = x / x_break
    return amplitude * np.power(xx, -alpha)


def astropy_fit(x, y, y_err, y_dist, outdir, label, verbose=False, **kwargs):
    result = bilby.core.result.Result(label=label, outdir=outdir)
    fitter = fitting.LevMarLSQFitter()

    # initialize a linear model
    line_init = models.BrokenPowerLaw1D()
    fitted_line = fitter(line_init, x, y, weights=1.0 / y_err ** 2)
    amplitude, x_break, alpha_1, alpha_2 = fitted_line.parameters
    posterior = {
        "amplitude": amplitude,
        "x_break": x_break,
        "alpha_1": alpha_1,
        "alpha_2": alpha_2,
    }
    result.posterior = pd.DataFrame(posterior, index=[0])
    result.parameter_labels = list(posterior.keys())
    result.samples = np.array(list(posterior.values())).T[np.newaxis, :]
    return result


def astropy_fit_mc(x, y, y_err, y_dist, outdir, label, verbose=False, **kwargs):
    result = bilby.core.result.Result(label=label, outdir=outdir)
    fitter = fitting.LevMarLSQFitter()

    posterior = {
        "amplitude": [],
        "x_break": [],
        "alpha_1": [],
        "alpha_2": [],
    }
    # initialize a linear model
    line_init = models.BrokenPowerLaw1D()

    # loop over the samples
    for y in tqdm(y_dist.T, disable=not verbose, desc="Fitting"):
        fitted_line = fitter(line_init, x, y)
        amplitude, x_break, alpha_1, alpha_2 = fitted_line.parameters
        posterior["amplitude"].append(amplitude)
        posterior["x_break"].append(x_break)
        posterior["alpha_1"].append(alpha_1)
        posterior["alpha_2"].append(alpha_2)
    result.posterior = pd.DataFrame.from_dict(posterior)
    result.parameter_labels = list(posterior.keys())
    result.samples = np.array(list(posterior.values())).T
    return result


def bilby_fit(x, y, y_err, y_dist, outdir, label, verbose=False, **kwargs):
    # initialize a linear model
    injection_parameters = dict(amplitude=1.0, x_break=1.0, alpha_1=1.0, alpha_2=1)
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
    return result


def structure_function(
    data: u.Quantity,
    errors: u.Quantity,
    coords: SkyCoord,
    samples: int,
    bins: Union[u.Quantity, int],
    weights: np.ndarray = None,
    show_plots: bool = False,
    save_plots: bool = False,
    verbose: bool = False,
    fit: str = None,
    outdir: str = None,
    **kwargs,
) -> Tuple[u.Quantity, u.Quantity, Tuple[u.Quantity, u.Quantity], np.ndarray]:


    """Compute the second order structure function with Monte-Carlo error propagation.

    Args:
        data (u.Quantity): 1D array of data values.
        errors (u.Quantity): 1D array of errors.
        coords (SkyCoord): 1D array of coordinates.
        samples (int): Number of samples to use for Monte-Carlo error propagation.
        bins (Union[u.Quantity, int]): Bin edges of the structure function, or number of bins.
        show_plots (bool, optional): Show plots. Defaults to False.
        verbose (bool, optional): Print progress. Defaults to False.
        fit (str, optional): How to fit the broken powerlaw. Can be 'astropy', 'astropy_mc' or 'bilby'. Defaults to None.
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
        rm_dist.append(
            np.random.normal(loc=data[i].value, scale=errors[i].value, size=samples)
        )
        d_rm_dist.append(
            np.random.normal(loc=errors[i].value, scale=errors[i].value, size=samples)
        )
    rm_dist = np.array(rm_dist)
    d_rm_dist = np.array(d_rm_dist)

    # Get all combinations of sources and compute the difference
    if verbose:
        print("Getting data differences...")
    F_dist = rm_dist[
        np.array(
            np.triu_indices(
                rm_dist.shape[0], 
                k=1
            )
        ).swapaxes(0, 1)
    ]
    if weights is None:
        weights = np.ones(data.shape[0])
    w_dist = np.mean(
            weights[
            np.array(
                np.triu_indices(
                    weights.shape[0], 
                    k=1
                )
            ).swapaxes(0, 1)
        ],
        axis=1,
    )

    diffs_dist = np.transpose((F_dist[:, 0] - F_dist[:, 1])**2)

    # Get all combinations of data_errs sources and compute the difference
    if verbose:
        print("Getting data error differences...")
    dF_dist = d_rm_dist[
        np.array(
            np.triu_indices(
                d_rm_dist.shape[0],
                k=1
            )
        ).swapaxes(0,1)
    ]
    d_diffs_dist = np.transpose((dF_dist[:, 0] - dF_dist[:, 1])**2)

    # Get the angular separation of the source pairs
    if verbose:
        print("Getting angular separations...")
    cx_ra_perm, cy_ra_perm = coords.ra.to(u.deg).value[
        np.array(
            np.triu_indices(
                coords.ra.to(u.deg).value.shape[0],
                k=1
            )
        ).swapaxes(0,1)
    ].T
    cx_dec_perm, cy_dec_perm = coords.dec.to(u.deg).value[
        np.array(
            np.triu_indices(
                coords.dec.to(u.deg).value.shape[0],
                k=1
            )
        ).swapaxes(0,1)
    ].T

    coords_x = SkyCoord(cx_ra_perm * u.deg, cx_dec_perm * u.deg)
    coords_y = SkyCoord(cy_ra_perm * u.deg, cy_dec_perm * u.deg)
    dtheta = coords_x.separation(coords_y)

    # Auto compute bins
    if type(bins) is int:
        if verbose:
            print("Auto-computing bins...")
        nbins = bins
        start = np.log10(np.min(dtheta).to(u.deg).value)
        stop = np.log10(np.max(dtheta).to(u.deg).value)
        bins = np.logspace(start, stop, nbins, endpoint=True)*u.deg
    else:
        nbins = len(bins)
    # Compute the SF
    if verbose:
        print("Computing SF...")
    sf_dists = np.zeros((nbins - 1, samples)) * np.nan
    d_sf_dists = np.zeros((nbins - 1, samples)) * np.nan
    count = np.zeros((nbins - 1)) * np.nan
    cbins = np.zeros((nbins - 1)) * np.nan * u.deg
    for i, b in enumerate(tqdm(bins[:-1], disable=not verbose)):
        bin_idx = (bins[i] <= dtheta) & (dtheta < bins[i + 1])
        centre = (bins[i] + bins[i + 1]) / 2

        cbins[i] = centre
        count[i] = np.sum(bin_idx)
        try:
            sf_dist = np.average(
                diffs_dist[:, bin_idx], axis=1, weights=w_dist[bin_idx]
            )
            d_sf_dist = np.average(
                d_diffs_dist[:, bin_idx], axis=1, weights=w_dist[bin_idx]
            )
            sf_dists[i] = sf_dist
            d_sf_dists[i] = d_sf_dist
        except ZeroDivisionError:
            continue

    # Get the final SF correcting for the errors
    sf_dists_cor = sf_dists - d_sf_dists
    medians = np.nanmedian(sf_dists_cor, axis=1)
    per16 = np.nanpercentile(sf_dists_cor, 16, axis=1)
    per84 = np.nanpercentile(sf_dists_cor, 84, axis=1)
    err_low = medians - per16
    err_high = per84 - medians
    err = np.array([err_low.astype(float), err_high.astype(float)])

    if fit:
        if verbose:
            print("Fitting SF with a broken power law...")
        # A few simple setup steps
        label = "linear_regression"
        if outdir is None:
            outdir = "outdir"
        bilby.utils.check_directory_exists_and_if_not_mkdir(outdir)

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
        y_err = (per84 - per16)[cut] / 2
        y_dist = sf_dists_cor[cut]

        if fit == "astropy":
            fit_func = astropy_fit
        elif fit == "astropy_mc":
            fit_func = astropy_fit_mc
        elif fit == "bilby":
            fit_func = bilby_fit
        else:
            raise ValueError("Invalid fit type")
        result = fit_func(x, y, y_err, y_dist, outdir, label, verbose=verbose, **kwargs)
        if show_plots and fit != "astropy":
            samps = result.samples
            labels = result.parameter_labels
            fig = plt.figure(figsize=(10, 10), facecolor="w")
            fig = corner.corner(samps, labels=labels, fig=fig)
            if save_plots:
                plt.savefig(os.path.join(outdir, "corner.pdf"))
        if verbose:
            amp_ps = np.nanpercentile(result.posterior["amplitude"], [16, 50, 84])
            break_ps = np.nanpercentile(result.posterior["x_break"], [16, 50, 84])
            a1_ps = np.nanpercentile(result.posterior["alpha_1"], [16, 50, 84])
            a2_ps = np.nanpercentile(result.posterior["alpha_2"], [16, 50, 84])

            amplitude = amp_ps[1]
            x_break = break_ps[1]
            alpha_1 = a1_ps[1]
            alpha_2 = a2_ps[1]

            if fit != "astropy": 
                amplitude = round(amplitude, uncertainty=amp_ps[2] - amp_ps[1])
                x_break = round(x_break, uncertainty=break_ps[2] - break_ps[1])
                alpha_1 = round(alpha_1, uncertainty=a1_ps[2] - a1_ps[1])
                alpha_2 = round(alpha_2, uncertainty=a2_ps[2] - a2_ps[1])

            print("Fitting results:")
            print(f"    Amplitude: {amplitude} [{data.unit**2}]")
            print(f"    Break point: {x_break} [{u.deg}]")
            print(f"    alpha 1 (theta < break): {alpha_1}")
            print(f"    alpha 2 (theta > break): {alpha_2}")
    else:
        result = None

    ##############################################################################

    ##############################################################################
    if show_plots:
        good_idx = count >= 10
        plt.figure(figsize=(6, 6), facecolor="w")
        plt.plot(
            cbins[good_idx],
            medians[good_idx],
            ".",
            c="tab:blue",
            label="Reliable bins (>= 10 source pairs)",
        )
        plt.plot(
            cbins[~good_idx],
            medians[~good_idx],
            ".",
            c="tab:red",
            label="Unreliable bins (< 10 source pairs)",
        )
        plt.errorbar(
            cbins.value[good_idx],
            medians[good_idx],
            yerr=err[:, good_idx],
            color="tab:blue",
            marker=None,
            fmt=" ",
        )
        plt.errorbar(
            cbins.value[~good_idx],
            medians[~good_idx],
            yerr=err[:, ~good_idx],
            color="tab:red",
            marker=None,
            fmt=" ",
        )
        if fit:
            cbins_hi = np.logspace(
                np.log10(cbins.value.min()), np.log10(cbins.value.max()), 1000
            )
            errmodel = []
            # Sample the posterior randomly 100 times
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
            # med = fitted_line(cbins_hi)
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
        if save_plots:
            plt.savefig(os.path.join(outdir, "errorbar.png"))

        plt.figure(figsize=(6, 6), facecolor="w")
        plt.plot(cbins, count, ".", color="tab:red", label="Median from MC")
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel(rf"$\Delta\theta$ [{cbins.unit:latex_inline}]")
        plt.ylabel(r"Number of source pairs")
        plt.xlim(bins[0].value, bins[-1].value)
        if save_plots:
            plt.savefig(os.path.join(outdir, "counts.png"))

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

        x = cbins
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
            plt.plot(cbins_hi, med, "-", color="tab:orange", label="Best fit")
            plt.fill_between(cbins_hi, low, high, color="tab:orange", alpha=0.5)
        plt.legend()
        plt.hlines(
            saturate,
            cbins.value.min(),
            cbins.value.max(),
            linestyle="--",
            color="tab:red",
            label="Expected saturation ($2\sigma^2$)",
        )
        if save_plots:
            plt.savefig(os.path.join(outdir, "PDF.png"))
    ##############################################################################

    return cbins, medians, err, count, result
