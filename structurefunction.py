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
import numba as nb
import xarray as xr
import logging as logger

logger.basicConfig(
    format="%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    force=True
)

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

def combinate(data: np.ndarray)->Tuple[np.ndarray, np.ndarray]:
    """Return all combinations of data with itself

    Args:
        data (np.ndarray): Data to combine.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Data_1 matched with Data_2
    """
    ix, iy = np.triu_indices(
                    data.shape[0],
                    k=1
                )
    idx = np.vstack((ix,iy)).T
    dx,dy = data[idx].swapaxes(0,1)
    return dx,dy

@nb.njit(parallel=True)
def mc_sample(
    data:np.ndarray, 
    errors:np.ndarray, 
    samples:int=1000
) -> np.ndarray:
    """Sample errors using Monte-Carlo
    Assuming Gaussian distribution.

    Args:
        data (np.ndarray): Measurements
        errors (np.ndarray): Errors
        samples (int, optional): Samples of the distribution. Defaults to 1000.

    Returns:
        np.ndarray: Sample array. Shape (len(data/errors),samples)
    """
    data_dist = np.zeros(
        (len(data), samples)
    ).astype(data.dtype)
    for i in nb.prange(data.shape[0]):
        data_dist[i] = np.random.normal(loc=data[i], scale=errors[i], size=samples)
    return data_dist


def structure_function(
    data: u.Quantity,
    errors: u.Quantity,
    coords: SkyCoord,
    samples: int,
    bins: Union[u.Quantity, int],
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

    if verbose:
        logger.basicConfig(
            level=logger.INFO,
            format="%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            force=True
        )

    # Sample the errors assuming a Gaussian distribution

    logger.info("Sampling errors...")
    rm_dist = mc_sample(
        data=data.value,
        errors=errors.value,
        samples=samples,
    )
    d_rm_dist = mc_sample(
        data=errors.value,
        errors=errors.value,
        samples=samples,
    )

    # Get all combinations of sources and compute the difference

    logger.info("Getting data differences...")
    diffs_dist = np.subtract(*combinate(rm_dist)).T**2

    # Get all combinations of data_errs sources and compute the difference

    logger.info("Getting data error differences...")
    d_diffs_dist = np.subtract(*combinate(d_rm_dist)).T**2

    # Get the angular separation of the source pairs

    logger.info("Getting angular separations...")
    ra_1, ra_2 = combinate(coords.ra)
    dec_1, dec2 = combinate(coords.dec)

    coords_1 = SkyCoord(ra_1, dec_1)
    coords_2 = SkyCoord(ra_2, dec2)
    dtheta = coords_1.separation(coords_2)

    # Auto compute bins
    if type(bins) is int:
    
        logger.info("Auto-computing bins...")
        nbins = bins
        start = np.log10(np.min(dtheta).to(u.deg).value)
        stop = np.log10(np.max(dtheta).to(u.deg).value)
        bins = np.logspace(start, stop, nbins, endpoint=True)*u.deg
    else:
        nbins = len(bins)
    # Compute the SF

    logger.info("Computing SF...")
    bins_idx = np.digitize(dtheta, bins, right=False)
    cbins = np.sqrt(bins[1:] * bins[:-1]) # Take geometric mean of bins - assuming log

    diffs_xr = xr.Dataset(
        dict(
            data=(["samples", "source pair"], diffs_dist),
            error=(["samples", "source pair"], d_diffs_dist)
            
        ),
        coords = dict(
            bins_idx=("source pair", bins_idx),
        )
    )

    # Compute SF
    sf_xr = diffs_xr.groupby("bins_idx").mean(dim="source pair")
    count_xr = diffs_xr["bins_idx"].groupby("bins_idx").count()
    # Get the final SF correcting for the errors
    sf_xr_cor = sf_xr.data - sf_xr.error
    per16_xr,medians_xr,per84_xr=sf_xr_cor.quantile(
        (0.16,0.5,0.84),
        dim="samples"
    )
    # Return to numpy arrays for use later
    count = np.zeros_like(cbins.value)
    medians = np.zeros_like(cbins.value)
    per16 = np.zeros_like(cbins.value)
    per84 = np.zeros_like(cbins.value)
    sf_dists_cor = np.zeros((len(cbins), samples))
    sf_dists = np.zeros((len(cbins), samples))
    d_sf_dists = np.zeros((len(cbins), samples))
    for arr, xarr in zip(
        (count, medians, per16, per84, sf_dists_cor, sf_dists, d_sf_dists), 
        (count_xr, medians_xr, per16_xr, per84_xr, sf_xr_cor, sf_xr.data, sf_xr.error),
    ):
        arr[count_xr.coords.to_index()[:-1]] = xarr[:-1]
    err_low = medians - per16
    err_high = per84 - medians
    err = np.array([err_low.astype(float), err_high.astype(float)])
    if fit:
    
        logger.info("Fitting SF with a broken power law...")
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

            logger.info("Fitting results:")
            logger.info(f"    Amplitude: {amplitude} [{data.unit**2}]")
            logger.info(f"    Break point: {x_break} [{u.deg}]")
            logger.info(f"    alpha 1 (theta < break): {alpha_1}")
            logger.info(f"    alpha 2 (theta > break): {alpha_2}")
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
        plt.pcolormesh(X, Y, counts.T, cmap=plt.cm.cubehelix_r)
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
