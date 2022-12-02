#!/usr/bin/env python3
import inspect
import io
import itertools
import logging as logger
import os
import warnings
from typing import Callable, NamedTuple, Tuple, Union

import astropy.units as u
import bilby
import corner
import matplotlib.pyplot as plt
import numba as nb
import numpy as np
import pandas as pd
import xarray as xr
from astropy.coordinates import SkyCoord, UnitSphericalRepresentation
from astropy.visualization import quantity_support
from scipy.optimize import curve_fit
from sigfig import round
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

####################################
### Helper functions and classes ###
####################################

SFResult = NamedTuple(
    "SFResult",
    [
        ("med", np.ndarray),
        ("err_low", np.ndarray),
        ("err_high", np.ndarray),
        ("count", np.ndarray),
        ("c_bins", np.ndarray),
    ],
)


class TqdmToLogger(io.StringIO):
    """
    Output stream for TQDM which will output to logger module instead of
    the StdOut.
    """

    logger = None
    level = None
    buf = ""

    def __init__(self, logger, level=None):
        super(TqdmToLogger, self).__init__()
        self.logger = logger
        self.level = level or logger.INFO

    def write(self, buf):
        self.buf = buf.strip("\r\n\t ")

    def flush(self):
        self.logger.log(self.level, self.buf)


tqdm_out = TqdmToLogger(logger, level=logger.INFO)
logger.basicConfig(
    format="%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    force=True,
)

quantity_support()
warnings.filterwarnings("ignore")


def broken_power_law(
    x: np.ndarray,
    amplitude: float,
    x_break: float,
    alpha_1: float,
    alpha_2: float,
) -> np.ndarray:
    """Broken power law model

    Args:
        x (np.ndarray): Frequency
        amplitude (float): Amplitude
        x_break (float): Break frequency
        alpha_1 (float): Power law index below break frequency
        alpha_2 (float): Power law index above break frequency

    Returns:
        np.ndarray: Model array
    """
    alpha = np.where(x < x_break, alpha_1, alpha_2)
    xx = x / x_break
    return amplitude * np.power(xx, alpha)


def power_law(
    x: np.ndarray, amplitude: float, x_break: float, alpha: float
) -> np.ndarray:
    """Power law model

    Args:
        x (np.ndarray): Frequency
        amplitude (float): Amplitude
        x_break (float): Reference frequency
        alpha (float): Power law index

    Returns:
        np.ndarray: Model array
    """
    return amplitude * np.power(x / x_break, alpha)


def lsq_fit(
    x: np.ndarray, y: np.ndarray, outdir: str, label: str, model=broken_power_law
) -> bilby.core.result.Result:
    """Least squares fit

    Args:
        x (np.ndarray): X data
        y (np.ndarray): Y data
        outdir (str): Output directory
        label (str): Fitting label
        model (func, optional): Model function. Defaults to broken_power_law.

    Raises:
        NotImplementedError: if model is not implemented

    Returns:
        Result: Fitting result
    """
    params = inspect.getfullargspec(model).args[1:]
    result = bilby.core.result.Result(label=label, outdir=outdir)
    p0 = []
    param_labels = []
    p0.append(np.average([y.min(), y.max()]))
    param_labels.append(r"$\alpha$")
    p0.append(np.average([x.min(), x.max()]))
    param_labels.append(r"$\theta_\mathrm{break}$")
    if model is broken_power_law:
        p0.append(0)
        param_labels.append(r"$\alpha_1$")
        p0.append(0)
        param_labels.append(r"$\alpha_2$")
    elif model is power_law:
        p0.append(0)
        param_labels.append(r"$\alpha$")
    else:
        raise NotImplementedError("Model not implemented")
    popt, pcov = curve_fit(
        f=model,
        xdata=x,
        ydata=y,
        p0=p0,
    )

    params = inspect.getfullargspec(model).args[1:]
    # Randomly sample models using covariance matrix
    n_samples = 10_000
    samples = np.random.default_rng().multivariate_normal(popt, pcov, n_samples)
    result.posterior = pd.DataFrame(samples, columns=params)
    result.parameter_labels = list(params)
    result.search_parameter_keys = list(params)
    result.samples = samples
    result.parameter_labels_with_unit = param_labels
    return result


def lsq_weight_fit(
    x: np.ndarray,
    y: np.ndarray,
    yerr: np.ndarray,
    outdir: str,
    label: str,
    model=broken_power_law,
) -> bilby.core.result.Result:
    """Weighted least squares fit

    Args:
        x (np.ndarray): X data
        y (np.ndarray): Y data
        yerr (np.ndarray): Y error
        outdir (str): Output directory
        label (str): Label
        model (func, optional): Model function. Defaults to broken_power_law.

    Raises:
        NotImplementedError: If model is not implemented

    Returns:
        bilby.core.result.Result: Fiting result
    """
    result = bilby.core.result.Result(label=label, outdir=outdir)
    p0 = []
    param_labels = []
    p0.append(np.average([y.min() - yerr.max(), y.max() + yerr.max()]))
    param_labels.append(r"$\alpha$")
    p0.append(np.average([x.min(), x.max()]))
    param_labels.append(r"$\theta_\mathrm{break}$")
    if model is broken_power_law:
        p0.append(0)
        param_labels.append(r"$\alpha_1$")
        p0.append(0)
        param_labels.append(r"$\alpha_2$")
    elif model is power_law:
        p0.append(0)
        param_labels.append(r"$\alpha$")
    else:
        raise NotImplementedError("Model not implemented")
    popt, pcov = curve_fit(
        model,
        x,
        y,
        sigma=yerr,
        p0=p0,
    )

    params = inspect.getfullargspec(model).args[1:]
    # Randomly sample models using covariance matrix
    n_samples = 10_000
    samples = np.random.default_rng().multivariate_normal(popt, pcov, n_samples)
    result.posterior = pd.DataFrame(samples, columns=params)
    result.parameter_labels = list(params)
    result.search_parameter_keys = list(params)
    result.samples = samples
    result.parameter_labels_with_unit = param_labels
    return result


def bilby_fit(
    x: np.ndarray,
    y: np.ndarray,
    y_err: np.ndarray,
    outdir: str,
    label: str,
    model=broken_power_law,
    **kwargs,
) -> bilby.core.result.Result:
    """Bilby fit

    Args:
        x (np.ndarray): X data
        y (np.ndarray): Y data
        y_err (np.ndarray): Y error
        outdir (str): Output directory
        label (str): Label
        model (func, optional): Model function. Defaults to broken_power_law.

    Raises:
        NotImplementedError: If model is not implemented

    Returns:
        bilby.core.result.Result: Fitting result
    """
    # initialize a linear model
    likelihood = bilby.likelihood.GaussianLikelihood(x, y, model, y_err)
    priors = dict()
    priors["amplitude"] = bilby.core.prior.Uniform(
        y.min() - y_err.max(),
        y.max() + y_err.max(),
        name="amplitude",
        latex_label="$a$",
    )
    priors["x_break"] = bilby.core.prior.Uniform(
        x.min(), x.max(), name="x_break", latex_label=r"$\theta_\mathrm{break}$"
    )
    if model is broken_power_law:
        injection_parameters = dict(amplitude=1.0, x_break=1.0, alpha_1=1.0, alpha_2=1)
        priors["alpha_1"] = bilby.core.prior.Uniform(
            -2, 2, name="alpha_1", latex_label=r"$\alpha_1$"
        )
        priors["alpha_2"] = bilby.core.prior.Uniform(
            -2, 2, name="alpha_2", latex_label=r"$\alpha_2$"
        )
    elif model is power_law:
        injection_parameters = dict(
            amplitude=1.0,
            x_break=1.0,
            alpha=1.0,
        )
        priors["alpha"] = bilby.core.prior.Uniform(
            -2, 2, name="alpha", latex_label=r"$\alpha$"
        )
    else:
        raise NotImplementedError("Model not implemented")

    result = bilby.run_sampler(
        likelihood=likelihood,
        priors=priors,
        sample="unif",
        injection_parameters=injection_parameters,
        outdir=outdir,
        label=label,
        **kwargs,
    )
    result.parameter_labels = list(priors.keys())
    return result


def combinate(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return all combinations of data with itself

    Args:
        data (np.ndarray): Data to combine.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Data_1 matched with Data_2
    """
    ix, iy = np.triu_indices(data.shape[0], k=1)
    idx = np.vstack((ix, iy)).T
    dx, dy = data[idx].swapaxes(0, 1)
    return dx, dy


@nb.njit(parallel=True)
def mc_sample(data: np.ndarray, errors: np.ndarray, samples: int = 1000) -> np.ndarray:
    """Sample errors using Monte-Carlo
    Assuming Gaussian distribution.

    Args:
        data (np.ndarray): Measurements
        errors (np.ndarray): Errors
        samples (int, optional): Samples of the distribution. Defaults to 1000.

    Returns:
        np.ndarray: Sample array. Shape (len(data/errors),samples)
    """
    data_dist = np.zeros((len(data), samples)).astype(data.dtype)
    for i in nb.prange(data.shape[0]):
        data_dist[i] = np.random.normal(loc=data[i], scale=errors[i], size=samples)
    return data_dist

############################
### SF compute functions ###
############################

def sf_two_point(
    rm_1: np.ndarray,
    rm_2: np.ndarray,
    rm_err_1: np.ndarray,
    rm_err_2: np.ndarray,
    dtheta: u.Quantity,
    bins: u.Quantity,
    bin_unit: u.Quantity,
) -> SFResult:
    """Compute the two-point structure function

    Args:
        rm_1 (np.ndarray): RMs of source 1 in pair (n_samples, n_pairs)
        rm_2 (np.ndarray): RMs of source 2 in pair (n_samples, n_pairs)
        rm_err_1 (np.ndarray): RM errors of source 1 in pair (n_samples, n_pairs)
        rm_err_2 (np.ndarray): RM errors of source 2 in pair (n_samples, n_pairs)
        dtheta (u.Quantity): Separation between sources in pair (n_pairs)
        bins (u.Quantity): Angular (or 3D Euclidean) separation bins
        bin_unit (u.Quantity): Unit to set bins to when resolving units

    Returns:
        SFResult: Structure function results
    """
    samples = rm_1.shape[0]

    data_xr = xr.Dataset(
        dict(
            rm_1=(["sample", "source_pair"], rm_1),
            rm_2=(["sample", "source_pair"], rm_2),
            rm_err_1=(["sample", "source_pair"], rm_err_1),
            rm_err_2=(["sample", "source_pair"], rm_err_2),
        ),
        coords=dict(
            seps=("source_pair", dtheta.to(bin_unit)),
            sample=("sample", np.arange(samples)),
        ),
    )

    # Groupby separation
    grp = data_xr.groupby_bins("seps", bins.to(bin_unit).value)

    # Compute Structure Function
    sf_xr = grp.apply(lambda x: ((x.rm_1 - x.rm_2) ** 2).mean(dim="source_pair"))
    # Correct for errors
    sf_err_xr = grp.apply(
        lambda x: ((x.rm_err_1 - x.rm_err_2) ** 2).mean(dim="source_pair")
    )
    sf_corr_xr = sf_xr - sf_err_xr

    # Compute error
    p1, med, p2 = sf_corr_xr.quantile([0.16, 0.5, 0.84], dim="sample")

    err_low = med - p1
    err_high = p2 - med

    # Get source pair count
    count = grp.count(dim="source_pair").rm_1[:, 0]

    # Get bin centers
    c_bins = np.array([i.mid for i in sf_corr_xr.seps_bins.values]) * bin_unit

    return SFResult(
        med.values,
        err_low.values,
        err_high.values,
        count.values,
        c_bins,
    )


def sf_three_point(
    rm_1: np.ndarray,
    rm_2: np.ndarray,
    rm_err_1: np.ndarray,
    rm_err_2: np.ndarray,
    src_1: np.ndarray,
    src_2: np.ndarray,
    dtheta: u.Quantity,
    bins: u.Quantity,
    bin_unit: u.Quantity,
) -> SFResult:
    """Compute the three-point structure function

    Args:
        rm_1 (np.ndarray): RMs of source 1 in pair (n_samples, n_pairs)
        rm_2 (np.ndarray): RMs of source 2 in pair (n_samples, n_pairs)
        rm_err_1 (np.ndarray): RM errors of source 1 in pair (n_samples, n_pairs)
        rm_err_2 (np.ndarray): RM errors of source 2 in pair (n_samples, n_pairs)
        src_1 (np.ndarray): Source 1 in pair (n_pairs)
        src_2 (np.ndarray): Source 2 in pair (n_pairs)
        dtheta (u.Quantity): Separation between sources in pair (n_pairs)
        bins (u.Quantity): Angular (or 3D Euclidean) separation bins
        bin_unit (u.Quantity): Unit to set bins to when resolving units

    Returns:
        SFResult: Structure function result
    """
    samples = rm_1.shape[0]

    data_xr = xr.Dataset(
        dict(
            rm_1=(["sample", "source_pair"], rm_1),
            rm_2=(["sample", "source_pair"], rm_2),
            rm_err_1=(["sample", "source_pair"], rm_err_1),
            rm_err_2=(["sample", "source_pair"], rm_err_2),
        ),
        coords=dict(
            seps=("source_pair", dtheta.to(bin_unit)),
            sample=("sample", np.arange(samples)),
            src_1=("source_pair", src_1),
            src_2=("source_pair", src_2),
        ),
    )

    # Groupby separation
    grp = data_xr.groupby_bins("seps", bins.to(bin_unit).value)

    rm_1s = []
    rm_2s = []
    rm_3s = []
    rm_err_1s = []
    rm_err_2s = []
    rm_err_3s = []
    centres = []

    for i, g in tqdm(grp, desc="Grouping triplets", file=tqdm_out):
        # Find repeats of source number in each pair
        # If a source is repeated, then a triple is formed
        src_1s, count_1s = np.unique(g.src_1.values, return_counts=True)
        src_2s, count_2s = np.unique(g.src_2.values, return_counts=True)

        for s, (srcs, counts) in enumerate(zip([src_1s, src_2s], [count_1s, count_2s])):
            # Loop over src 1 then src 2
            s1 = 1 if s == 0 else 2
            s2 = 2 if s == 0 else 1
            for _src_a, ca in zip(srcs[counts > 1], counts[counts > 1]):
                t = g.where(g[f"src_{s1}"] == _src_a, drop=True)
                if len(t[f"src_{s1}"]) < 1:
                    continue
                _rm_1 = t[f"rm_{s1}"][:, 0].values
                _rm_err_1 = t[f"rm_err_{s1}"][:, 0].values
                for j in range(ca - 1):
                    _rm_2 = t[f"rm_{s2}"][:, j].values
                    _rm_3 = t[f"rm_{s2}"][:, j + 1].values
                    _rm_err_2 = t[f"rm_err_{s2}"][:, j].values
                    _rm_err_3 = t[f"rm_err_{s2}"][:, j + 1].values

                    rm_1s.append(_rm_1)
                    rm_2s.append(_rm_2)
                    rm_3s.append(_rm_3)
                    rm_err_1s.append(_rm_err_1)
                    rm_err_2s.append(_rm_err_2)
                    rm_err_3s.append(_rm_err_3)
                    centres.append(i.mid)

    # Create triplets dataset
    triple = xr.Dataset(
        dict(
            rm_1=(["source_triplet", "sample"], np.array(rm_1s)),
            rm_2=(["source_triplet", "sample"], np.array(rm_2s)),
            rm_3=(["source_triplet", "sample"], np.array(rm_3s)),
            rm_err_1=(["source_triplet", "sample"], np.array(rm_err_1s)),
            rm_err_2=(["source_triplet", "sample"], np.array(rm_err_2s)),
            rm_err_3=(["source_triplet", "sample"], np.array(rm_err_3s)),
        ),
        coords=dict(
            samples=("sample", np.arange(samples)),
            seps=("source_triplet", np.array(centres)),
        ),
    )

    # Groupby separtion 'bin'
    triple_grp = triple.groupby("seps")
    # Compute Structure Function
    sf_t_xr = triple_grp.apply(
        lambda x: ((x.rm_2 - 2 * x.rm_1 + x.rm_3) ** 2).mean(dim="source_triplet")
    )
    # TODO: Check if this is correct
    sf_err_t_xr = triple_grp.apply(
        lambda x: ((x.rm_err_2 - 2 * x.rm_err_1 + x.rm_err_3) ** 2).mean(
            dim="source_triplet"
        )
    )
    logger.warning("Correcting for errors in three point SF")
    sf_t_xr_corr = sf_t_xr - sf_err_t_xr

    p1, med, p2 = sf_t_xr_corr.quantile([0.16, 0.5, 0.84], dim="sample")

    err_low = med - p1
    err_high = p2 - med

    # Get source pair count
    count = triple_grp.count(dim="source_triplet").rm_1[:, 0]

    # Get bin centers
    c_bins = np.array([i for i in sf_t_xr_corr.seps.values]) * bin_unit

    return SFResult(
        med.values,
        err_low.values,
        err_high.values,
        count.values,
        c_bins,
    )


def fit_data(
    sf_result: SFResult,
    fit: str = "bilby",
    outdir: str = None,
    model_name: str = None,
    n_point: int = 2,
    show_plots: bool = False,
    save_plots: bool = False,
    **kwargs,
) -> Union[None, bilby.core.result.Result]:
    """Fit the structure function data

    Args:
        sf_result (SFResult): Structure function result
        fit (str, optional): Fit type. Defaults to "bilby".
        outdir (str, optional): Output directory for bilby. Defaults to None.
        model_name (str, optional): Model to fit. Defaults to None.
        n_point (int, optional): Number of points in SF. Defaults to 2.
        show_plots (bool, optional): Show fitting plots. Defaults to False.
        save_plots (bool, optional): Save fitting plots. Defaults to False.

    Raises:
        NotImplementedError: If model_name is not implemented
        ValueError: If fit is not implemented.

    Returns:
        Union[None, bilby.core.result.Result]: _description_
    """
    if outdir is None:
        outdir = "outdir"
    bilby.utils.check_directory_exists_and_if_not_mkdir(outdir)

    medians, err_low, err_high, count, c_bins = sf_result

    if not fit:
        return None, None, outdir

    if model_name is None:
        model_name = "broken_power_law"
    if model_name == "broken_power_law":
        model = broken_power_law
    elif model_name == "power_law":
        model = power_law
    else:
        raise NotImplementedError("Only implemented for broken_power_law and power_law")

    logger.info(f"Fitting SF with a {model_name.replace('_',' ')}...")
    # A few simple setup steps
    label = f"{model_name}_{n_point}_point"

    # Only use bins with at least 10 sources
    cut = (
        (count >= 10)
        & np.isfinite(c_bins)
        & np.isfinite(medians)
        & np.isfinite(err_low)
        & np.isfinite(err_high)
    )
    x = np.array(c_bins[cut].value)
    y = medians[cut]
    per84 = err_high + medians
    per16 = -err_low + medians
    y_err = (per84 - per16)[cut] / 2

    if fit == "lsq":
        result = lsq_fit(
            x=x,
            y=y,
            model=model,
            outdir=outdir,
            label=label,
        )
    elif fit == "lsq_weight":
        result = lsq_weight_fit(
            x=x,
            y=y,
            yerr=y_err,
            model=model,
            outdir=outdir,
            label=label,
        )
    elif fit == "bilby":
        result = bilby_fit(
            x=x, y=y, y_err=y_err, model=model, outdir=outdir, label=label, **kwargs
        )
    else:
        raise ValueError("Invalid fit type")

    if show_plots:
        try:
            result.plot_corner(dpi=300, save=save_plots)
        except:
            pass
        samps = result.samples
        labels = result.parameter_labels
        fig = plt.figure(facecolor="w")
        fig = corner.corner(samps, labels=labels, fig=fig)
        if save_plots:
            plt.savefig(
                os.path.join(outdir, f"{label}_corner.pdf"),
                dpi=300,
                bbox_inches="tight",
            )
    perc_dict = {
        key: np.nanpercentile(result.posterior[key], [16, 50, 84])
        for key in result.parameter_labels
    }

    round_dict = {
        key: round(
            perc_dict[key][1].astype(float),
            uncertainty=(perc_dict[key][2] - perc_dict[key][1]).astype(float),
        )
        for key in result.parameter_labels
    }
    logger.info("Fitting results:")
    for key in round_dict.keys():
        logger.info(f"{key}: {round_dict[key]}")
    logger.info(f"Fit log evidence: {result.log_evidence} ± {result.log_evidence_err}")

    return result, model, outdir


def plot_sf(
    data: u.Quantity,
    bins: u.Quantity,
    sf_result: SFResult,
    saturate: float,
    fit: str = None,
    result: bilby.core.result.Result = None,
    model: Callable = None,
    outdir: str = ".",
    save_plots: bool = False,
    label: str = "",
    n_point: int = 2,
):
    medians, err_low, err_high, count, cbins = sf_result
    word = "pairs" if n_point == 2 else "triplets"
    good_idx = count >= 10
    plt.figure(facecolor="w")
    plt.plot(
        cbins[good_idx],
        medians[good_idx],
        ".",
        c="tab:blue",
        label=f"Reliable bins (>= 10 source {word})",
    )
    plt.plot(
        cbins[~good_idx],
        medians[~good_idx],
        ".",
        c="tab:red",
        label=f"Unreliable bins (< 10 source {word})",
    )
    plt.errorbar(
        cbins.value[good_idx],
        medians[good_idx],
        yerr=(err_low[good_idx], err_high[good_idx]),
        color="tab:blue",
        marker=None,
        fmt=" ",
    )
    plt.errorbar(
        cbins.value[~good_idx],
        medians[~good_idx],
        yerr=(err_low[~good_idx], err_high[~good_idx]),
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
            s_dict = {
                key: result.posterior[key][idx] for key in result.parameter_labels
            }
            _mod = model(
                x=cbins_hi,
                **s_dict,
            )
            # errDict[name] = model_dict['posterior'][name][idx]
            errmodel.append(_mod)
        errmodel = np.array(errmodel)
        low, med, high = np.percentile(errmodel, [16, 50, 84], axis=0)
        # med = fitted_line(cbins_hi)
        plt.plot(cbins_hi, med, "-", color="tab:orange", label="Best fit")
        plt.fill_between(cbins_hi, low, high, color="tab:orange", alpha=0.5)

    plt.axhline(
        saturate,
        linestyle="--",
        color="tab:red",
        label="Expected saturation ($2\sigma^2$)"
        if n_point == 2
        else "Expected saturation ($6\sigma^2$)",
    )
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel(rf"$\Delta\theta$ [{cbins.unit:latex_inline}]")
    plt.ylabel(rf"SF [{data.unit**2:latex_inline}]")
    plt.xlim(bins[0].value, bins[-1].value)
    plt.ylim(np.nanmin(medians) / 10, np.nanmax(medians) * 10)
    plt.legend()
    if save_plots:
        plt.savefig(
            os.path.join(outdir, f"{label}_errorbar.pdf"), dpi=300, bbox_inches="tight"
        )

    plt.figure(facecolor="w")
    plt.plot(cbins, count, ".", color="tab:red", label="Median from MC")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel(rf"$\Delta\theta$ [{cbins.unit:latex_inline}]")
    plt.ylabel(rf"Number of source {word}")
    plt.xlim(bins[0].value, bins[-1].value)
    if save_plots:
        plt.savefig(
            os.path.join(outdir, f"{label}_counts.pdf"), dpi=300, bbox_inches="tight"
        )

#####################
### Main function ###
#####################

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
    model_name: str = None,
    n_point: int = 2,
    **kwargs,
) -> Tuple[SFResult, bilby.core.result.Result]:

    """Compute the second or third order structure function with Monte-Carlo error propagation.

    Args:
        data (u.Quantity): 1D array of data values.
        errors (u.Quantity): 1D array of errors.
        coords (SkyCoord): 1D array of coordinates.
        samples (int): Number of samples to use for Monte-Carlo error propagation.
        bins (Union[u.Quantity, int]): Bin edges of the structure function, or number of bins.
        show_plots (bool, optional): Show plots. Defaults to False.
        save_plots (bool, optional): Save plots. Defaults to False.
        verbose (bool, optional): Print progress. Defaults to False.
        fit (str, optional): How to fit the broken powerlaw. Can be 'astropy', 'astropy_mc' or 'bilby'. Defaults to None.
        outdir (str, optional): Output directory for bilby. Defaults to None.
        model_name (str, optional): Name of the model. Defaults to None. Can be 'broken_power_law' or 'power_law'.
        **kwargs: Additional keyword arguments to pass to the bilby.core.run_sampler function.

    Returns:
        Tuple[SFResult, bilby.core.result.Result]: The structure function result and the fitting result.
    """

    if verbose:
        logger.basicConfig(
            level=logger.INFO,
            format="%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            force=True,
        )
    else:
        logger.basicConfig(level=logger.ERROR)

    # Sample the errors assuming a Gaussian distribution

    logger.info("Sampling errors...")
    rm_dist = mc_sample(
        data=data.value.astype(np.float64),
        errors=errors.value.astype(np.float64),
        samples=samples,
    )
    d_rm_dist = mc_sample(
        data=errors.value.astype(np.float64),
        errors=errors.value.astype(np.float64),  # Yo dawg...
        samples=samples,
    )

    # Get all combinations of sources
    rm_1, rm_2 = combinate(rm_dist)
    d_rm_1, d_rm_2 = combinate(d_rm_dist)
    # Get the angular separation of the source pairs

    # Check if coords have distance
    has_distance = not issubclass(
        coords.data.__class__,
        UnitSphericalRepresentation,
    )

    logger.info("Getting angular separations...")
    ra_1, ra_2 = combinate(coords.ra)
    dec_1, dec2 = combinate(coords.dec)
    if has_distance:
        dist_1, dist_2 = combinate(
            coords.distance
        )

    coords_1 = SkyCoord(
        ra_1,
        dec_1,
        distance=dist_1 if has_distance else None,
    )
    coords_2 = SkyCoord(
        ra_2,
        dec2,
        distance=dist_2 if has_distance else None,
    )
    if has_distance:
        dtheta = coords_1.separation_3d(coords_2)
        bin_unit = u.pc
    else:
        dtheta = coords_1.separation(coords_2)
        bin_unit = u.deg

    # Auto compute bins
    if type(bins) is int:

        logger.info("Auto-computing bins...")
        nbins = bins
        start = np.log10(np.min(dtheta).to(bin_unit).value)
        stop = np.log10(np.max(dtheta).to(bin_unit).value)
        bins = np.logspace(start, stop, nbins, endpoint=True) * bin_unit
        logger.info(f"Maximal angular separation: {np.max(dtheta)}")
        logger.info(f"Minimal angular separation: {np.min(dtheta)}")
    else:
        nbins = len(bins)
    # Compute the SF

    logger.info("Computing SF...")

    if n_point == 2:
        sf_result = sf_two_point(
            rm_1=rm_1.T,
            rm_2=rm_2.T,
            rm_err_1=d_rm_1.T,
            rm_err_2=d_rm_2.T,
            dtheta=dtheta,
            bins=bins,
            bin_unit=bin_unit,
        )
        saturate = np.nanvar(data) * 2
    elif n_point == 3:
        source_ids = np.arange(len(coords))
        src_1, src_2 = combinate(source_ids)
        sf_result = sf_three_point(
            rm_1=rm_1.T,
            rm_2=rm_2.T,
            rm_err_1=d_rm_1.T,
            rm_err_2=d_rm_2.T,
            src_1=src_1,
            src_2=src_2,
            dtheta=dtheta,
            bins=bins,
            bin_unit=bin_unit,
        )
        saturate = np.nanvar(data) * 6
    else:
        raise NotImplementedError("Only 2 and 3 point SF are implemented.")

    # Fit the SF
    result, model, outdir = fit_data(
        sf_result=sf_result,
        fit=fit,
        outdir=outdir,
        model_name=model_name,
        n_point=n_point,
        show_plots=show_plots,
        save_plots=save_plots,
        **kwargs,
    )

    if show_plots:
        plot_sf(
            data=data,
            bins=bins,
            sf_result=sf_result,
            saturate=saturate,
            fit=fit,
            result=result,
            model=model,
            outdir=outdir,
            save_plots=save_plots,
            label=model_name,
            n_point=n_point,
        )

    return sf_result, result
