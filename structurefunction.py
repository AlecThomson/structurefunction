from typing import Tuple
import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import itertools


def structure_function(
    data: u.Quantity,
    errors: u.Quantity,
    coords: SkyCoord,
    samples: int,
    bins: u.Quantity,
    show_plots=False,
) -> Tuple[u.Quantity, u.Quantity, Tuple[u.Quantity, u.Quantity], np.ndarray]:

    """Compute the second order structure function with Monte-Carlo error propagation.

    Args:
        data (u.Quantity): 1D array of data values.
        errors (u.Quantity): 1D array of errors.
        coords (SkyCoord): 1D array of coordinates.
        samples (int): Number of samples to use for Monte-Carlo error propagation.
        bins (u.Quantity): Bin edges of the structure function.
        show_plots (bool, optional): Show plots. Defaults to False.

    Returns:
        Tuple[u.Quantity, u.Quantity, Tuple[u.Quantity, u.Quantity], np.ndarray]: 
            cbins: center of bins.
            medians: median of the structure function.
            errors: upper and lower error of the structure function.
            counts: number of source pairs in each bin.
    """

    # Sample the errors assuming a Gaussian distribution
    print("Sampling errors...")
    rm_dist = []
    d_rm_dist = []
    for i in tqdm(range(data.shape[0]), "Sample Gaussian"):
        rm_dist.append(np.random.normal(loc=data[i], scale=errors[i], size=samples))
        d_rm_dist.append(np.random.normal(loc=errors[i], scale=errors[i], size=samples))
    rm_dist = np.array(rm_dist)
    d_rm_dist = np.array(d_rm_dist)

    # Get all combinations of sources and compute the difference
    print("Getting data differences...")
    F_dist = np.array(list(itertools.combinations(rm_dist, r=2)))

    diffs_dist = ((F_dist[:, 0] - F_dist[:, 1]) ** 2).T

    # Get all combinations of data_errs sources and compute the difference
    print("Getting data error differences...")
    dF_dist = np.array(list(itertools.combinations(d_rm_dist, r=2)))

    d_diffs_dist = ((dF_dist[:, 0] - dF_dist[:, 1]) ** 2).T

    # Get the angular separation of the source paris
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
    print("Computing SF...")
    sf_dists = np.zeros((len(bins) - 1, samples)) * np.nan
    d_sf_dists = np.zeros((len(bins) - 1, samples)) * np.nan
    count = np.zeros((len(bins) - 1)) * np.nan
    cbins = np.zeros((len(bins) - 1)) * np.nan * u.deg
    for i, b in enumerate(tqdm(bins)):
        if i + 1 == len(bins):
            break
        else:
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
    ##############################################################################

    ##############################################################################
    if show_plots:
        plt.figure(figsize=(6, 6), facecolor="w")
        plt.plot(cbins, medians, ".", label="Median from MC")
        plt.errorbar(
            cbins.value, medians, yerr=err, color="tab:blue", marker=None, fmt=" "
        )
        plt.xscale("log")
        plt.yscale("log")
        # plt.legend()
        plt.xlabel(rf"$\Delta\theta$ [{cbins.unit:latex_inline}]")
        plt.ylabel(rf"SF [{data.unit**2:latex_inline}]")
        spec = cbins ** (2 / 3)
        spec = spec / np.nanmax(spec) * np.nanmax(medians)
        plt.plot(cbins, spec, "--", label="Kolmogorov")
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
        for dist in tqdm(cor_dists):
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
        plt.pcolormesh(X, Y, counts.T, cmap=plt.cm.cubehelix_r)
        plt.colorbar()
        plt.xticks(x)
        plt.yticks(y)
        plt.xscale("log")
        plt.yscale("log")
        plt.plot(cbins, spec, "--", label="Kolmogorov")
        plt.xlabel(rf"$\Delta\theta$ [{cbins.unit:latex_inline}]")
        plt.ylabel(rf"SF [{data.unit**2:latex_inline}]")
        plt.xlim(bins[0].value, bins[-1].value)
        plt.ylim(abs(np.nanmin(medians) / 10), np.nanmax(medians) * 10)
        plt.legend()
    ##############################################################################

    return cbins, medians, err, count
