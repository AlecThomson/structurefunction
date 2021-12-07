import numpy as np
from astropy.table import Table
from astropy.coordinates import SkyCoord
import astropy.units as u
from tqdm.auto import tqdm, trange
import matplotlib.pyplot as plt
import healpy as hp
import dask.array as da
import dask.dataframe as dd



def rm_structure(RM, d_RM, coords, samples, bins):
    '''
    Comes the RM structure function with Monte-Carlo error propagation.
    Inputs: 
        RM - list of RMs or FDs
        d_RM - Errors in RM/FD (absolute)
        coords - catalogue SkyCoord object
        samlpes - Number of samples of Gaussian distribution for errors
        bins - array of angular separation bins to compute SF
    Returns:
        rm_sf - the RM structure function
        err_low - -ve error bar
        err_high - +ve error bar
        nums - No. of sources per bin
    '''
    ##############################################################################
    rm_dist = []
    d_rm_dist = []
    for i in tqdm(range(RM.shape[0]), 'Sample Gaussian'):
        rm_dist.append(np.random.normal(loc = RM[i], 
                                        scale = d_RM[i], 
                                        size = samples))
        d_rm_dist.append(np.random.normal(loc = d_RM[i], 
                                          scale = d_RM[i], 
                                          size = samples))
    rm_dist = np.array(rm_dist)
    d_rm_dist = np.array(d_rm_dist)
    ##############################################################################
    
    ##############################################################################
    # Compute differences and separations
    FX_dist, FY_dist = [],[]
    for dist in tqdm(rm_dist.T, desc='Making grids'):
        FX, FY = np.meshgrid(dist, dist)
        FX_dist.append(FX)
        FY_dist.append(FY)
    FX_dist = np.array(FX_dist)
    FY_dist = np.array(FY_dist)



    # Compute differences and separations
    d_FX_dist, d_FY_dist = [],[]
    for dist in tqdm(d_rm_dist.T, desc='Making grids'):
        d_FX, d_FY = np.meshgrid(dist, dist)
        d_FX_dist.append(d_FX)
        d_FY_dist.append(d_FY)
    d_FX_dist = np.array(d_FX_dist)
    d_FY_dist = np.array(d_FY_dist)

    FX, FY = np.meshgrid(RM, RM)
    dFX, dFY = np.meshgrid(d_RM, d_RM)
    ##############################################################################
    
    ##############################################################################
    diff_dist = (FX_dist - FY_dist)**2
    diff = (FX - FY)**2
    d_diff = (dFX - dFY)**2

    diffs_dist = []
    for d in tqdm(diff_dist, desc='Computing RM difference'):
        diffs = d[np.triu_indices_from(d, k=+1)]
        diffs_dist.append(diffs)
    diffs_dist = np.array(diffs_dist)

    d_diff_dist = (d_FX_dist - d_FY_dist)**2

    d_diffs_dist = []
    for d in tqdm(d_diff_dist, desc='Computing RM difference'):
        diffs = d[np.triu_indices_from(d, k=+1)]
        d_diffs_dist.append(diffs)
    d_diffs_dist = np.array(d_diffs_dist)

    diffs  = diff[np.triu_indices_from(diff, k=+1)]
    d_diffs = d_diff[np.triu_indices_from(d_diff, k=+1)]
    ##############################################################################
    
    ##############################################################################
    print('Making coord grid')
    cX_ra, cY_ra = np.meshgrid(coords.ra, coords.ra)
    cX_dec, cY_dec = np.meshgrid(coords.dec, coords.dec)

    print('Computing angular separations')
    cx_ra_perm = cX_ra[np.triu_indices_from(cX_ra, k=+1)]
    cx_dec_perm = cX_dec[np.triu_indices_from(cX_dec, k=+1)]

    cy_ra_perm = cY_ra[np.triu_indices_from(cY_ra, k=+1)]
    cy_dec_perm = cY_dec[np.triu_indices_from(cY_dec, k=+1)]

    coords_x = SkyCoord(cx_ra_perm, cx_dec_perm)
    coords_y = SkyCoord(cy_ra_perm, cy_dec_perm)

    dtheta = coords_x.separation(coords_y)
    ##############################################################################
    
    ##############################################################################
    sf_dists = np.zeros((len(bins)-1,samples))*np.nan
    d_sf_dists = np.zeros((len(bins)-1,samples))*np.nan
    d_sfs = np.zeros((len(bins)-1))*np.nan
    sfs = np.zeros((len(bins)-1))*np.nan
    count = np.zeros((len(bins)-1))*np.nan
    cbins = np.zeros((len(bins)-1))*np.nan*u.deg
    for i, b in enumerate(tqdm(bins)):
        if i+1 == len(bins):
            break
        else:
            bin_idx = (bins[i] <= dtheta) & (dtheta < bins[i+1])
            centre = (bins[i]+bins[i+1])/2

            cbins[i] = centre
            count[i] = np.sum(bin_idx)
            sf_dist = np.nanmean(diffs_dist[:, bin_idx], axis=1)
            d_sf_dist = np.nanmean(d_diffs_dist[:, bin_idx], axis=1)

            sf = np.nanmean(diffs[bin_idx])
            d_sf = np.nanmean(d_diffs[bin_idx])
            sf_dists[i] = sf_dist
            d_sf_dists[i] = d_sf_dist
            sfs[i] = sf
            d_sfs[i] = d_sf

    medians = np.nanmedian(sf_dists - d_sf_dists, axis=1)
    per16 = np.nanpercentile(sf_dists- d_sf_dists, 16, axis = 1) 
    per84 = np.nanpercentile(sf_dists- d_sf_dists, 84, axis = 1)
    err_low = medians - per16
    err_high = per84 - medians
    err = [err_low.astype(float), err_high.astype(float)]
    ##############################################################################
    
    ##############################################################################
    plt.figure(figsize=(6,6))
    plt.plot(cbins, medians, '.', label='Median from MC')
    plt.errorbar(cbins.value, medians, yerr=err, color='tab:blue', marker=None, fmt=' ')
    plt.xscale('log')
    plt.yscale('log')
    #plt.legend()
    plt.xlabel(r"$\Delta\theta$ [deg]")
    plt.ylabel(r"RM SF [rad$^2\,$m$^{-4}$]")
    spec = cbins ** (2/3)
    spec = spec/np.nanmax(spec)*np.nanmax(medians)
    plt.plot(cbins, spec, '--', label='Kolmogorov')
    plt.xlim(bins[0].value, bins[-1].value)
    plt.ylim(np.nanmin(medians)/10, np.nanmax(medians)*10)
    plt.legend()
    
    plt.figure(figsize=(6,6))
    plt.plot(cbins, count, '.', color='tab:red', label='Median from MC')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r"$\Delta\theta$ [deg]")
    plt.ylabel(r"Number of source pairs")
    plt.xlim(bins[0].value, bins[-1].value)
    
    counts = []
    cor_dists = sf_dists - d_sf_dists
    plt.figure()
    for dist in tqdm(cor_dists):
        n, hbins, _ = plt.hist(dist, range=(np.nanmin(cor_dists),np.nanmax(cor_dists)), bins=100)
        plt.clf()
        counts.append(n)
        c_hbins = []
        for i in range(len(hbins)-1):
            c = (hbins[i]+hbins[i+1])/2
            c_hbins.append(c)
    counts = np.array(counts)
    c_hbins = np.array(c_hbins)
    
    x = bins.value
    y = c_hbins
    X,Y = np.meshgrid(x,y)
    plt.figure(figsize=(7,6), facecolor='k')
    plt.pcolormesh(X,Y,counts.T, cmap=plt.cm.cubehelix)
    plt.colorbar()
    plt.xticks(x)
    plt.yticks(y)
    plt.xscale('log')
    plt.yscale('log')
    plt.plot(cbins, spec, '--', label='Kolmogorov')
    plt.xlabel(r"$\Delta\theta$ [deg]")
    plt.ylabel(r"RM SF [rad$^2\,$m$^{-4}$]")
    plt.xlim(bins[0].value, bins[-1].value)
    plt.ylim(abs(np.nanmin(medians)/10), np.nanmax(medians)*10)
    plt.legend()
    ##############################################################################
    
    return cbins, medians, err, count