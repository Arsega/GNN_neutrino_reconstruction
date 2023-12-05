# Imports
import os
import subprocess
import numpy as np
import time
from constants import dataset
from radiotools import helper as hp
from radiotools import stats
from NuRadioReco.utilities import units
import pickle


# -------
# Imports for histogram2d
import math
import os
from matplotlib import colors as mcolors
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import sys
# -----------------------


def root_dir():
    return ".."

def common_dir():
    return f"{root_dir()}/aserra/aserra/GNN_MSE_direction"

def models_dir(run_name):
    return f"{common_dir()}/models/{run_name}"


# Loading data and label files
def load_file(i_file, norm=1e-6):
    # Load 500 MHz filter
    filt = np.load("/home/aserra/aserra/bandpass_filters/500MHz_filter.npy")

    #     t0 = time.time()
    #     print(f"loading file {i_file}", flush=True)
    data = np.load(os.path.join(dataset.datapath, f"{dataset.data_filename}{i_file:04d}.npy"), allow_pickle=True)
    #print('1',data.shape)
    data = np.fft.irfft(np.fft.rfft(data, axis=-1) * filt, axis=-1)
    #print('2', data.shape)
    data = data[:, :, :, np.newaxis]
    #print('3', data.shape)

    labels_tmp = np.load(os.path.join(dataset.datapath, f"{dataset.label_filename}{i_file:04d}.npy"), allow_pickle=True)
    #     print(f"finished loading file {i_file} in {time.time() - t0}s")
    nu_zenith_data = np.array(labels_tmp.item()["nu_zenith"])
    nu_azimuth_data = np.array(labels_tmp.item()["nu_azimuth"])


    # check for nans and remove them
    idx = ~(np.isnan(data))
    idx = np.all(idx, axis=1) 
    idx = np.all(idx, axis=1)
    idx = np.all(idx, axis=1)
    data = data[idx, :, :, :]
    nu_zenith_data = nu_zenith_data[idx]
    nu_azimuth_data = nu_azimuth_data[idx]

    labels = hp.spherical_to_cartesian(nu_zenith_data, nu_azimuth_data)
    data /= norm

    return data, labels

# Loading data and label files and also other properties
def load_file_all_properties(i_file, norm=1e-6):
    t0 = time.time()
    print(f"loading file {i_file}", flush=True)

    # Load 500 MHz filter
    filt = np.load(f"{common_dir()}/bandpass_filters/500MHz_filter.npy")
    print(dataset.datapath, dataset.data_filename)
    data = np.load(os.path.join(dataset.datapath, f"{dataset.data_filename}{i_file:04d}.npy"), allow_pickle=True)
    data = np.fft.irfft(np.fft.rfft(data, axis=-1) * filt, axis=-1)
    data = data[:, :, :, np.newaxis]
    
    labels_tmp = np.load(os.path.join(dataset.datapath, f"{dataset.label_filename}{i_file:04d}.npy"), allow_pickle=True)
    print(f"finished loading file {i_file} in {time.time() - t0}s")
    nu_zenith_data = np.array(labels_tmp.item()["nu_zenith"])
    nu_azimuth_data = np.array(labels_tmp.item()["nu_azimuth"])
    nu_direction_data = hp.spherical_to_cartesian(nu_zenith_data, nu_azimuth_data)

    nu_energy_data = np.array(labels_tmp.item()["nu_energy"])
    nu_flavor_data = np.array(labels_tmp.item()["nu_flavor"])
    shower_energy_data = np.array(labels_tmp.item()["shower_energy"])

    # check for nans and remove them
    idx = ~(np.isnan(data))
    idx = np.all(idx, axis=1)
    idx = np.all(idx, axis=1)
    idx = np.all(idx, axis=1)
    data = data[idx, :, :, :]
    data /= norm

    nu_zenith_data = nu_zenith_data[idx]
    nu_azimuth_data = nu_azimuth_data[idx]
    nu_direction_data = nu_direction_data[idx]
    nu_energy_data = nu_energy_data[idx]
    nu_flavor_data = nu_flavor_data[idx]
    shower_energy_data = shower_energy_data[idx]

    return data, nu_direction_data, nu_zenith_data, nu_azimuth_data, nu_energy_data, nu_flavor_data, shower_energy_data


def calculate_percentage_interval(energy_difference_data, percentage=0.68):
    # Redefine N
    N = energy_difference_data.size
    weights = np.ones(N)

    # Take abs due to the fact that the energy difference can be negative
    energy = stats.quantile_1d(np.abs(energy_difference_data), weights, percentage)

    # OLD METHOD -------------------------------
    # Calculate Rayleigh fit
    # loc, scale = stats.rayleigh.fit(angle)
    # xl = np.linspace(angle.min(), angle.max(), 100) # linspace for plotting

    # Calculate 68 %
    #index_at_68 = int(0.68 * N)
    #angle_68 = np.sort(angle_difference_data)[index_at_68]
    # ------------------------------------------

    return energy

def get_pred_direction_diff_data(run_name, do_return_data=False):
    prediction_file = f'{models_dir(run_name)}/model.{run_name}.h5_predicted.pkl'
    with open(prediction_file, "br") as fin:
        zenith, azimuth, zenith_pred, azimuth_pred = pickle.load(fin)

    # Remove extra dimension of array (it comes from the model)
    azimuth_pred = np.squeeze(azimuth_pred)
    zenith_pred = np.squeeze(zenith_pred)

    # Only pick first 100000 data
    # N = 100000
    # nu_direction_predict = nu_direction_predict[:N]
    # nu_direction = nu_direction[:N]
    direction_pred = hp.spherical_to_cartesian(zenith_pred, azimuth_pred)
    direction = hp.spherical_to_cartesian(zenith, azimuth)
    direction_diff = np.array([ hp.get_angle(direction[i], direction_pred[i]) for i in range(len(direction))]) /units.deg


    '''
    if do_return_data:
        return energy_difference_data, shower_energy_log10_predict, shower_energy_log10
    else:
        return energy_difference_data
    '''
    return direction_diff, azimuth_pred, zenith_pred, azimuth, zenith

def find_68_interval(run_name):
    energy_difference_data = get_pred_direction_diff_data(run_name)

    energy_68 = calculate_percentage_interval(energy_difference_data, 0.68)

    return energy_68




def get_histogram2d(x=None, y=None, z=None,
                bins=10, range=None,
                xscale="linear", yscale="linear", cscale="linear",
                normed=False, cmap=None, clim=(None, None),
                ax1=None, grid=True, shading='flat', colorbar={},
                cbi_kwargs={'orientation': 'vertical'},
                xlabel="", ylabel="", clabel="", title="",
                fname="hist2d.png"):
    """
    creates a 2d histogram
    Parameters
    ----------
    x, y, z :
        x and y coordinaten for z value, if z is None the 2d histogram of x and z is calculated
    numpy.histogram2d parameters:
        range : array_like, shape(2,2), optional
        bins : int or array_like or [int, int] or [array, array], optional
    ax1: mplt.axes
        if None (default) a olt.figure is created and histogram is stored
        if axis is give, the axis and a pcolormesh object is returned
    colorbar : dict
    plt.pcolormesh parameters:
        clim=(vmin, vmax) : scalar, optional, default: clim=(None, None)
        shading : {'flat', 'gouraud'}, optional
    normed: string
        colum, row, colum1, row1 (default: None)
    {x,y,c}scale: string
        'linear', 'log' (default: 'linear')
    """

    if z is None and (x is None or y is None):
        sys.exit("z and (x or y) are all None")

    if ax1 is None:
        fig, ax = plt.subplots(1)
        fig.subplots_adjust(wspace=0.3, left=0.05, right=0.95)
    else:
        ax = ax1

    if z is None:
        z, xedges, yedges = np.histogram2d(x, y, bins=bins, range=range)
        z = z.T

    else:
        xedges, yedges = x, y

    if normed:
        if normed == "colum":
            z = z / np.sum(z, axis=0)
        elif normed == "row":
            z = z / np.sum(z, axis=1)[:, None]
        elif normed == "colum1":
            z = z / np.amax(z, axis=0)
        elif normed == "row1":
            z = z / np.amax(z, axis=1)[:, None]
        else:
            sys.exit("Normalisation %s is not known.")

    color_norm = mpl.colors.LogNorm() if cscale == "log" else None
    vmin, vmax = clim
    im = ax.pcolormesh(xedges, yedges, z, shading=shading, vmin=vmin, vmax=vmax, norm=color_norm, cmap=cmap)

    if colorbar is not None:
        cbi = plt.colorbar(im, **cbi_kwargs)
        cbi.ax.tick_params(axis='both', **{"labelsize": 14})
        cbi.set_label(clabel)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    ax.set_xscale(xscale)
    ax.set_yscale(yscale)

    ax.set_title(title)

    return fig, ax, im
    

def get_histogram(data, bins=10, xlabel="", ylabel="entries", weights=None,
                  title="", stats=True, show=False, stat_kwargs=None, funcs=None, overflow=True,
                  ax=None, fit=None, kwargs={'facecolor':'0.7', 'alpha':1, 'edgecolor':"k"},
                  figsize=None):
    """ creates a histogram using matplotlib from array """

    N = data.size
    weights = np.ones(N)
    if(ax is None):
        if figsize is None:
            fig, ax1 = plt.subplots(1, 1, figsize=(6, 6))
        else:
            fig, ax1 = plt.subplots(1, 1, figsize=figsize)
    else:
        ax1 = ax

    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel)
    ax1.set_title(title)
    n, bins, patches = ax1.hist(data, bins, weights=weights, **kwargs)
    if(funcs):
        for func in funcs:
            xlim = np.array(ax1.get_xlim())
            xx = np.linspace(xlim[0], xlim[1], 100)
            if('args' in func):
                ax1.plot(xx, func['func'](xx), *func['args'])
                if('kwargs' in func):
                    ax1.plot(xx, func['func'](xx), *func['args'], **func['kwargs'])
            else:
                ax1.plot(xx, func['func'](xx))
    ax1.set_ylim(0, n.max() * 1.2)
    ax1.set_xlim(bins[0], bins[-1])

    if stats:
        if overflow:
            underflow = np.sum(data < bins[0])
            overflow = np.sum(data > bins[-1])
        else:
            underflow = None
            underflow = None
        if(stat_kwargs is None):
            plot_hist_stats(ax1, data, overflow=overflow, underflow=underflow, weights=weights, fit=fit)
        else:
            plot_hist_stats(ax1, data, overflow=overflow, underflow=underflow, weights=weights, **stat_kwargs)
    if(show):
        plt.show()
    if(ax is None):
        return fig, ax1

def plot_hist_stats(ax, data, weights=None, posx=0.65, posy=0.95, overflow=None,
                    underflow=None, rel=False, fit=None,
                    additional_text="", additional_text_pre="",
                    fontsize=12, color="k", va="top", ha="left",
                    median=True, quantiles=True, mean=True, std=False, N=True,
                    single_sided=True):
    data = np.array(data)
    textstr = additional_text_pre
    if (textstr != ""):
        textstr += "\n"
    if N:
        textstr += "$N=%i$\n" % data.size
    if not single_sided:
        tmean = data.mean()
        tstd = data.std()
        if weights is not None:

            def weighted_avg_and_std(values, weights):
                """
                Return the weighted average and standard deviation.

                values, weights -- Numpy ndarrays with the same shape.
                """
                average = np.average(values, weights=weights)
                variance = np.average((values - average) ** 2, weights=weights)  # Fast and numerically precise
                return (average, variance ** 0.5)

            tmean, tstd = weighted_avg_and_std(data, weights)

    #     import SignificantFigures as serror
        if mean:
            if weights is None:
    #             textstr += "$\mu = %s \pm %s$\n" % serror.formatError(tmean,
    #                                                 tstd / math.sqrt(data.size))
                textstr += "$\mu = {:.3g}$\n".format(tmean)
            else:
                textstr += "$\mu = {:.3g}$\n".format(tmean)
        if median:
            tweights = np.ones_like(data)
            if weights is not None:
                tweights = weights
            if quantiles:
                q1 = stats.quantile_1d(data, tweights, 0.16)
                q2 = stats.quantile_1d(data, tweights, 0.84)
                median = stats.median(data, tweights)
    #             median_str = serror.formatError(median, 0.05 * (np.abs(median - q2) + np.abs(median - q1)))[0]
                textstr += "$\mathrm{median} = %.3g^{+%.2g}_{-%.2g}$\n" % (median, np.abs(median - q2),
                                                                           np.abs(median - q1))
            else:
                textstr += "$\mathrm{median} = %.3g $\n" % stats.median(data, tweights)
        if std:
            if rel:
                textstr += "$\sigma = %.2g$ (%.1f\%%)\n" % (tstd, tstd / tmean * 100.)
            else:
                textstr += "$\sigma = %.2g$\n" % (tstd)
    else:
        if(weights is None):
            w = np.ones_like(data)
        else:
            w = weights
        q68 = stats.quantile_1d(data, weights=w, quant=.68)
        q95 = stats.quantile_1d(data, weights=w, quant=.95)
        textstr += "$\sigma_\mathrm{{68}}$ = {:.1f}$^\circ$\n".format(q68)

    if(overflow):
        textstr += "$\mathrm{overflows} = %i$\n" % overflow
    if(underflow):
        textstr += "$\mathrm{underflows} = %i$\n" % underflow
    textstr += 'Moffat/King fit\n'
    textstr += '$ \sigma = %.2g$\n' % (fit[1])
    textstr += '$ \gamma = %.2g$\n' % (fit[2])

    textstr += additional_text
    textstr = textstr[:-1]

    props = dict(boxstyle='square', facecolor='w', alpha=0.5)
    ax.text(posx, posy, textstr, transform=ax.transAxes, fontsize=fontsize,
            verticalalignment=va, ha=ha, multialignment='left',
            bbox=props, color=color)
