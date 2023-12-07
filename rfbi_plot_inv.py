#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 10:52:15 2023

@author: Emile DENISE
"""

import os
import argparse
import configparser
import numpy as np
import pandas as pd
import scipy.stats as ss
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
import matplotlib.ticker as ti
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerTuple
import seaborn as sb
import colorsys
from pyraysum import prs

# %% Read arguments and check

parser = argparse.ArgumentParser(description='Plot inversion results')
parser.add_argument('config', help='Path to config file.')
args = parser.parse_args()

path_config = args.config

if not os.path.isfile(path_config):
    msg = "The config file does not exist."
    raise NameError(msg)

config = configparser.ConfigParser()
config.read(path_config)

wkdir = config['INPUT']['wkdir']
datadir = wkdir + '/data/'
outdir = wkdir + '/models/'

filter_freq = tuple(map(float, config['INVERSION SETUP']['filter_freq'].split(',')))
dt = config.getfloat('INVERSION SETUP', 'dt')
npts = config.getint('INVERSION SETUP', 'npts')
n_params = config.getint('INVERSION SETUP', 'n_params')
n_layers = config.getint('STRUCTURE SETUP', 'n_layers')

sampling = config['INVERSION SETUP']['sampling']

# %% Plots directory

plotdir = wkdir + "/figures"

if not os.path.exists(plotdir):
    os.makedirs(plotdir)


# %% Functions

def make_masked_array_nan(array):
    return np.ma.array(array, mask=np.isnan(array))


def find_idx_nearest(array, values):
    """
    Array: array where to find the nearest values to values
    values : list of values to find the nearest elements in array
    """

    idx_nearest = []
    
    for v in values:
        idx = np.argmin(np.abs(np.array(array)-v))
        idx_nearest.append(idx)

    return idx_nearest


def find_values_idx(array, idx):
    """
    Array: array where to find the desired values 
    idx : list of indexes (possibly containing nans) to find elements in array
    """
    values = []

    for i in idx:
        if np.isfinite(i):
            values.append(array[i])
        else:
            values.append(np.nan)
    
    return values


def extract_RF_timepolamp(result, phases2extract):
    """
    result is the result of run with run_full (RTZ rotation)
    Returns time peaks, their amplitudes, polarities, and the corresponding phase names with different naming systems.
    """

    n_RF = len(result.rfs)
    t_RF = result.rfs[0][0].stats.taxis

    t_arrival = []
    pol_trans = []
    amp_trans = []

    for k in range(n_RF):

        t_arrival.append([])
        pol_trans.append([])
        amp_trans.append([])

        # Arrival times of converted waves PS, PpS, PsS

        if 'P' in result.rfs[k][0].stats.conversion_names:
        
            t_P = result.rfs[k][0].stats.phase_times[result.rfs[k][0].stats.conversion_names == 'P'][0]

            for phase in phases2extract:

                if phase in result.rfs[k][0].stats.conversion_names:

                    t_arrival[k].append(result.rfs[k][0].stats.phase_times[result.rfs[k][0].stats.conversion_names == phase][0] - t_P)
                
                else:
                    t_arrival[k].append(np.nan)
        
        else: #si on a pas de P, on met des NaN partout

            t_arrival[k].append(len(phases2extract)*[np.nan])

        idx_arrival = find_idx_nearest(t_RF, t_arrival[k])

        # Amplitude of converted waves PS, PpS, PsS on the transverse component
        amp_trans[k] = find_values_idx(result.rfs[k][1].data, idx_arrival)

        # Polarity of converted waves PS, PpS, PsS on the transverse component
        pol_trans[k] = np.array(amp_trans[k]) / np.abs(amp_trans[k])

    return np.array(t_arrival), np.array(pol_trans), np.array(amp_trans)


def predict_RF(struct, geomray, run_control, freq_filt, phases2extract):

    result = prs.run(struct, geomray, run_control, rf=True)

    result.filter('rfs', 'bandpass', freqmin=freq_filt[0], freqmax=freq_filt[1], zerophase=True, corners=2)

    rfarray = np.empty((geomray.ntr, 2, run_control.npts))

    for k in range(geomray.ntr):

        rfarray[k, 0, :] = result.rfs[k][0].data
        rfarray[k, 1, :] = result.rfs[k][1].data

    RF_arrival, RF_pol, RF_amp = extract_RF_timepolamp(result, phases2extract)

    tau_PS = (RF_arrival[:, 2] - RF_arrival[:, 1])[..., np.newaxis]
    tau_PpS = (RF_arrival[:, 5] - RF_arrival[:, 4])[..., np.newaxis]

    if 'PsS' in phases2extract:
        tau_PsS = (RF_arrival[:, 8] - RF_arrival[:, 7])[..., np.newaxis]
        RF_arrival = np.concatenate((RF_arrival, tau_PpS/tau_PS, tau_PsS/tau_PS), axis=1)
    
    else:
        RF_arrival = np.concatenate((RF_arrival, tau_PpS/tau_PS), axis=1)

    return rfarray, RF_arrival, RF_pol, RF_amp


def read_csv_model(path_csv_model):
    """
    """
    param_list = pd.read_csv(path_csv_model, sep=';')

    n_inv = np.sum(param_list['param_type'] == 'inv')
    n_set = np.sum(param_list['param_type'] == 'set')
    n_same = np.sum(param_list['param_type'] == 'same')

    N = (n_inv, n_set, n_same)

    param_inv = {}
    param_set = {}
    param_same = {}

    for k in range(len(param_list)):
    
        line = param_list.iloc[k]
    
        if line['param_type'] == 'inv':
            param_inv[(line['param_name'], line['nb_layer'])] = (line['param_info1'], tuple(line[['param_info2', 'param_info3']].values))
    
        elif line['param_type'] == 'set':
            param_set[(line['param_name'], line['nb_layer'])] = float(line['param_info1'])
    
        elif line['param_type'] == 'same':
            param_same[(line['param_name'], line['nb_layer'])] = (line['param_info1'], int(line['param_info2']))

        else:
            print('error')

    param_set_list = list(param_set.keys())
    param_set_values = [param_set.get(p) for p in param_set_list]

    param_inv_list = list(param_inv.keys())
    param_inv_bounds = [param_inv.get(p)[1] for p in param_inv_list]

    param_same_list = list(param_same.keys())
    param_same_link = [param_same.get(p) for p in param_same_list]

    return param_set_list, param_set_values, param_inv_list, param_inv_bounds, param_same_list, param_same_link


def gen_struct_from_invsetsame(n_layers, param_set_list, param_set_values, param_inv_list, param_inv_values, param_same_list, param_same_link):
    thickn = np.full(n_layers, np.nan)
    rho = np.full(n_layers, np.nan)
    vp = np.full(n_layers, np.nan)
    vpvs = np.full(n_layers, np.nan)
    strike = np.full(n_layers, np.nan)
    dip = np.full(n_layers, np.nan)
     
    struct = prs.Model(thickn=thickn,
                       rho=rho,
                       vp=vp,
                       vpvs=vpvs,
                       strike=strike,
                       dip=dip)

    for k in range(len(param_set_list)):
        struct[int(param_set_list[k][1]), str(param_set_list[k][0])] = float(param_set_values[k])

    for k in range(len(param_inv_list)):
        struct[int(param_inv_list[k][1]), str(param_inv_list[k][0])] = float(param_inv_values[k])

    for k in range(len(param_same_list)):
        struct[int(param_same_list[k][1]), str(param_same_list[k][0])] = float(struct[param_same_link[k][1], param_same_link[k][0]])

    return struct


def adjust_lightness(color, amount):
    try:
        c = colors.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*colors.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])


def plot_mesh_rf(rfarray, baz, slow, dt, **kwargs):
    idx_sort = np.argsort(baz)
    n_rf = rfarray.shape[0]
    npts = rfarray.shape[2]

    fig, ax = plt.subplots(3, 2, height_ratios=[3, 1, 1], figsize=(12, 6), sharex=True, sharey='row')

    minmax = np.nanmax(np.abs(rfarray))
    norm  = colors.Normalize(vmin=-minmax, vmax=minmax)
    cmap = 'seismic_r'
    mappable = cmx.ScalarMappable(norm=norm, cmap=cmap)
    
    time = np.array(range(npts))*dt-npts*dt/2
    
    ax[0, 0].set_title('RF radial')
    ax[0, 0].pcolormesh(list(range(n_rf)), time, rfarray[idx_sort, 0, :].T, norm=norm, cmap=cmap)
    
    ax[0, 1].set_title('RF transverse')
    ax[0, 1].pcolormesh(list(range(n_rf)), time, rfarray[idx_sort, 1, :].T, norm=norm, cmap=cmap)

    if 'RF_arrival' in kwargs:
        RF_arrival = np.array(kwargs.get('RF_arrival'))
        tcolors = 3*['yellow'] + 3*['lime'] + 3*['magenta']

        for k in range(RF_arrival.shape[1]):
            ax[0, 0].scatter(list(range(n_rf)), RF_arrival[idx_sort, k], c=tcolors[k], s=1)
            ax[0, 1].scatter(list(range(n_rf)), RF_arrival[idx_sort, k], c=tcolors[k], s=1)

    ax[1, 0].set_ylabel('Back-azimuth (°)')
    ax[1, 0].set_ylim(0, 360)
    ax[1, 0].yaxis.set_major_locator(ti.MultipleLocator(180))
    ax[1, 0].yaxis.set_minor_locator(ti.MultipleLocator(90))
    
    ax[1, 0].scatter(list(range(n_rf)), baz[idx_sort], c='k', s=1)
    ax[1, 1].scatter(list(range(n_rf)), baz[idx_sort], c='k', s=1)
    
    ax[2, 0].set_ylabel('Slowness\n' + r'($s.km^{-1}$)')
    ax[2, 0].set_ylim(.04, .08)
    ax[2, 0].yaxis.set_major_locator(ti.MultipleLocator(.02))
    ax[2, 0].yaxis.set_minor_locator(ti.MultipleLocator(.01))
    
    ax[2, 0].scatter(list(range(n_rf)), slow[idx_sort], c='k', s=1)
    ax[2, 1].scatter(list(range(n_rf)), slow[idx_sort], c='k', s=1)

    for i in range(1, 3):
        for j in range(2):
            ax[i, j].grid(which='both', axis='y', ls='--', color='lightgrey', zorder=-10)
            ax[i, j].grid(which='major', axis='x', ls='--', color='lightgrey', zorder=-10)
    
    ax[0, 0].set_ylabel('Time (s)')
    ax[0, 0].set_ylim(-2.5, 35)
    ax[0, 0].yaxis.set_major_locator(ti.MultipleLocator(5))
    ax[0, 0].yaxis.set_minor_locator(ti.MultipleLocator(2.5))
    ax[0, 0].invert_yaxis()
    
    ax[-1, 0].set_xlabel('Event #')
    ax[-1, 1].set_xlabel('Event #')

    ax[-1, 0].xaxis.set_major_locator(ti.MultipleLocator(20))
    ax[-1, 0].xaxis.set_minor_locator(ti.MultipleLocator(10))

    ax[-1, 1].xaxis.set_major_locator(ti.MultipleLocator(20))
    ax[-1, 1].xaxis.set_minor_locator(ti.MultipleLocator(10))
    
    fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=.03, hspace=.08)

    if 'path_figure' in kwargs:
        fig.savefig(kwargs.get('path_figure'), dpi=300, bbox_inches='tight')
    
    return fig, ax


def plot_stack_rf(rfarray, baz, slow, dt, alpha=1., **kwargs):
    idx_sort = np.argsort(baz)
    n_rf = rfarray.shape[0]
    npts = rfarray.shape[2]

    fig, ax = plt.subplots(1, 4, width_ratios=[3, 1, 1, 3], figsize=(12, 10), sharey=True)

    time = np.array(range(npts))*dt-npts*dt/2

    minmax = np.nanmax(np.abs(rfarray))
    rfarray /= minmax
    fact = 5

    if 'RF_arrival' in kwargs:
        RF_arrival = np.array(kwargs.get('RF_arrival'))
        tcolors = np.tile(['yellow', 'lime', 'magenta'], 3)
        
    for k, i in enumerate(idx_sort):

        ax[0].plot(time, k+fact*rfarray[i, 0, :], c='k', lw=.5)
        ax[0].fill_between(time, k+fact*rfarray[i, 0, :], npts*[k], where=rfarray[i, 0, :]<0, color='r', alpha=alpha)
        ax[0].fill_between(time, k+fact*rfarray[i, 0, :], npts*[k], where=rfarray[i, 0, :]>=0, color='b', alpha=alpha)

        ax[3].plot(time, k+fact*rfarray[i, 1, :], c='k', lw=.5)
        ax[3].fill_between(time, k+fact*rfarray[i, 1, :], npts*[k], where=rfarray[i, 1, :]<0, color='r', alpha=alpha)
        ax[3].fill_between(time, k+fact*rfarray[i, 1, :], npts*[k], where=rfarray[i, 1, :]>=0, color='b', alpha=alpha)

        if 'RF_arrival' in kwargs:

            ax[0].scatter(RF_arrival[i], 9*[k], c=tcolors, s=1)
            ax[3].scatter(RF_arrival[i], 9*[k], c=tcolors, s=1)

    ax[0].set_title('RF radial')
    ax[3].set_title('RF transverse')

    for k in [0, 3]:
        ax[k].spines['top'].set_visible(False)
        ax[k].spines['right'].set_visible(False)

    ax[1].set_xlabel('Back-azimuth (°)')
    ax[1].set_xlim(0, 360)
    ax[1].xaxis.set_major_locator(ti.MultipleLocator(180))
    ax[1].xaxis.set_minor_locator(ti.MultipleLocator(90))
    ax[1].grid(which='both', axis='x', ls='--', color='lightgrey', zorder=-10)
    ax[1].grid(which='major', axis='y', ls='--', color='lightgrey', zorder=-10)
    plt.setp(ax[1].xaxis.get_majorticklabels(), ha="right")
    
    ax[1].scatter(baz_data[idx_sort], list(range(n_rf)), c='k', s=1)
    ax[1].scatter(baz_data[idx_sort], list(range(n_rf)), c='k', s=1)
    
    ax[2].set_xlabel('Slowness\n' + r'($s.km^{-1}$)')
    ax[2].set_xlim(.04, .08)
    ax[2].xaxis.set_major_locator(ti.MultipleLocator(.02))
    ax[2].xaxis.set_minor_locator(ti.MultipleLocator(.01))
    ax[2].grid(which='both', axis='x', ls='--', color='lightgrey', zorder=-10)
    ax[2].grid(which='major', axis='y', ls='--', color='lightgrey', zorder=-10)
    plt.setp(ax[2].xaxis.get_majorticklabels(), ha='left')
    
    ax[2].scatter(slow_data[idx_sort], list(range(n_rf)), c='k', s=1)
    ax[2].scatter(slow_data[idx_sort], list(range(n_rf)), c='k', s=1)
    
    ax[0].set_xlabel('Time (s)')
    ax[0].set_xlim(-2.5, 35)
    ax[0].xaxis.set_major_locator(ti.MultipleLocator(5))
    ax[0].xaxis.set_minor_locator(ti.MultipleLocator(2.5))
    plt.setp(ax[0].xaxis.get_majorticklabels(), ha="right")

    ax[3].set_xlabel('Time (s)')
    ax[3].set_xlim(-2, 45)
    ax[3].xaxis.set_major_locator(ti.MultipleLocator(5))
    ax[3].xaxis.set_minor_locator(ti.MultipleLocator(2.5))
    plt.setp(ax[3].xaxis.get_majorticklabels(), ha="left")
    
    ax[0].set_ylabel('Event #')
    ax[0].set_ylim(-1, n_rf+2.5)
    ax[0].yaxis.set_major_locator(ti.MultipleLocator(20))
    ax[0].yaxis.set_minor_locator(ti.MultipleLocator(10))

    fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=.03, hspace=.08)

    if 'path_figure' in kwargs:
        fig.savefig(kwargs.get('path_figure'), dpi=300, bbox_inches='tight')

    return fig, ax


def plot_dataVSmodel(data, prediction, baz, slow, title, phases2extract, tmax=45, **kwargs):
    idx_sort = np.argsort(baz)

    RF_arrival_d, RF_sigarrival_d, RF_pol_d, RF_amp_d = data
    RF_arrival_m, RF_pol_m, RF_amp_m = prediction
    
    n_rf = RF_arrival_d.shape[0]

    fig, ax = plt.subplots(ncols=2, nrows=11, figsize=(16, 8), sharex=True, sharey='row')

    gs = ax[0, 0].get_gridspec()

    if RF_arrival_d.shape[1] > len(phases2extract):

        for a in ax[:6, 0]:
            a.remove()
        axbig = fig.add_subplot(gs[:6, 0])

        for a in ax[6:9, 0]:
            a.remove()
        axtau = fig.add_subplot(gs[6:9, 0])

        axtau.xaxis.set_major_locator(ti.MultipleLocator(20))
        axtau.xaxis.set_minor_locator(ti.MultipleLocator(10))

        axtau.xaxis.set_tick_params(labelbottom=False)
        axtau.set_ylabel(r'$\dfrac{\tau_i}{\tau_j}$')

        axtau.grid(which='major', ls='--', color='lightgrey')
        axtau.set_ylim(0, 10)
        axtau.set_xlim(0, n_rf)
        axtau.set_axisbelow(True)
    
    else:

        for a in ax[:9, 0]:
            a.remove()
        axbig = fig.add_subplot(gs[:9, 0])

    tcolors = ['red', 'blue', 'green']
    tlight = [.5, .75, 1]
    phase_list = ['PS moho', 'PS slab top', 'PS moho ocean.', 'PpS moho', 'PpS slab top', 'PpS moho ocean.', 'PsS moho', 'PsS slab top', 'PsS moho ocean.']

    minmax = np.nanmax([np.nanmax(np.abs(RF_amp_d)), np.nanmax(np.abs(RF_amp_m))])

    for k in range(len(phases2extract)):

        if k >= RF_arrival_d.shape[1]:
            ax[8-k, 1].axis('off')
        else:
            axbig.errorbar(x=list(range(n_rf)),
                           y=RF_arrival_d[idx_sort, k],
                           yerr=RF_sigarrival_d[idx_sort, k],
                           xerr=None,
                           fmt='.', c=adjust_lightness(tcolors[k%3], amount=tlight[k//3]), zorder=1, markersize=5, elinewidth=.5, capsize=1.8, ecolor='lightgrey')

            axbig.scatter(list(range(n_rf)), RF_arrival_m[idx_sort, k], s=70, color=adjust_lightness(tcolors[k%3], amount=tlight[k//3]),
                          alpha=.25, zorder=0, edgecolor='None')

            ax[8-k, 1].pcolormesh(np.vstack((RF_amp_d[:, k], RF_amp_m[:, k])), cmap='bwr_r', vmin=-minmax, vmax=minmax)
            ax[8-k, 1].yaxis.set_tick_params(left=False)
            ax[8-k, 1].plot([0, n_rf], 2*[1.], ls='--', lw=1, c='k')

            ax[8-k, 1].text(.99, .9, phase_list[k], transform=ax[8-k, 1].transAxes, va='top', ha='right')

    if RF_arrival_d.shape[1] > len(phases2extract):

        for k in range(len(phases2extract), RF_arrival_d.shape[1]):

            axtau.scatter(list(range(n_rf)), RF_arrival_d[idx_sort, k], s=5, color=adjust_lightness('k', amount=1), zorder=5)
            axtau.scatter(list(range(n_rf)), RF_arrival_m[idx_sort, k], s=70, color=adjust_lightness('k', amount=.5), alpha=.25, zorder=4, edgecolor='None')
    
    ax[8, 1].text(.01, .25, 'Data', transform=ax[8, 1].transAxes, va='center', ha='left')
    ax[8, 1].text(.01, .75, 'Prediction', transform=ax[8, 1].transAxes, va='center', ha='left')

    axbig.set_ylim(0, tmax)
    axbig.set_xlim(0, n_rf)
    ax[-1, 0].set_xlim(0, n_rf)

    ax[-2, 0].set_ylabel('Back\nazimuth (°)')
    ax[-2, 0].set_ylim(0, 360)
    ax[-2, 0].yaxis.set_major_locator(ti.MultipleLocator(180))
    ax[-2, 0].yaxis.set_minor_locator(ti.MultipleLocator(90))
    
    ax[-2, 0].scatter(list(range(n_rf)), baz[idx_sort], c='k', s=1)
    ax[-2, 1].scatter(list(range(n_rf)), baz[idx_sort], c='k', s=1)
    
    ax[-1, 0].set_ylabel('Slowness\n' + r'($s.km^{-1}$)')
    ax[-1, 0].set_ylim(.04, .08)
    ax[-1, 0].yaxis.set_major_locator(ti.MultipleLocator(.02))
    ax[-1, 0].yaxis.set_minor_locator(ti.MultipleLocator(.01))
    
    ax[-1, 0].scatter(list(range(n_rf)), slow[idx_sort], c='k', s=1)
    ax[-1, 1].scatter(list(range(n_rf)), slow[idx_sort], c='k', s=1)

    for i in range(-2, 0):
        for j in range(2):
            ax[i, j].grid(which='both', axis='y', ls='--', color='lightgrey')
            ax[i, j].grid(which='major', axis='x', ls='--', color='lightgrey')
            ax[i, j].set_axisbelow(True)
    
    ax[-1, 0].set_xlabel('Event #')
    ax[-1, 1].set_xlabel('Event #')

    ax[-1, 0].xaxis.set_major_locator(ti.MultipleLocator(20))
    ax[-1, 0].xaxis.set_minor_locator(ti.MultipleLocator(10))

    ax[-1, 1].xaxis.set_major_locator(ti.MultipleLocator(20))
    ax[-1, 1].xaxis.set_minor_locator(ti.MultipleLocator(10))

    axbig.xaxis.set_major_locator(ti.MultipleLocator(20))
    axbig.xaxis.set_minor_locator(ti.MultipleLocator(10))

    axbig.xaxis.set_tick_params(labelbottom=False)
    axbig.set_ylabel('Arrival time (s)')

    legend_elements = [tuple([Line2D([0], [0], marker='o', color='none', markersize=5, markerfacecolor=adjust_lightness(tcolors[k], amount=tlight[i]), markeredgecolor='none') for i in range(3)] +
                             [Line2D([0], [0], marker='o', color='none', markersize=10, alpha=.25, markerfacecolor=adjust_lightness(tcolors[k], amount=tlight[i]), markeredgecolor='none') for i in range(3)]) for k in range(3)]
    
    axbig.legend(legend_elements, ['Moho converted waves (PS/PpS/PsS)', 'Slab top converted waves (PS/PpS/PsS)', 'ocean. Moho converted waves (PS/PpS/PsS)'],
              handler_map={tuple: HandlerTuple(ndivide=6)}, loc='upper right')

    axbig.set_title('Converted waves arrival times', fontweight='bold')

    axbig.grid(which='major', ls='--', color='lightgrey')
    axbig.set_axisbelow(True)

    ax[0, 1].set_title('Converted waves amplitudes (transverse component)', fontweight='bold')
    
    fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=.02, hspace=.2)
    
    fig.savefig(plotdir + '/' + title + '_dataVSmodel.jpeg', dpi=300, bbox_inches='tight')
    plt.close(fig)


def gen_phaselist2extract(n_layers, waves):
    """
    n_layers number of layers
    waves list among ['PS', 'PpS', 'PsS']
    """
    phaselist = []

    if 'PS' in waves:
        phaselist += ['P%iS' %k for k in range(1, n_layers)]
    
    if 'PpS' in waves:
        phaselist += ['P0p%iS' %k for k in range(1, n_layers)]
    
    if 'PsS' in waves:
        phaselist += ['P0s%iS' %k for k in range(1, n_layers)]

    return phaselist


def plot_evol_model(z_accepted, name, plotdir):

    nrows = np.unique([b[0] for b in param_inv_list]).size
    ncols = n_layers
    n_metro = len(z_accepted)

    if nrows == 1:
        fig, ax = plt.subplots(nrows=nrows+1, ncols=ncols, sharex=True, figsize=(4*ncols, 2*(nrows+1)))
    else:
        fig, ax = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, figsize=(4*ncols, 2*nrows)) 

    ax[0, 0].set_xlim(0, n_metro)

    for j in range(ncols):

        if nrows == 1:
            ax[nrows, j].axis('off')

        for i in range(nrows):

            if (np.unique([b[0] for b in param_inv_list])[i], j) in param_inv_list:

                idx = [x for x, y in enumerate(param_inv_list) if y[0] == np.unique([b[0] for b in param_inv_list])[i] and y[1] == j][0]

                ax[i, j].set_title(param_inv_list[idx][0] + ' %i' %param_inv_list[idx][1])

                ax[i, j].plot(list(range(len(z_accepted))), np.array(z_accepted)[:, idx], c='slateblue', zorder=10)
                # ax[i, j].plot(list(range(len(z_rejected))),np.array(z_rejected)[:, idx], c='firebrick', zorder=9, alpha=.5)

                ax[i, j].plot([0, n_metro], 2*[param_inv_bounds[idx][0]], ls='--', c='k')
                ax[i, j].plot([0, n_metro], 2*[param_inv_bounds[idx][1]], ls='--', c='k')

                if i == nrows:
                    ax[i, j].set_xlabel('accepted model #')

                #ax[i, j].plot(2*[N_burn], [param_inv_bounds[idx][0] - (param_inv_bounds[idx][1]-param_inv_bounds[idx][0])/5, param_inv_bounds[idx][1] + (param_inv_bounds[idx][1]-param_inv_bounds[idx][0])/5])
                #ax[i, j].set_ylim(param_inv_bounds[idx][0] - (param_inv_bounds[idx][1]-param_inv_bounds[idx][0])/5, param_inv_bounds[idx][1] + (param_inv_bounds[idx][1]-param_inv_bounds[idx][0])/5)

            else:
                ax[i, j].axis('off')
    


    fig.savefig(plotdir + '/evolution_' + name + '_models.jpeg', dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_pdf(z_accepted, z_rejected, plotdir):
    nrow = 4
    ncol = 2

    fig, ax = plt.subplots(nrows=nrow, ncols=ncol, sharex='col', figsize=(6*ncol, 3*nrow))#, sharey='row')

    pdf_names = ['log a posteriori pdf', 'log arrival time likelihood', 'log polarity likelihood']
    c = np.array([[adjust_lightness('r', amount=a), adjust_lightness('b', a)] for a in [1, .75, .4]])

    for k in range(3):

        ax[k, 0].plot(list(range(len(z_accepted))), z_accepted[:, -k-1], c=c[k, 0], lw=.8, label=pdf_names[k])
        ax[k, 0].grid(ls='--', c='lightgrey')
        ax[k, 0].legend(loc='lower right')
        ax[k, 0].set_ylabel('log pdf')

        ax[k, 1].plot(list(range(len(z_rejected))), z_rejected[:, -k-1], c=c[k, 1], lw=.8, label=pdf_names[k])
        ax[k, 1].grid(ls='--', c='lightgrey')
        ax[k, 1].legend(loc='lower right')

    ax[-1, 0].plot(list(range(len(z_accepted))), np.abs(z_accepted[:, -2]-np.nanmean(z_accepted[:, -2])), lw=.8, label='abs demeaned log arrival time likelihood', c=c[1, 0])
    ax[-1, 0].plot(list(range(len(z_accepted))), np.abs(z_accepted[:, -3]-np.nanmean(z_accepted[:, -3])), lw=.8, label='abs demeaned log polarity likelihood', c=c[2, 0])
    ax[-1, 0].legend(loc='upper right')
    ax[-1, 0].grid(ls='--', c='lightgrey')

    ax[-1, 1].plot(list(range(len(z_rejected))), np.abs(z_rejected[:, -2]-np.nanmean(z_rejected[:, -2])), lw=.8, label='abs demeaned log arrival time likelihood', c=c[1, 1])
    ax[-1, 1].plot(list(range(len(z_rejected))), np.abs(z_rejected[:, -3]-np.nanmean(z_rejected[:, -3])), lw=.8, label='abs demeaned log polarity likelihood', c=c[2, 1])
    ax[-1, 1].legend(loc='upper right')
    ax[-1, 1].grid(ls='--', c='lightgrey')

    ax[-1, 0].set_xlim(0, len(z_accepted)-1)
    ax[-1, 0].set_xlabel('accepted model #')

    ax[-1, 1].set_xlim(0, len(z_rejected)-1)
    ax[-1, 1].set_xlabel('rejected model #')

    ax[0, 0].set_title('Accepted models', fontweight='bold')
    ax[0, 1].set_title('Rejected models', fontweight='bold')

    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=.05, hspace=.1)

    fig.savefig(plotdir + '/evolution_pdf.jpeg', dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_marginals(z_accepted, plotdir, param_inv_list, param_inv_bounds):
    n = len(param_inv_list)
    param_with_unit = [param_inv_list[k][0] + ' %i' %param_inv_list[k][1] for k in range(n)]

    if n == 1:
        fig, ax = plt.subplots(nrows=n+1, ncols=n+1, figsize=(12, 12))
    else:
        fig, ax = plt.subplots(nrows=n, ncols=n, figsize=(12, 12))
    
    fig.tight_layout(pad=1)
        
    for i in range(n):
        for j in range(i+1):

            if i==j:
                isamples = [s[i] for s in z_accepted[:, :-4]]

                sb.histplot(isamples, ax=ax[i, j], stat='density', color='lightgrey')
                sb.kdeplot(isamples, color='darkmagenta', ax=ax[i, j])

                ax[i, j].spines['right'].set_visible(False)
                ax[i, j].spines['top'].set_visible(False)
                ax[i, j].tick_params(left=False, right=False , labelleft=False)
                ax[i, j].set_title(param_with_unit[j], fontweight='bold')
                ax[i, j].set_ylabel(' ')
                ax[i, j].set_box_aspect(1)
                ax[i, j].xaxis.set_major_locator(plt.MaxNLocator(2))

                ax[i, j].set_xlim(param_inv_bounds[j][0], param_inv_bounds[j][1])

            else:
                isamples = [s[i] for s in z_accepted[:, :-4]]
                jsamples = [s[j] for s in z_accepted[:, :-4]]
 
                # Extract x and y
                x = jsamples
                y = isamples
                # Define the borders
                minx, maxx = param_inv_bounds[j][0], param_inv_bounds[j][1]
                miny, maxy = param_inv_bounds[i][0], param_inv_bounds[i][1]
                deltaX = (maxx - minx)/10
                deltaY = (maxy - miny)/10
                xmin = minx - deltaX
                xmax = maxx + deltaX
                ymin = miny - deltaY
                ymax = maxy + deltaY
                # Create meshgrid
                xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]

                positions = np.vstack([xx.ravel(), yy.ravel()])
                values = np.vstack([x, y])
                kernel = ss.gaussian_kde(values)
                f = np.reshape(kernel(positions).T, xx.shape)

                ax[i, j].contourf(xx, yy, f, levels=15, cmap='plasma')

                ax[j, i].axis('off')
                ax[i, j].xaxis.set_major_locator(plt.MaxNLocator(2))
                ax[i, j].yaxis.set_major_locator(plt.MaxNLocator(2))

                ax[i, j].set_xlim(minx, maxx)
                ax[i, j].set_ylim(miny, maxy)

                if j == 0:
                    ax[i, j].set_ylabel(param_with_unit[i])
                if i == n-1:
                    ax[i, j].set_xlabel(param_with_unit[j])

                ax[i, j].set_box_aspect(1)
    
    if n == 1:
        for k in range(n+1):
            ax[-1, k].axis('off')
            ax[k, -1].axis('off')

    fig.savefig(plotdir + '/marginals_pdf.jpeg', dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_struct(struct, zmax, title, plotdir):
    fig = plt.figure(figsize=(10, 5))

    ax1 = fig.add_subplot(1, 4, 1)
    struct.plot_profile(zmax=zmax, ax=ax1)

    ax2 = fig.add_subplot(1, 4, 2)
    struct.plot_layers(zmax=zmax, ax=ax2)

    ax3 = fig.add_subplot(1, 4, (3, 4))
    struct.plot_interfaces(zmax=zmax, ax=ax3)

    fig.savefig(plotdir + '/' + title + '.jpeg', dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_marginals_grid_search(sampling, n_params, n_sample, plotdir, param_inv_list, param_inv_bounds):

    marg_2D = {}
    marg_1D = []
    param_with_unit = [param_inv_list[k][0] + ' %i' %param_inv_list[k][1] for k in range(len(param_inv_bounds))]

    for i in range(n_params):
        for j in range(n_params):

            if i != j:

                marg = np.zeros((n_sample, n_sample))
                vali = np.unique(sampling[:, i], axis=0)
                valj = np.unique(sampling[:, j], axis=0)

                for ivi, vi in enumerate(vali):
                    for ivj, vj in enumerate(valj):

                        f = np.exp(sampling[(sampling[:, i] == vi) | (sampling[:, j] == vj)][:, -1])
                        f = np.nansum(f)
                        # f *= np.nanprod(np.delete(np.nanmax(sampling[(sampling[:, i] == vi) | (sampling[:, j] == vj)], axis=0)[:n_sample] - np.nanmin(sampling[(sampling[:, i] == vi) | (sampling[:, j] == vj)], axis=0)[:n_sample], [i, j]))
                        # f /= len(sampling[(sampling[:, i] == vi) | (sampling[:, j] == vj)][:, -1])

                        marg[ivi, ivj] = f
                
                marg_2D[(i, j)] = (vali, valj, marg)
            
            else:

                marg = []

                for v in np.unique(sampling[:, i], axis=0):

                    f = np.exp(sampling[sampling[:, i] == v][:, -1])
                    f = np.nansum(f)
                    # f *= np.nanprod(np.delete(np.nanmax(sampling[sampling[:, i] == v], axis=0)[:-1] - np.nanmin(sampling[sampling[:, i] == v], axis=0)[:-1], i))
                    # f /= len(sampling[sampling[:, i] == v][:, -1])

                    marg.append([v, f])
                
                marg_1D.append(marg)
    
    marg_1D = np.array(marg_1D)
    max_model = [marg_1D[:, i, 0][k] for k, i in enumerate(np.nanargmax(marg_1D[:, :, -1], axis=1))]
    
    fig, ax = plt.subplots(ncols=n_params, nrows=n_params, figsize=(n_params*3, n_params*3))

    for i in range(n_params):
        for j in range(n_params):

            minx, maxx = param_inv_bounds[j][0], param_inv_bounds[j][1]
            miny, maxy = param_inv_bounds[i][0], param_inv_bounds[i][1]

            ax[i, j].set_box_aspect(1)

            if j < i:

                ax[i, j].contourf(marg_2D[(i, j)][1], marg_2D[(i, j)][0], marg_2D[(i, j)][2], levels=5, cmap='plasma')

                ax[i, j].set_xlim(minx, maxx)
                ax[i, j].set_ylim(miny, maxy)

                if j == 0:
                    ax[i, j].set_ylabel(param_with_unit[i])
                if i == n_params-1:
                    ax[i, j].set_xlabel(param_with_unit[j])

            elif i == j:

                ax[i, i].plot(marg_1D[i][:, 0], marg_1D[i][:, 1])
                ax[i, i].tick_params(left=False, labelleft=False)

                ax[i, j].set_title(param_with_unit[j], fontweight='bold')

                ax[i, j].set_xlim(minx, maxx)

            else:
                ax[i, j].axis('off')
    
    fig.savefig(plotdir + '/marginals_pdf.jpeg', dpi=300, bbox_inches='tight')
    plt.close(fig)

    return max_model


def plot_cov_AMmatrix(C, t0, plotdir):
    step = t0//5
    C = C[..., ::step]

    nrows = np.ceil(np.sqrt(C.shape[-1])).astype('int')
    ncols = np.ceil(np.sqrt(C.shape[-1])).astype('int')

    minmax = np.nanmax(np.abs([np.nanmin(C), np.nanmax(C)]))
    cmap='turbo'
    norm = colors.SymLogNorm(vmin=-minmax, vmax=minmax, linthresh=1e-3)
    mappable = cmx.ScalarMappable(cmap=cmap, norm=norm)

    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 12))

    for k in range(nrows*ncols):
        i, j = k//ncols, k%ncols

        if k < C.shape[-1]:
            ax[i, j].matshow(C[:, :, k], cmap=cmap, norm=norm)
            ax[i, j].tick_params(left=0, top=0, bottom=0, labelleft=0, labeltop=0)
            ax[i, j].text(.99, .99, k*step, transform=ax[i, j].transAxes, va='top', ha='right', fontweight='bold')
        
        if k*step == t0:
            for axis in ['top','bottom','left','right']:
                ax[i, j].spines[axis].set_linewidth(3)
        
        else:
            ax[i, j].axis('off')

    cb_ax = fig.add_axes([.92, 0.15, 0.01, 0.7])

    fig.colorbar(mappable, cax=cb_ax, pad=.01)
    fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=.1, hspace=.1)

    fig.savefig(plotdir + '/proposal_covariance_matrix.jpeg', dpi=300, bbox_inches='tight')
    plt.close(fig)

# %% Load data

time_data =  make_masked_array_nan(pd.read_csv(datadir + '/data_time.csv', sep=';').values[:, 2:])

time_data_sig = make_masked_array_nan(pd.read_csv(datadir + '/data_time_sigma.csv', sep=';').values[:, 2:])

pol_data = make_masked_array_nan(pd.read_csv(datadir + '/data_pol_trans.csv', sep=';').values[:, 2:])

# RF_pol_gamma = pd.read_csv(datadir + '/data_pol_trans_gamma.csv', sep=';')
# pol_data_gamma = make_masked_array_nan(RF_pol_gamma.values[:, 2:])

amp_data = make_masked_array_nan(pd.read_csv(datadir + '/data_amp_trans.csv', sep=';').values[:, 2:])

baz_data = pd.read_csv(datadir + '/data_time.csv', sep=';').values[:, 0]
slow_data = pd.read_csv(datadir + '/data_time.csv', sep=';').values[:, 1]

geomray_data = prs.Geometry(baz=baz_data, slow=slow_data)

# %% Load run conditions

param_set_list, param_set_values, param_inv_list, param_inv_bounds, param_same_list, param_same_link = read_csv_model(path_csv_model=wkdir+'/parameters_inversion.csv')

run_control = prs.Control(
    verbose=False,
    rot='RTZ',
    dt=dt,
    npts=npts,
    align=0,
    mults=2
)

# %% Plots

phases2extract = config['INVERSION SETUP']['target_phases'].split(',')
phases2extract = gen_phaselist2extract(n_layers, phases2extract)

if sampling in ['metropolis', 'adaptative_metropolis']:

    z_accepted = np.load(outdir + '/accepted_models.npy')
    z_rejected = np.load(outdir + '/rejected_models.npy')

    mean_model = np.nanmean(z_accepted[:, :n_params], axis=0)
    median_model = np.nanmedian(z_accepted[:, :n_params], axis=0)
    max_model = z_accepted[np.nanargmax(z_accepted[:, -1]), :n_params]

    struct = gen_struct_from_invsetsame(n_layers,
                                        param_set_list, param_set_values,
                                        param_inv_list, mean_model,
                                        param_same_list, param_same_link)

    # run avec le mean model pour obtenir la liste des phases à calculer
    result = prs.run(struct, geomray_data, run_control)
    phase_list = result.descriptors()
    run_control.set_phaselist(phase_list, equivalent=True)
    result = phase_list = None

    names = ['mean_model', 'median_model', 'max_model']

    for i, m in enumerate([mean_model, median_model, max_model]):
        struct = gen_struct_from_invsetsame(n_layers,
                                            param_set_list, param_set_values,
                                            param_inv_list, m,
                                            param_same_list, param_same_link)

        plot_struct(struct, 100, names[i], plotdir)

        rfarray, RF_arrival, RF_pol, RF_amp = predict_RF(struct, geomray_data, run_control, filter_freq, phases2extract)

        data = [time_data, time_data_sig, pol_data, amp_data]
        prediction = [RF_arrival, RF_pol, RF_amp]

        plot_dataVSmodel(data, prediction, baz_data, slow_data, names[i], phases2extract, tmax=45)

    plot_evol_model(z_accepted, 'accepted', plotdir)
    plot_evol_model(z_rejected, 'rejected', plotdir)

    plot_pdf(z_accepted[10:], z_rejected, plotdir)

    plot_marginals(z_accepted, plotdir, param_inv_list, param_inv_bounds)

    if sampling =='adaptative_metropolis':
        C = np.load(outdir + '/covariance_proposal.npy')
        t0 = config.getint('INVERSION SETUP', 't0')

        plot_cov_AMmatrix(C, t0, plotdir)

elif sampling == 'grid_search':

    n_sample = config.getint('INVERSION SETUP', 'n_sample')
    sampling = np.load(outdir + '/grid_search_samples.npy')

    max_model = plot_marginals_grid_search(sampling, n_params, n_sample, plotdir, param_inv_list, param_inv_bounds)

    struct = gen_struct_from_invsetsame(n_layers,
                                        param_set_list, param_set_values,
                                        param_inv_list, max_model,
                                        param_same_list, param_same_link)

    plot_struct(struct, 100, 'max_model', plotdir)

    rfarray, RF_arrival, RF_pol, RF_amp = predict_RF(struct, geomray_data, run_control, filter_freq)

    data = [time_data, np.zeros(time_data.shape), pol_data, amp_data]
    prediction = [RF_arrival, RF_pol, RF_amp]

    plot_dataVSmodel(data, prediction, baz_data, slow_data, 'max_model', tmax=45)

exit(0)