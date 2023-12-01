#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 09:59:53 2023

@author: Emile DENISE
"""

import os
import numpy as np
import numpy.random as rd
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
import matplotlib.ticker as ti
import colorsys

from pyraysum import prs


# %% Functions

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


def save_array_synthe(baz, slow, data_array, datadir, header, name):
    idx_sort = np.argsort(baz)
    df = pd.DataFrame(data=np.concatenate((baz[idx_sort][:, np.newaxis], slow[idx_sort][:, np.newaxis], data_array[idx_sort]), axis=1),
                  columns=header)
    df.to_csv(datadir + '/' + name + '.csv', index=False, sep=';')


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


def extract_RF_timepolamp(result):
    """
    result is the result of run with run_full (RTZ rotation)
    Returns time peaks, their amplitudes, polarities, and the corresponding phase names with different naming systems.
    """

    n_RF = len(result.rfs)
    t_RF = result.rfs[0][0].stats.taxis

    t_arrival = []
    pol_trans = []
    amp_trans = []

    phases2extract = ['P1S', 'P2S', 'P3S', 'P0p1S', 'P0p2S', 'P0p3S', 'P0s1S', 'P0s2S', 'P0s3S']

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
        
        else: #si on a pas de phase P, on met des NaN partout

            t_arrival[k].append(len(phases2extract)*[np.nan])

        # Polarity of converted waves PS, PpS, PsS on the transverse component
        idx_arrival = find_idx_nearest(t_RF, t_arrival[k])
        pol_trans[k] = result.rfs[k][1].data[idx_arrival] / np.abs(result.rfs[k][1].data[idx_arrival])

        # Amplitude of converted waves PS, PpS, PsS on the transverse component
        amp_trans[k] = result.rfs[k][1].data[idx_arrival]

    return np.array(t_arrival), np.array(pol_trans), np.array(amp_trans)


def predict_RF(struct, geomray, run_control, freq_filt):

    result = prs.run(struct, geomray, run_control, rf=True)

    result.filter('rfs', 'bandpass', freqmin=freq_filt[0], freqmax=freq_filt[1], zerophase=True, corners=2)

    rfarray = np.empty((geomray.ntr, 2, run_control.npts))

    for k in range(geomray.ntr):

        rfarray[k, 0, :] = result.rfs[k][0].data
        rfarray[k, 1, :] = result.rfs[k][1].data

    RF_arrival, RF_pol, RF_amp = extract_RF_timepolamp(result)

    return rfarray, RF_arrival, RF_pol, RF_amp


def adjust_lightness(color, amount):
    try:
        c = colors.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*colors.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])


def plot_mesh_rf(rfarray, baz, slow, dt, tmax, **kwargs):
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
    ax[0, 0].set_ylim(-2.5, tmax)
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


def plot_stack_rf(rfarray, baz, slow, dt, tmax, alpha=1., **kwargs):
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
    ax[0].set_xlim(-2.5, tmax)
    ax[0].xaxis.set_major_locator(ti.MultipleLocator(5))
    ax[0].xaxis.set_minor_locator(ti.MultipleLocator(2.5))
    plt.setp(ax[0].xaxis.get_majorticklabels(), ha="right")

    ax[3].set_xlabel('Time (s)')
    ax[3].set_xlim(-2, tmax)
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


def plot_dataVSmodel(data, prediction, baz, slow, **kwargs):
    idx_sort = np.argsort(baz)

    RF_arrival_d, RF_sigarrival_d, RF_pol_d, RF_amp_d = data
    RF_arrival_m, RF_pol_m, RF_amp_m = prediction
    
    n_rf = RF_arrival_d.shape[0]

    fig, ax = plt.subplots(ncols=2, nrows=11, figsize=(16, 8), sharex=True, sharey='row')

    gs = ax[0, 0].get_gridspec()
    for a in ax[:9, 0]:
        a.remove()
    axbig = fig.add_subplot(gs[:9, 0])

    tcolors = ['red', 'blue', 'green']
    tlight = [.5, .75, 1]

    for k in range(RF_arrival_d.shape[1]):

        axbig.scatter(list(range(n_rf)), RF_arrival_d[idx_sort, k], s=5, color=adjust_lightness(tcolors[k%3], amount=tlight[k//3]), zorder=5)
        axbig.scatter(list(range(n_rf)), RF_arrival_m[idx_sort, k], s=70, color=adjust_lightness(tcolors[k%3], amount=tlight[k//3]),
                      alpha=.25, zorder=4, edgecolor='None')

        ax[8-k, 1].imshow(np.vstack((RF_pol_d[:, k], RF_pol_m[:, k])), cmap='bwr', vmin=-1, vmax=1)

    axbig.set_ylim(0, 35)
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
            ax[i, j].grid(which='both', axis='y', ls='--', color='lightgrey', zorder=-10)
            ax[i, j].grid(which='major', axis='x', ls='--', color='lightgrey', zorder=-10)
    
    ax[-1, 0].set_xlabel('Event #')
    ax[-1, 1].set_xlabel('Event #')

    ax[-1, 0].xaxis.set_major_locator(ti.MultipleLocator(20))
    ax[-1, 0].xaxis.set_minor_locator(ti.MultipleLocator(10))

    ax[-1, 1].xaxis.set_major_locator(ti.MultipleLocator(20))
    ax[-1, 1].xaxis.set_minor_locator(ti.MultipleLocator(10))

    axbig.xaxis.set_major_locator(ti.MultipleLocator(20))
    axbig.xaxis.set_minor_locator(ti.MultipleLocator(10))

    axbig.set_xticks([])
    axbig.set_ylabel('Arrival time (s)')

    ax[0, 1].set_title('Polarities')
    
    fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=.02, hspace=.2)

    if 'path_figure' in kwargs:
        fig.savefig(kwargs.get('path_figure'), dpi=300, bbox_inches='tight')
    
    return fig, ax

# %% 

datadir = '/Users/emile/Documents/Etude/2023_2024_Cesure/Stage_Mines/inversion_synthetics/data_test/'

if not os.path.isdir(datadir):
    os.makedirs(datadir)

freq_filt = (1/20, 1/2)  # Hz bandpass filter (2-20 s)

thickn_data = [25000, 40000, 9000, 0]
rho_data = [2700, 3000, 2900, 3300]
vp_data = [5000, 7000, 6500, 8000]
vpvs_data = [1.73, 1.73, 1.8, 1.73]
strike_data = [0, 0, 0, 0]
dip_data = [0, 0, 15, 15]

struct_data = prs.Model(thickn=thickn_data,
                        rho=rho_data,
                        vp=vp_data,
                        vpvs=vpvs_data,
                        strike=strike_data,
                        dip=dip_data)

N_data = 150

a = 0
b = 360
baz_data = (b - a) * rd.random_sample(size=N_data) + a

a = .04
b = .08
slow_data = (b - a) * rd.random_sample(size=N_data) + a

geomray_data = prs.Geometry(baz=baz_data, slow=slow_data)

dt = .025
npts = 10000

run_control = prs.Control(
    verbose=False,
    rot='RTZ',
    dt=dt,
    npts=npts,
    align=0,
    mults=2
)

result = prs.run(struct_data, geomray_data, run_control)
phase_list = result.descriptors()
run_control.set_phaselist(phase_list, equivalent=True)
result, phase_list = None, None

RF_data, RF_arrival, RF_pol, RF_amp = predict_RF(struct_data, geomray_data, run_control, freq_filt)


# %% Save data

phase_list = gen_phaselist2extract(struct_data.nlay, ['PS', 'PpS', 'PsS'])

header = ['baz_degrees', 'slowness_s/km'] + phase_list
        #   'PS_moho', 'PS_slab_top', 'PS_ocean_moho',
        #   'PpS_moho', 'PpS_slab_top', 'PpS_ocean_moho',
        #   'PsS_moho', 'PsS_slab_top', 'PsS_ocean_moho']

# Arrival times

tau_PS = (RF_arrival[:, 2] - RF_arrival[:, 1])[..., np.newaxis]
tau_PpS = (RF_arrival[:, 5] - RF_arrival[:, 4])[..., np.newaxis]
# tau_PsS = (RF_arrival[:, 8] - RF_arrival[:, 7])[..., np.newaxis]

err = np.sqrt((0.05/tau_PS)**2 + (0.05/tau_PpS)**2)

save_array_synthe(baz_data, slow_data, np.concatenate((RF_arrival, tau_PpS/tau_PS), axis=1),
                  datadir, header + ['tau'], 'data_time')

save_array_synthe(baz_data, slow_data, np.concatenate((np.full(RF_arrival.shape, .05), err), axis=1),
                  datadir, header + ['tau'], 'data_time_sigma')

# Polarities

save_array_synthe(baz_data, slow_data, RF_pol,
                  datadir, header, 'data_pol_trans')

save_array_synthe(baz_data, slow_data, np.full(RF_pol.shape, 0),
                  datadir, header, 'data_pol_trans_gamma')

# Amplitudes

save_array_synthe(baz_data, slow_data, RF_amp,
                  datadir, header, 'data_amp_trans')

# Structure

plot_struct(struct_data, 100, 'structure_synthe_data', datadir)
struct_data.write(datadir + '/structure_synthe_data.txt')

# Geometry

geomray_data.plot(show=False)
plt.savefig(datadir + '/geometry_rays_data.jpeg', dpi=300, bbox_inches='tight')
plt.close()

geomray_data.write(datadir + '/geometry_rays.txt')

# Plot RF

fig, ax = plot_mesh_rf(RF_data, baz_data, slow_data, dt, tmax=45, RF_arrival=RF_arrival)
fig.savefig(datadir + '/RF_mesh.jpeg', dpi=300, bbox_inches='tight')
plt.close()

fig, ax = plot_stack_rf(RF_data, baz_data, slow_data, dt, tmax=42, alpha=.5, RF_arrival=RF_arrival)
fig.savefig(datadir + '/RF_stack.jpeg', dpi=300, bbox_inches='tight')
plt.close()

exit(0)