#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 10:52:15 2023

@author: Emile DENISE
"""

# %% Packages

import warnings
import numpy as np
import scipy.stats as ss
import scipy.optimize as so
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
import matplotlib.ticker as ti
from matplotlib.legend_handler import HandlerTuple
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import seaborn as sb
import colorsys
from tools.rfbi_tools import *


# %% Functions

def adjust_lightness(color, amount):
    try:
        c = colors.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*colors.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])


def plot_struct(struct, zmax, title, plotdir):
    fig = plt.figure(figsize=(10, 5))

    ax1 = fig.add_subplot(1, 4, 1)
    struct.plot_profile(zmax=zmax, ax=ax1)

    ax2 = fig.add_subplot(1, 4, 2)
    struct.plot_layers(zmax=zmax, ax=ax2)

    ax3 = fig.add_subplot(1, 4, (3, 4))
    struct.plot_interfaces(zmax=zmax, ax=ax3)

    fig.savefig(plotdir + '/' + title + '.png', dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_geomray(geomray, title, plotdir):
    geomray.plot(show=False)
    plt.savefig(plotdir + '/' + title + '.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_RF_mesh(rfarray, baz, slow, taxis, rot, show=False, plotdir=None, **kwargs):
    idx_sort = np.argsort(baz)
    n_rf = rfarray.shape[0]
    npts = rfarray.shape[2]

    fig, ax = plt.subplots(3, 2, height_ratios=[3, 1, 1], figsize=(12, 6), sharex=True, sharey='row')

    minmax = np.nanmax(np.abs(rfarray))
    rfarray /= minmax
    norm  = colors.Normalize(vmin=-1, vmax=1)
    cmap = 'seismic_r'
    mappable = cmx.ScalarMappable(norm=norm, cmap=cmap)
    
    tmax = np.nanmax(taxis)
    if 'tmax' in kwargs:
        tmax = float(kwargs.get('tmax'))
    tmin = np.nanmin(taxis)
    if 'tmin' in kwargs:
        tmin = float(kwargs.get('tmin'))
    
    ax[0, 0].pcolormesh(list(range(n_rf)), taxis, rfarray[idx_sort, 0, :].T, norm=norm, cmap=cmap)
    ax[0, 1].pcolormesh(list(range(n_rf)), taxis, rfarray[idx_sort, 1, :].T, norm=norm, cmap=cmap)

    if rot == 'rtz':
        ax[0, 0].set_title('RF comp. R', fontweight='bold')
        ax[0, 1].set_title('RF comp. T', fontweight='bold')
    elif rot == 'lqt':
        ax[0, 0].set_title('RF comp. Q', fontweight='bold')
        ax[0, 1].set_title('RF comp. T', fontweight='bold')

    if 'RF_arrival' in kwargs:
        RF_arrival = np.array(kwargs.get('RF_arrival'))
        list_color = 3*['yellow'] + 3*['lime'] + 3*['magenta']

        for k in range(RF_arrival.shape[1]):
            ax[0, 0].scatter(list(range(n_rf)), RF_arrival[idx_sort, k], c=list_color[k], s=1)
            ax[0, 1].scatter(list(range(n_rf)), RF_arrival[idx_sort, k], c=list_color[k], s=1)

    ax[1, 0].set_ylabel('baz (°)')
    ax[1, 0].set_ylim(0, 360)
    ax[1, 0].yaxis.set_major_locator(ti.MultipleLocator(180))
    ax[1, 0].yaxis.set_minor_locator(ti.MultipleLocator(90))
    
    ax[1, 0].scatter(list(range(n_rf)), baz[idx_sort], c='k', s=1)
    ax[1, 1].scatter(list(range(n_rf)), baz[idx_sort], c='k', s=1)
    
    ax[2, 0].set_ylabel('p (s/km)')
    ax[2, 0].set_ylim(.04, .08)
    ax[2, 0].yaxis.set_major_locator(ti.MultipleLocator(.02))
    ax[2, 0].yaxis.set_minor_locator(ti.MultipleLocator(.01))
    
    ax[2, 0].scatter(list(range(n_rf)), slow[idx_sort], c='k', s=1)
    ax[2, 1].scatter(list(range(n_rf)), slow[idx_sort], c='k', s=1)

    for i in range(1, 3):
        for j in range(2):
            ax[i, j].grid(which='both', axis='y', ls='--', color='lightgrey', zorder=-10)
            ax[i, j].grid(which='major', axis='x', ls='--', color='lightgrey', zorder=-10)
    
    axin = inset_axes(ax[1, -1], width="20%", height="20%", loc="upper left", bbox_to_anchor=(.025, 0, 1, 1), bbox_transform=ax[1, -1].transAxes)
    axin.xaxis.set_ticks_position("bottom")
    cb = fig.colorbar(mappable, cax=axin, orientation='horizontal')
    cb.set_ticks([-1, 0, 1])
    cb.ax.tick_params(size=0, labelsize=6)
    cb.set_label('norm. amplitude', fontsize=6, labelpad=.1)
    
    ax[0, 0].set_ylabel('time (s)')
    ax[0, 0].set_ylim(tmin, tmax)
    ax[0, 0].yaxis.set_major_locator(ti.MultipleLocator(5))
    ax[0, 0].yaxis.set_minor_locator(ti.MultipleLocator(2.5))
    ax[0, 0].invert_yaxis()
    
    ax[-1, 0].set_xlabel('event #')
    ax[-1, 1].set_xlabel('event #')

    ax[-1, 0].xaxis.set_major_locator(ti.MultipleLocator(50))
    ax[-1, 0].xaxis.set_minor_locator(ti.MultipleLocator(10))

    ax[-1, 1].xaxis.set_major_locator(ti.MultipleLocator(50))
    ax[-1, 1].xaxis.set_minor_locator(ti.MultipleLocator(10))
    
    fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=.03, hspace=.08)

    if show:
        plt.show()
    else:
        if 'title' in kwargs:
            title = str(kwargs.get('title')) + '_' + rot
        else:
            title = 'RF_mesh_' + rot
        fig.savefig(plotdir + '/' + title + '.png', dpi=300, bbox_inches='tight')
        plt.close(fig)


def plot_RF_plate(rfarray, baz, slow, taxis, rot, show=False, plotdir=None, **kwargs):
    idx_sort = np.argsort(baz)
    n_rf = rfarray.shape[0]
    npts = rfarray.shape[2]
    alpha = .5

    fig, ax = plt.subplots(1, 4, width_ratios=[3, 1, 1, 3], figsize=(12, 10), sharey=True)

    minmax = np.nanmax(np.abs(rfarray))
    rfarray /= minmax

    if 'RF_arrival' in kwargs:
        RF_arrival = np.array(kwargs.get('RF_arrival'))
        list_color = np.tile(['yellow', 'lime', 'magenta'], 3)
    
    tmax = np.nanmax(taxis)
    if 'tmax' in kwargs:
        tmax = float(kwargs.get('tmax'))
    tmin = np.nanmin(taxis)
    if 'tmin' in kwargs:
        tmin = float(kwargs.get('tmin'))
    fact = 2.5
    if 'fact' in kwargs:
        fact = float(kwargs.get('fact'))
        
    for k, i in enumerate(idx_sort):

        ax[0].plot(taxis, k+fact*rfarray[i, 0, :], c='k', lw=.5)
        ax[0].fill_between(taxis, k+fact*rfarray[i, 0, :], npts*[k], where=rfarray[i, 0, :]<0, color='r', alpha=alpha)
        ax[0].fill_between(taxis, k+fact*rfarray[i, 0, :], npts*[k], where=rfarray[i, 0, :]>=0, color='b', alpha=alpha)

        ax[3].plot(taxis, k+fact*rfarray[i, 1, :], c='k', lw=.5)
        ax[3].fill_between(taxis, k+fact*rfarray[i, 1, :], npts*[k], where=rfarray[i, 1, :]<0, color='r', alpha=alpha)
        ax[3].fill_between(taxis, k+fact*rfarray[i, 1, :], npts*[k], where=rfarray[i, 1, :]>=0, color='b', alpha=alpha)

        if 'RF_arrival' in kwargs:

            ax[0].scatter(RF_arrival[i], 9*[k], c=list_color, s=1)
            ax[3].scatter(RF_arrival[i], 9*[k], c=list_color, s=1)

    if rot == 'rtz':
        ax[0].set_title('RF comp. R', fontweight='bold')
        ax[3].set_title('RF comp. T', fontweight='bold')
    elif rot == 'lqt':
        ax[0].set_title('RF comp. Q', fontweight='bold')
        ax[3].set_title('RF comp. T', fontweight='bold')

    for k in [0, 3]:
        ax[k].spines['top'].set_visible(False)
        ax[k].spines['right'].set_visible(False)

    for k in [0, 3]:
        ax[k].spines['top'].set_visible(False)
        ax[k].spines['right'].set_visible(False)

    ax[1].scatter(baz[idx_sort], list(range(n_rf)), c='k', s=1)
    ax[1].set_xlabel('baz (°)')
    ax[1].set_xlim(0, 360)
    ax[1].xaxis.set_major_locator(ti.MultipleLocator(180))
    ax[1].xaxis.set_minor_locator(ti.MultipleLocator(90))
    ax[1].grid(which='both', axis='x', ls='--', color='lightgrey', zorder=-10)
    ax[1].grid(which='major', axis='y', ls='--', color='lightgrey', zorder=-10)
    ax[1].tick_params(axis='x', which='both', top='on', labeltop='on', bottom=False, labelbottom=False)
    plt.setp(ax[1].xaxis.get_majorticklabels(), ha="right")
    
    ax[2].scatter(slow[idx_sort], list(range(n_rf)), c='k', s=1)
    ax[2].set_xlabel('p (s/km)')
    ax[2].set_xlim(.04, .08)
    ax[2].xaxis.set_major_locator(ti.MultipleLocator(.02))
    ax[2].xaxis.set_minor_locator(ti.MultipleLocator(.01))
    ax[2].grid(which='both', axis='x', ls='--', color='lightgrey', zorder=-10)
    ax[2].grid(which='major', axis='y', ls='--', color='lightgrey', zorder=-10)
    ax[2].tick_params(axis='x', which='both', top='on', labeltop='on', bottom=False, labelbottom=False)
    plt.setp(ax[2].xaxis.get_majorticklabels(), ha="left")
    
    for k in [0, 3]:
        ax[k].set_xlabel('time (s)', labelpad=.1)
        ax[k].set_xlim(tmin, tmax)
        ax[k].xaxis.set_major_locator(ti.MultipleLocator(5))
        ax[k].xaxis.set_minor_locator(ti.MultipleLocator(2.5))
    
    ax[0].set_ylabel('event #', labelpad=.1)
    ax[0].set_ylim(-fact, n_rf-1+fact)
    ax[0].yaxis.set_major_locator(ti.MultipleLocator(10))
    ax[0].yaxis.set_minor_locator(ti.MultipleLocator(5))

    fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=.03, hspace=.08)
    
    if show:
        plt.show()
    else:
        if 'title' in kwargs:
            title = str(kwargs.get('title')) + '_' + rot
        else:
            title = 'RF_plate_' + rot
        fig.savefig(plotdir + '/' + title + '.png', dpi=300, bbox_inches='tight')
        plt.close(fig)


def plot_dataVSmodel(data, prediction, baz, slow, phases2extract, invert_tau_ratio, show=False, plotdir=None, **kwargs):
    idx_sort = np.argsort(baz)
    RF_time_d, RF_sigtime_d, RF_amp_d, RF_sigpol_d = data
    RF_time_m, RF_amp_m = prediction    
    n_rf = RF_time_d.shape[0]
    n_phase = len(phases2extract)
    n_tau = len(invert_tau_ratio)

    fig, ax = plt.subplots(ncols=2, nrows=n_phase+3, figsize=(16, 8), sharex=True, sharey='row')
    gs = ax[0, 0].get_gridspec()

    warnings.filterwarnings(action='ignore', message='All-NaN slice encountered')
    tmax = np.nanmax([np.nanmax(RF_time_d), np.nanmax(RF_time_m)])*1.1
    if 'tmax' in kwargs:
        tmax = float(kwargs.get('tmax'))

    if len(invert_tau_ratio) != 0:
        for a in ax[:int(n_phase*.8), 0]:
            a.remove()
        axtime = fig.add_subplot(gs[:int(n_phase*.8), 0])

        for a in ax[int(n_phase*.8):n_phase+1, 0]:
            a.remove()
        axtau = fig.add_subplot(gs[int(n_phase*.8):n_phase+1, 0])
    else:
        for a in ax[:n_phase+1, 0]:
            a.remove()
        axtime = fig.add_subplot(gs[:n_phase+1, 0])

    list_color = sb.color_palette('Set1', 15)
    list_light = np.linspace(1, .5, 3)

    RF_amp_d /= np.nanmax(np.abs(RF_amp_d))
    RF_amp_m /= np.nanmax(np.abs(RF_amp_m))
    norm  = colors.Normalize(vmin=-1, vmax=1)
    cmap = 'bwr_r'
    mappable = cmx.ScalarMappable(norm=norm, cmap=cmap)

    legends = []
    for k in range(n_phase):

        td = axtime.errorbar(x=list(range(n_rf)), y=RF_time_d[idx_sort, k], yerr=RF_sigtime_d[idx_sort, k], xerr=None,
                             fmt='.', c=adjust_lightness(list_color[k%3], amount=list_light[k//3]),
                             zorder=1, markersize=5, elinewidth=.5, capsize=1.8, ecolor='lightgrey')

        tm = axtime.scatter(list(range(n_rf)), RF_time_m[idx_sort, k],
                            s=70, color=adjust_lightness(list_color[k%3], amount=list_light[k//3]), alpha=.25, zorder=0, edgecolor='None')
        
        legends.append((td, tm))

        ax[-4-k, 1].pcolormesh(np.vstack((RF_amp_d[idx_sort, k], RF_amp_m[idx_sort, k])), cmap=cmap, norm=norm)
        ax[-4-k, 1].yaxis.set_tick_params(left=False)
        ax[-4-k, 1].plot([0, n_rf], 2*[1], ls='--', lw=1, c='k')
        ax[-4-k, 1].text(.99, .9, phases2extract[k], transform=ax[-4-k, 1].transAxes, va='top', ha='right')
        ax[-4-k, 1].set_facecolor('lightgrey')
    
    # t = 0
    ax[-3, 1].pcolormesh(np.vstack((RF_amp_d[idx_sort, k+1], RF_amp_m[idx_sort, k+1])), cmap=cmap, norm=norm)
    ax[-3, 1].yaxis.set_tick_params(left=False)
    ax[-3, 1].plot([0, n_rf], 2*[1], ls='--', lw=1, c='k')
    ax[-3, 1].text(.99, .9, 't = 0', transform=ax[-3, 1].transAxes, va='top', ha='right')
    ax[-3, 1].set_facecolor('lightgrey')

    axtime.legend(legends, phases2extract, handler_map={tuple: HandlerTuple(ndivide=None)},
                  loc='upper right', ncol=3, columnspacing=0.2, handletextpad=0.2)
    
    list_color = sb.color_palette('Set2', 15)

    if len(invert_tau_ratio) != 0:
        legends = []
        legends_label = []
        for k in range(n_tau):
            
            taud = axtau.errorbar(x=list(range(n_rf)), y=RF_time_d[idx_sort, n_phase+2*k],
                                yerr=RF_sigtime_d[idx_sort, n_phase+2*k], xerr=None,
                                fmt='.', c=adjust_lightness(list_color[k], amount=1),
                                zorder=5, markersize=5, elinewidth=.5, capsize=1.8, ecolor='lightgrey')

            taum = axtau.scatter(list(range(n_rf)), RF_time_m[idx_sort, n_phase+2*k],
                                s=70, color=adjust_lightness(list_color[k], amount=1), alpha=.25, zorder=4, edgecolor='None')
            
            legends.append((taud, taum))
            legends_label.append("(P0p{0:d}S-P0p{1:d}S)/(P{0:d}S-P{1:d}S)".format(invert_tau_ratio[k]+1, invert_tau_ratio[k]))

            taud = axtau.errorbar(x=list(range(n_rf)), y=RF_time_d[idx_sort, n_phase+2*k+1],
                                yerr=RF_sigtime_d[idx_sort, n_phase+2*k+1], xerr=None,
                                fmt='.', c=adjust_lightness(list_color[k], amount=.5),
                                zorder=5, markersize=5, elinewidth=.5, capsize=1.8, ecolor='lightgrey')

            taum = axtau.scatter(list(range(n_rf)), RF_time_m[idx_sort, n_phase+2*k+1],
                                s=70, color=adjust_lightness(list_color[k], amount=.5), alpha=.25, zorder=4, edgecolor='None')
            
            legends.append((taud, taum))
            legends_label.append("(P0s{0:d}S-P0s{1:d}S)/(P{0:d}S-P{1:d}S)".format(invert_tau_ratio[k]+1, invert_tau_ratio[k]))
        
        axtau.legend(legends, legends_label, handler_map={tuple: HandlerTuple(ndivide=None)},
                    loc='upper right', ncol=2, columnspacing=0.2, handletextpad=0.2)

        axtau.xaxis.set_tick_params(labelbottom=False)
        axtau.set_ylabel(r'$\dfrac{\tau_i}{\tau_j}$')
        axtau.grid(which='major', ls='--', color='lightgrey')
        axtau.set_ylim(bottom=0)
        axtau.set_xlim(0, n_rf)
        axtau.set_axisbelow(True)

    ax[-3, 1].text(.01, .25, 'Data', transform=ax[-3, 1].transAxes, va='center', ha='left')
    ax[-3, 1].text(.01, .75, 'Prediction', transform=ax[-3, 1].transAxes, va='center', ha='left')

    axtime.set_ylim(0, tmax)
    axtime.set_xlim(0, n_rf)
    ax[-1, 0].set_xlim(0, n_rf)

    ax[-2, 0].set_ylabel('baz (°)')
    ax[-2, 0].set_ylim(0, 360)
    ax[-2, 0].scatter(list(range(n_rf)), baz[idx_sort], c='k', s=1)
    ax[-2, 1].scatter(list(range(n_rf)), baz[idx_sort], c='k', s=1)
    
    ax[-1, 0].set_ylabel('p (s/km)')
    ax[-1, 0].set_ylim(.04, .08)
    ax[-1, 0].scatter(list(range(n_rf)), slow[idx_sort], c='k', s=1)
    ax[-1, 1].scatter(list(range(n_rf)), slow[idx_sort], c='k', s=1)

    for i in range(-2, 0):
        for j in range(2):
            ax[i, j].grid(which='both', axis='y', ls='--', color='lightgrey')
            ax[i, j].grid(which='major', axis='x', ls='--', color='lightgrey')
            ax[i, j].set_axisbelow(True)
    
    ax[-1, 0].set_xlabel('event #')
    ax[-1, 1].set_xlabel('event #')

    axtime.xaxis.set_tick_params(labelbottom=False)
    axtime.set_ylabel('Arrival time (s)')
    axtime.set_title('Converted waves arrival times', fontweight='bold')
    axtime.grid(which='major', ls='--', color='lightgrey')
    axtime.set_axisbelow(True)
    
    ax[0, 1].set_title('Converted waves amplitudes (transverse component)', fontweight='bold')

    axin = inset_axes(ax[-2, -1], width="20%", height="20%", loc="upper left", bbox_to_anchor=(.025, -.1, 1, 1), bbox_transform=ax[-2, -1].transAxes)
    axin.xaxis.set_ticks_position("bottom")
    cb = fig.colorbar(mappable, cax=axin, orientation='horizontal')
    cb.set_ticks([-1, 0, 1])
    cb.ax.tick_params(size=0, labelsize=6)
    cb.set_label('norm. amplitude', fontsize=6, labelpad=.5)
    cb.ax.xaxis.set_label_position('top')
    
    fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=.02, hspace=.2)

    if show:
        plt.show()
    else:
        if 'title' in kwargs:
            title = str(kwargs.get('title'))
        else:
            title = 'dataVSmodel'
        fig.savefig(plotdir + '/' + title + '.png', dpi=300, bbox_inches='tight')
        plt.close(fig)


def plot_evol_model(z_accepted, name, param_inv_list, param_inv_bounds, n_layers, color, plotdir):
    """
    name: accepted or rejected
    """
    nrows = np.unique([b[0] for b in param_inv_list]).size
    ncols = n_layers
    n_metro = len(z_accepted)

    fig, ax = plt.subplots(nrows=nrows+(nrows==1), ncols=ncols, sharex=True, figsize=(4*ncols, 2*nrows)) 

    ax[0, 0].set_xlim(0, n_metro)

    for j in range(ncols):

        if nrows == 1:
            ax[nrows, j].axis('off')

        for i in range(nrows):

            if (np.unique([b[0] for b in param_inv_list])[i], j) in param_inv_list:

                idx = [x for x, y in enumerate(param_inv_list) if y[0] == np.unique([b[0] for b in param_inv_list])[i] and y[1] == j][0]

                ax[i, j].set_title(param_inv_list[idx][0] + ' %i (' %param_inv_list[idx][1] + get_param_unit([param_inv_list[idx][0]])[0][3] + ')')

                ax[i, j].plot(list(range(len(z_accepted))), np.array(z_accepted)[:, idx], c=color, zorder=10)

                ax[i, j].plot([0, n_metro], 2*[param_inv_bounds[idx][1]], ls='--', c='k', zorder=15)
                ax[i, j].plot([0, n_metro], 2*[param_inv_bounds[idx][2]], ls='--', c='k', zorder=15)

                if i == nrows:
                    ax[i, j].set_xlabel('accepted model #')

            else:
                ax[i, j].axis('off')

    fig.savefig(plotdir + '/evolution_' + name + '_models.png', dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_evol_pdf(z_accepted, z_rejected, plotdir):

    z_accepted = z_accepted[10:]
    z_rejected = z_rejected[10:]

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

    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=.1, hspace=.1)

    fig.savefig(plotdir + '/evolution_pdf.png', dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_marginals(z_accepted, param_inv_list, param_inv_bounds, plotdir):
    n = len(param_inv_list)
    param_with_unit = [param_inv_list[k][0] + ' %i (' %param_inv_list[k][1] + get_param_unit([param_inv_list[k][0]])[0][3] + ')' for k in range(n)]

    mean_model = np.nanmean(z_accepted[:, :n], axis=0)
    max_model = z_accepted[np.nanargmax(z_accepted[:, -1]), :n]
    median_model = np.nanmedian(z_accepted[:, :n], axis=0)

    fig, ax = plt.subplots(nrows=n+(n==1), ncols=n+(n==1), figsize=(12, 12), sharex='col')
    
    fig.tight_layout(pad=1)
        
    for i in range(n):
        for j in range(i+1):

            if i==j:
                isamples = [s[i] for s in z_accepted[:, :-4]]

                min, max = param_inv_bounds[j][1], param_inv_bounds[j][2]

                kde = ss.gaussian_kde(isamples)
                x = np.linspace(min, max, 100)
                f = kde(x)

                sb.histplot(np.array(isamples), ax=ax[i, j], stat='density', color='lightgrey')
                ax[i, j].plot(x, f, c='purple')

                sb.histplot(isamples, ax=ax[i, j], stat='density', color='lightgrey')
                sb.kdeplot(isamples, color='darkmagenta', ax=ax[i, j])

                ax[i, j].spines['right'].set_visible(False)
                ax[i, j].spines['top'].set_visible(False)
                ax[i, j].tick_params(left=False, right=False , labelleft=False)
                ax[i, j].set_title(param_with_unit[j], fontweight='bold')
                ax[i, j].set_ylabel(' ')
                ax[i, j].set_box_aspect(1)
                ax[i, j].xaxis.set_major_locator(plt.MaxNLocator(2))

                unit = get_param_unit([param_inv_list[i][0]])[0][3]

                ax[i, j].text(.95, .8,
                              'max_glob: %.4f ' % max_model[i] + unit +
                              '\nmax_marg: %.4f ' % x[np.nanargmax(f)] + unit +
                              '\nmean: %.4f ' % mean_model[i] + unit +
                              '\nmedian: %.4f ' % median_model[i] + unit, transform=ax[i, j].transAxes)

                ax[i, j].set_xlim(param_inv_bounds[j][1], param_inv_bounds[j][2])

            else:
                isamples = [s[i] for s in z_accepted[:, :-4]]
                jsamples = [s[j] for s in z_accepted[:, :-4]]
 
                # Extract x and y
                x = jsamples
                y = isamples
                # Define the borders
                minx, maxx = param_inv_bounds[j][1], param_inv_bounds[j][2]
                miny, maxy = param_inv_bounds[i][1], param_inv_bounds[i][2]
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
                    ax[i, j].set_ylabel(param_with_unit[i], fontweight='bold')
                if i == n-1:
                    ax[i, j].set_xlabel(param_with_unit[j], fontweight='bold')

                if j >= 1:
                    ax[i, j].tick_params(labelleft=False)

                ax[i, j].set_box_aspect(1)
    
    if n == 1:
        for k in range(n+1):
            ax[-1, k].axis('off')
            ax[k, -1].axis('off')

    fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=.12, hspace=.12)

    fig.savefig(plotdir + '/marginals_pdf.png', dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_confidence_marginals(z_accepted, param_inv_list, param_inv_bounds, plotdir):
    n = len(param_inv_list)
    param_names = [param_inv_list[k][0] + ' %i (' %param_inv_list[k][1] + get_param_unit([param_inv_list[k][0]])[0][3] + ')' for k in range(n)]

    mean_model = np.nanmean(z_accepted[:, :n], axis=0)
    max_model = z_accepted[np.nanargmax(z_accepted[:, -1]), :n]
    median_model = np.nanmedian(z_accepted[:, :n], axis=0)

    fig, ax = plt.subplots(nrows=n+(n==1), ncols=n+(n==1), figsize=(12, 12), sharex='col')
    
    fig.tight_layout(pad=1)

    for i in range(n):
        for j in range(i+1):

            if i==j:
                isamples = [s[i] for s in z_accepted[:, :-4]]

                min, max = param_inv_bounds[j][1], param_inv_bounds[j][2]

                kde = ss.gaussian_kde(isamples)
                x = np.linspace(min, max, 100)
                f = kde(x)
                sumf = np.nansum(f)

                sb.histplot(np.array(isamples), ax=ax[i, j], stat='density', color='lightgrey')
                ax[i, j].plot(x, f, c='k')

                g = lambda x, f, alpha : np.abs(np.nansum(f[f >= x]) - alpha)
                f68 = so.minimize(g, x0=np.nanmean(f/sumf), args=(f/sumf, .68), method='Powell')['x'][0]
                f95 = so.minimize(g, x0=np.nanmean(f/sumf)/2, args=(f/sumf, .95), method='Powell')['x'][0]

                ax[i, j].fill_between(x, f, len(f)*[0], where=f/sumf>=f95, color=adjust_lightness('darkorchid', 1.5), alpha=.75)
                ax[i, j].fill_between(x, f, len(f)*[0], where=f/sumf>=f68, color=adjust_lightness('darkorchid', 1), alpha=.75)

                ax[i, j].spines['right'].set_visible(False)
                ax[i, j].spines['top'].set_visible(False)
                ax[i, j].tick_params(left=False, right=False , labelleft=False)
                ax[i, j].set_title(param_names[j], fontweight='bold')
                ax[i, j].set_ylabel(' ')
                ax[i, j].set_box_aspect(1)
                ax[i, j].xaxis.set_major_locator(plt.MaxNLocator(2))
                ax[i, j].set_xlim(min, max)
                ax[i, j].set_ylim(0, 1.1*np.nanmax(f))

                unit = get_param_unit([param_inv_list[i][0]])[0][3]

                ax[i, j].text(.95, .8,
                              'max_glob: %.4f ' % max_model[i] + unit + 
                              '\nmax_marg: %.4f ' % x[np.nanargmax(f)] + unit + 
                              '\nmean: %.4f ' % mean_model[i] + unit + 
                              '\nmedian: %.4f ' % median_model[i] + unit, transform=ax[i, j].transAxes)

            else:

                isamples = [s[i] for s in z_accepted[:, :-4]]
                jsamples = [s[j] for s in z_accepted[:, :-4]]
                x = jsamples
                y = isamples

                minx, maxx = param_inv_bounds[j][1], param_inv_bounds[j][2]
                miny, maxy = param_inv_bounds[i][1], param_inv_bounds[i][2]
                deltaX = (maxx - minx)/10
                deltaY = (maxy - miny)/10
                xmin = minx - deltaX
                xmax = maxx + deltaX
                ymin = miny - deltaY
                ymax = maxy + deltaY
                nsample = 100
                xx, yy = np.mgrid[xmin:xmax:nsample*1j, ymin:ymax:nsample*1j]

                positions = np.vstack([xx.ravel(), yy.ravel()])
                values = np.vstack([x, y])
                kernel = ss.gaussian_kde(values)
                f = np.reshape(kernel(positions).T, xx.shape)
                f /= np.nansum(f)

                g = lambda x, f, alpha : np.abs(np.nansum(f[f >= x]) - alpha)
                f68 = so.minimize(g, x0=np.nanmean(f), args=(f, .68), method='Powell')['x'][0]
                f95 = so.minimize(g, x0=np.nanmean(f)/2, args=(f, .95), method='Powell')['x'][0]

                ax[i, j].contourf(xx, yy, f, levels=[f95, f68, 1], alpha=.75, colors=[adjust_lightness('orange', 1.5),
                                                                                     adjust_lightness('orange', 1)])
                
                ax[i, j].contour(xx, yy, f, levels=[f95, f68], colors=[adjust_lightness('orange', 1.5),
                                                                                     adjust_lightness('orange', 1)])

                ax[i, j].scatter(x, y, c='k', s=1, alpha=.25, zorder=-2)

                ax[j, i].axis('off')
                ax[i, j].xaxis.set_major_locator(plt.MaxNLocator(2))
                ax[i, j].yaxis.set_major_locator(plt.MaxNLocator(2))

                ax[i, j].set_xlim(minx, maxx)
                ax[i, j].set_ylim(miny, maxy)

                if j == 0:
                    ax[i, j].set_ylabel(param_names[i], fontweight='bold')
                if i == n-1:
                    ax[i, j].set_xlabel(param_names[j], fontweight='bold')
                if j >= 1:
                    ax[i, j].tick_params(labelleft=False)

                ax[i, j].set_box_aspect(1)
    
    if n == 1:
        for k in range(n+1):
            ax[-1, k].axis('off')
            ax[k, -1].axis('off')
    
    fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=.12, hspace=.12)

    fig.savefig(plotdir + '/marginals_pdf_confidence.png', dpi=300, bbox_inches='tight')
    plt.close(fig)


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

    fig.savefig(plotdir + '/proposal_covariance_matrix.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
