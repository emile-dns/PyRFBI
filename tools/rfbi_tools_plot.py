#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 10:52:15 2023

@author: Emile DENISE
"""

import numpy as np
import scipy.stats as ss
import scipy.optimize as so
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
import matplotlib.ticker as ti
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerTuple
import seaborn as sb
import colorsys


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


def plot_RF_mesh(rfarray, baz, slow, dt, plotdir, **kwargs):
    idx_sort = np.argsort(baz)
    n_rf = rfarray.shape[0]
    npts = rfarray.shape[2]

    fig, ax = plt.subplots(3, 2, height_ratios=[3, 1, 1], figsize=(12, 6), sharex=True, sharey='row')

    minmax = np.nanmax(np.abs(rfarray))
    norm  = colors.Normalize(vmin=-minmax, vmax=minmax)
    cmap = 'seismic_r'
    mappable = cmx.ScalarMappable(norm=norm, cmap=cmap)
    
    time = np.array(range(npts))*dt-npts*dt/2

    tmax = 45
    if 'tmax' in kwargs:
        tmax = float(kwargs.get('tmax'))
    
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
    
    fig.savefig(plotdir + '/RF_mesh.png', dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_RF_plate(rfarray, baz, slow, dt, plotdir, **kwargs):
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
    
    tmax = 45
    if 'tmax' in kwargs:
        tmax = float(kwargs.get('tmax'))
    
    alpha = .5
        
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
    
    ax[1].scatter(baz[idx_sort], list(range(n_rf)), c='k', s=1)
    ax[1].scatter(baz[idx_sort], list(range(n_rf)), c='k', s=1)
    
    ax[2].set_xlabel('Slowness\n' + r'($s.km^{-1}$)')
    ax[2].set_xlim(.04, .08)
    ax[2].xaxis.set_major_locator(ti.MultipleLocator(.02))
    ax[2].xaxis.set_minor_locator(ti.MultipleLocator(.01))
    ax[2].grid(which='both', axis='x', ls='--', color='lightgrey', zorder=-10)
    ax[2].grid(which='major', axis='y', ls='--', color='lightgrey', zorder=-10)
    plt.setp(ax[2].xaxis.get_majorticklabels(), ha='left')
    
    ax[2].scatter(slow[idx_sort], list(range(n_rf)), c='k', s=1)
    ax[2].scatter(slow[idx_sort], list(range(n_rf)), c='k', s=1)
    
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
    
    fig.savefig(plotdir + '/RF_plate.png', dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_dataVSmodel(data, prediction, baz, slow, title, phases2extract, plotdir, tmax=45, **kwargs):
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

            ax[8-k, 1].pcolormesh(np.vstack((RF_amp_d[idx_sort, k], RF_amp_m[idx_sort, k])), cmap='bwr_r', vmin=-minmax, vmax=minmax)
            ax[8-k, 1].yaxis.set_tick_params(left=False)
            ax[8-k, 1].plot([0, n_rf], 2*[1.], ls='--', lw=1, c='k')

            ax[8-k, 1].text(.99, .9, phase_list[k], transform=ax[8-k, 1].transAxes, va='top', ha='right')
            ax[8-k, 1].set_facecolor('lightgrey')
    
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
    
    fig.savefig(plotdir + '/' + title + '.png', dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_evol_model(z_accepted, name, param_inv_list, param_inv_bounds, n_layers, plotdir):
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

                ax[i, j].set_title(param_inv_list[idx][0] + ' %i' %param_inv_list[idx][1])

                ax[i, j].plot(list(range(len(z_accepted))), np.array(z_accepted)[:, idx], c='slateblue', zorder=10)

                ax[i, j].plot([0, n_metro], 2*[param_inv_bounds[idx][0]], ls='--', c='k')
                ax[i, j].plot([0, n_metro], 2*[param_inv_bounds[idx][1]], ls='--', c='k')

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
    param_with_unit = [param_inv_list[k][0] + ' %i' %param_inv_list[k][1] for k in range(n)]

    fig, ax = plt.subplots(nrows=n+(n==1), ncols=n+(n==1), figsize=(12, 12), sharex='col')
    
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
    param_names = [param_inv_list[k][0] + ' %i' %param_inv_list[k][1] for k in range(n)]

    fig, ax = plt.subplots(nrows=n+(n==1), ncols=n+(n==1), figsize=(12, 12), sharex='col')
    
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
                ax[i, j].set_title(param_names[j], fontweight='bold')
                ax[i, j].set_ylabel(' ')
                ax[i, j].set_box_aspect(1)
                ax[i, j].xaxis.set_major_locator(plt.MaxNLocator(2))
                ax[i, j].set_xlim(param_inv_bounds[j][0], param_inv_bounds[j][1])

            else:

                isamples = [s[i] for s in z_accepted[:, :-4]]
                jsamples = [s[j] for s in z_accepted[:, :-4]]
                x = jsamples
                y = isamples

                minx, maxx = param_inv_bounds[j][0], param_inv_bounds[j][1]
                miny, maxy = param_inv_bounds[i][0], param_inv_bounds[i][1]
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


def plot_marginals_grid_search(sampling, n_params, n_sample, plotdir, param_inv_list, param_inv_bounds):
    # valable pour log uniquement

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

                        marg[ivi, ivj] = f
                
                marg_2D[(i, j)] = (vali, valj, marg)
            
            else:

                marg = []

                for v in np.unique(sampling[:, i], axis=0):

                    f = np.exp(sampling[sampling[:, i] == v][:, -1])
                    f = np.nansum(f)

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
    
    fig.savefig(plotdir + '/marginals_pdf.png', dpi=300, bbox_inches='tight')
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
