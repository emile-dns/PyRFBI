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
import re
from pyraysum import prs
from tools.rfbi_tools_plot import *

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
plotdir = wkdir + "/figures"
if not os.path.exists(plotdir):
    os.makedirs(plotdir)

filter_freq = tuple(map(float, config['INVERSION SETUP']['filter_freq'].split(',')))
dt = config.getfloat('INVERSION SETUP', 'dt')
npts = config.getint('INVERSION SETUP', 'npts')
n_params = config.getint('INVERSION SETUP', 'n_params')
n_layers = config.getint('STRUCTURE SETUP', 'n_layers')

sampling = config['INVERSION SETUP']['sampling']

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

    if 'PsS' in  ["".join(re.findall("[a-zA-Z]+", w)) for w in phases2extract]:
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
        RF_arrival[np.logical_not(np.isfinite(time_data))] = np.nan
        RF_pol[np.logical_not(np.isfinite(pol_data))] = np.nan
        RF_amp[np.logical_not(np.isfinite(amp_data))] = np.nan
        prediction = [RF_arrival, RF_pol, RF_amp]

        plot_dataVSmodel(data, prediction, baz_data, slow_data, names[i], phases2extract, plotdir, tmax=45)

    plot_evol_model(z_accepted, 'accepted', param_inv_list, param_inv_bounds, n_layers, plotdir)
    plot_evol_model(z_rejected, 'rejected', param_inv_list, param_inv_bounds, n_layers, plotdir)
    plot_evol_pdf(z_accepted, z_rejected, plotdir)
    plot_marginals(z_accepted, param_inv_list, param_inv_bounds, plotdir)
    plot_confidence_marginals(z_accepted, param_inv_list, param_inv_bounds, plotdir)

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