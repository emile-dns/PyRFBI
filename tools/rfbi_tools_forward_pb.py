#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 10:52:15 2023

@author: Emile DENISE
"""

# %% Packages

import numpy as np
from pyraysum import prs
import multiprocessing as mp
from tools.rfbi_tools import *


# %% Functions

def extract_RF_time_amp(result, phases2extract, invert_tau_ratio):
    """
    result is the result of run with run_full (RTZ rotation)
    Returns time peaks, their amplitudes, polarities, and eventually the RF data array.
    """
    n_RF = len(result.rfs)
    n_phases = len(phases2extract)
    taxis_RF = result.rfs[0][0].stats.taxis

    RF_time = np.full((n_RF, n_phases), np.nan)
    RF_transamp = np.full((n_RF, n_phases), np.nan)

    for i, rf in enumerate(result.rfs):

        if 'P' in rf[0].stats.conversion_names:
            t_P = rf[0].stats.phase_times[rf[0].stats.conversion_names == 'P'][0]

            for j, phase in enumerate(phases2extract):

                if phase in rf[0].stats.conversion_names:
                    RF_time[i, j] = rf[0].stats.phase_times[rf[0].stats.conversion_names == phase][0] - t_P

        idx_arrival = find_idx_nearest(taxis_RF, RF_time[i])
        RF_transamp[i, :] = find_values_idx(rf[1].data, idx_arrival)
    
    # t=0
    idx_t0 = find_idx_nearest(taxis_RF, [0])[0]
    transamp_t0 = np.array([result.rfs[k][1].data[idx_t0] for k in range(n_RF)])
    RF_transamp = np.concatenate((RF_transamp, transamp_t0[..., np.newaxis]), axis=1)
    
    for l in invert_tau_ratio:

        if l == 0:

            # to-do: calculate the theoretical ratio to see if it is intesting for Vp/Vs in the first layer
            t_PS = RF_time[:, np.argwhere(phases2extract == 'P' + str(l+1) + 'S')[0][0]]
            t_PpS = RF_time[:, np.argwhere(phases2extract == 'P0p' + str(l+1) + 'S')[0][0]]
            t_PsS = RF_time[:, np.argwhere(phases2extract == 'P0s' + str(l+1) + 'S')[0][0]]

            RF_time = np.concatenate((RF_time, (t_PpS/t_PS)[..., np.newaxis]), axis=1)
            RF_time = np.concatenate((RF_time, (t_PsS/t_PS)[..., np.newaxis]), axis=1)
        
        else:
            
            tau_PS = RF_time[:, np.argwhere(phases2extract == 'P' + str(l+1) + 'S')[0][0]] - RF_time[:, np.argwhere(phases2extract == 'P' + str(l) + 'S')[0][0]]
            tau_PpS = RF_time[:, np.argwhere(phases2extract == 'P0p' + str(l+1) + 'S')[0][0]] - RF_time[:, np.argwhere(phases2extract == 'P0p' + str(l) + 'S')[0][0]]
            tau_PsS = RF_time[:, np.argwhere(phases2extract == 'P0s' + str(l+1) + 'S')[0][0]] - RF_time[:, np.argwhere(phases2extract == 'P0s' + str(l) + 'S')[0][0]]

            RF_time = np.concatenate((RF_time, (tau_PpS/tau_PS)[..., np.newaxis]), axis=1)
            RF_time = np.concatenate((RF_time, (tau_PsS/tau_PS)[..., np.newaxis]), axis=1)

    return RF_time, RF_transamp


def predict_RF(struct, geomray, run_control, type_filt, freq_filt, phases2extract, invert_tau_ratio, return_rf=False):
    result = prs.run(struct, geomray, run_control, rf=True)

    if type_filt == 'bandpass':
        result.filter('rfs', 'bandpass', freqmin=freq_filt[0], freqmax=freq_filt[1], zerophase=True, corners=2)
    else:
        result.filter('rfs', type_filt, freq=freq_filt, zerophase=True, corners=2)

    RF_time, RF_transamp = extract_RF_time_amp(result, phases2extract, invert_tau_ratio)
    
    if return_rf:

        RF_array = np.empty((geomray.ntr, 2, run_control.npts))
        for k in range(geomray.ntr):
            RF_array[k, 0, :] = result.rfs[k][0].data
            RF_array[k, 1, :] = result.rfs[k][1].data

        return RF_time, RF_transamp, RF_array, result

    result = None
    return RF_time, RF_transamp


def predict_RF_par(struct, geomray_split, run_control, type_filt, freq_filt, phases2extract, invert_tau_ratio, return_rf=False, n_proc=4):

    with mp.get_context("fork").Pool(n_proc) as pool:
        result_par = pool.starmap(predict_RF, [(struct, geom, run_control, type_filt, freq_filt, phases2extract, invert_tau_ratio, return_rf) for geom in geomray_split])

    RF_time = result_par[0][0]
    RF_transamp = result_par[0][1]
    if return_rf:
        RF_array = result_par[0][2]
    pool.close()

    for k in range(1, n_proc): 
        RF_time = np.vstack((RF_time, result_par[k][0]))
        RF_transamp = np.vstack((RF_transamp, result_par[k][1]))
        if return_rf:
            RF_array = np.vstack((RF_array, result_par[k][2]))
    
    if return_rf:
        return RF_time, RF_transamp, RF_array
    return RF_time, RF_transamp


def g(struct, geomray, run_control, type_filt, freq_filt, sigma_amp, phases2extract, invert_tau_ratio):
    RF_time, RF_transamp = predict_RF(struct, geomray, run_control, type_filt, freq_filt, phases2extract, invert_tau_ratio, return_rf=False)
    RF_sigamp = np.full(RF_transamp.shape, sigma_amp)

    RF_time = make_masked_array_nan(RF_time)
    RF_transamp = make_masked_array_nan(RF_transamp)
    RF_sigamp = make_masked_array_nan(RF_sigamp)

    return RF_time, RF_transamp, RF_sigamp


def g_par(struct, geomray_split, run_control, type_filt, freq_filt, sigma_amp, phases2extract, invert_tau_ratio, n_proc):
    RF_time, RF_transamp = predict_RF_par(struct, geomray_split, run_control, type_filt, freq_filt, phases2extract, invert_tau_ratio, return_rf=False, n_proc=n_proc)
    RF_sigamp = np.full(RF_transamp.shape, sigma_amp)

    RF_time = make_masked_array_nan(RF_time)
    RF_transamp = make_masked_array_nan(RF_transamp)
    RF_sigamp = make_masked_array_nan(RF_sigamp)

    return RF_time, RF_transamp, RF_sigamp
