#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 10:52:15 2023

@author: Emile DENISE
"""

import numpy as np
from tools.rfbi_tools import *
from pyraysum import prs
import multiprocessing as mp
import re

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


def predict_RF_par(struct, geomray_split, run_control, freq_filt, phases2extract, n_proc):

    with mp.get_context("fork").Pool(n_proc) as pool:
        result = pool.starmap(predict_RF, [(struct, geom, run_control, freq_filt, phases2extract)  for geom in geomray_split])

    rfarray, RF_arrival, RF_pol, RF_amp = result[0][0], result[0][1], result[0][2], result[0][3]
    pool.close()

    for k in range(1, n_proc):

        rfarray = np.vstack((rfarray, result[k][0]))
        RF_arrival = np.vstack((RF_arrival, result[k][1]))
        RF_pol = np.vstack((RF_pol, result[k][2]))
        RF_amp = np.vstack((RF_amp, result[k][3]))

    return rfarray, RF_arrival, RF_pol, RF_amp


def g(struct, geomray, run_control, freq_filt, sigma_amp, phases2extract):
    _, RF_arrival, RF_pol, RF_amp = predict_RF(struct, geomray, run_control, freq_filt, phases2extract)
    RF_sigamp = np.full(RF_amp.shape, sigma_amp)

    RF_arrival = make_masked_array_nan(RF_arrival)
    RF_pol = make_masked_array_nan(RF_pol)
    RF_amp = make_masked_array_nan(RF_amp)
    RF_sigamp = make_masked_array_nan(RF_sigamp)

    return RF_arrival, RF_pol, RF_amp, RF_sigamp


def g_par(struct, geomray, run_control, freq_filt, sigma_amp, phases2extract, n_proc):
    _, RF_arrival, RF_pol, RF_amp = predict_RF_par(struct, geomray, run_control, freq_filt, phases2extract, n_proc)
    RF_sigamp = np.full(RF_amp.shape, sigma_amp)

    RF_arrival = make_masked_array_nan(RF_arrival)
    RF_pol = make_masked_array_nan(RF_pol)
    RF_amp = make_masked_array_nan(RF_amp)
    RF_sigamp = make_masked_array_nan(RF_sigamp)

    return RF_arrival, RF_pol, RF_amp, RF_sigamp
