#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 09:59:53 2023

@author: Emile DENISE
"""

import os
import numpy as np
import numpy.random as rd
import scipy.stats as ss
import pandas as pd
from pyraysum import prs
import obspy.io.sac as obsac
from tools.rfbi_tools import *
from tools.rfbi_tools_plot import *
from tools.rfbi_tools_forward_pb import *

datadir = '/Users/emile/Documents/Etude/2023_2024_Cesure/Stage_Mines/data/synthetics/SERG_like/synthe_data_dipping_moho'
DEG2KM = 111.2

type_filt = 'bandpass'
freq_filt = (.08, .8)
invert_tau_ratio = np.array([2])

def save_array_synthe(baz, slow, data_array, datadir, header, name):
    idx_sort = np.argsort(baz)
    df = pd.DataFrame(data=np.concatenate((baz[idx_sort][:, np.newaxis], slow[idx_sort][:, np.newaxis], data_array[idx_sort]), axis=1),
                  columns=header)
    df.to_csv(datadir + '/' + name + '.csv', index=False, sep=';', na_rep='NaN')

def save_RF_synthe_SAC(RF_result, dt, npts, savedir):

    if not os.path.isdir(savedir + '/RF/'):
        os.makedirs(savedir + '/RF/')

    for k in range(len(RF_result.rfs)):

        header = {'kstnm': 'SYNTHE',
                'kcmpnm': 'SYNTHE',
                'npts': npts,
                'baz': RF_result.rfs[k][0].stats.baz,
                'delta': dt,
                'user3': RF_result.rfs[k][0].stats.slow * DEG2KM}

        sac = obsac.SACTrace(data=RF_result.rfs[k][0].data, **header)
        sac.write(savedir + '/RF/synthe_RF_' + str(k) + '.r')

        header = {'kstnm': 'SYNTHE',
                'kcmpnm': 'SYNTHE',
                'npts': npts,
                'baz': RF_result.rfs[k][1].stats.baz,
                'delta': dt,
                'user3': RF_result.rfs[k][1].stats.slow * DEG2KM}

        sac = obsac.SACTrace(data=RF_result.rfs[k][1].data, **header)
        sac.write(savedir + '/RF/synthe_RF_' + str(k) + '.t')

if not os.path.isdir(datadir):
    os.makedirs(datadir)

thickn_data = [36000, 24000, 9000, 0]
rho_data = [2700, 3000, 2900, 3300]
vp_data = [6200, 7000, 6500, 8000]
vpvs_data = [1.73, 1.75, 1.8, 1.75]
strike_data = [0, 20, 310, 310]
dip_data = [0, 5, 15, 15]

struct_data = prs.Model(thickn=thickn_data,
                        rho=rho_data,
                        vp=vp_data,
                        vpvs=vpvs_data,
                        strike=strike_data,
                        dip=dip_data)

N_data = 180

a, b = 0, 360
baz_data = (b - a) * rd.random_sample(size=N_data) + a

a, b = .04, .08
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

phaselist_header = np.array(gen_phaselist(struct_data.nlay, ['PS', 'PpS', 'PsS']))
RF_time, RF_transamp, RF_array, RF_result = predict_RF(struct_data, geomray_data, run_control, type_filt, freq_filt, phaselist_header, invert_tau_ratio, return_rf=True)
RF_transpol = RF_transamp / np.abs(RF_transamp)

# %% Save data

header = ['baz_degrees', 'slowness_s/km'] + list(phaselist_header)

sig_time = .1
RF_time_sigma = np.full(RF_time.shape, sig_time)
gamma_pol = .1

RF_transpol_gamma = np.full(RF_transpol.shape, gamma_pol)

tau_PS = (RF_time[:, 2] - RF_time[:, 1])
tau_PpS = (RF_time[:, 5] - RF_time[:, 4])
tau_PsS = (RF_time[:, 8] - RF_time[:, 7])
RF_time_sigma[:, -2] = np.abs(tau_PpS/tau_PS) * ((2*RF_time_sigma[:, 2]/tau_PS) + (2*RF_time_sigma[:, 5]/tau_PpS))
RF_time_sigma[:, -1] = np.abs(tau_PsS/tau_PS) * ((2*RF_time_sigma[:, 2]/tau_PS) + (2*RF_time_sigma[:, 8]/tau_PsS))

save_array_synthe(baz_data, slow_data, RF_time, datadir, header + ['(P0p3S-P0p2S)/(P3S-P2S)', '(P0s3S-P0s2S)/(P3S-P2S)'], 'data_time')
save_array_synthe(baz_data, slow_data, RF_time_sigma, datadir, header + ['(P0p3S-P0p2S)/(P3S-P2S)', '(P0s3S-P0s2S)/(P3S-P2S)'], 'data_time_sigma')
save_array_synthe(baz_data, slow_data, RF_transpol, datadir, header + ['t=0'], 'data_pol_trans')
save_array_synthe(baz_data, slow_data, RF_transamp, datadir, header + ['t=0'], 'data_amp_trans')
save_array_synthe(baz_data, slow_data, RF_transpol_gamma, datadir, header + ['t=0'], 'data_pol_trans_gamma')
save_RF_synthe_SAC(RF_result, dt, npts, datadir)

struct_data.write(datadir + '/structure_synthe_data.txt')

# %% Noised data

datadir += '_noise'
if not os.path.isdir(datadir):
    os.makedirs(datadir)
sig_noise = .075
RF_time_noise = np.zeros(RF_time.shape)

for k in range(9):

    if k in list(range(0, 9, 3)):
        RF_time_noise[:, k] += ss.norm.rvs(loc=0, scale=sig_noise, size=RF_time_noise[:, k].size)
    
    else:
        a = RF_time[:, k] - ( RF_time[:, k-1] + RF_time_noise[:, k-1])
        b = np.inf
        mu = 0

        RF_time_noise[:, k] += ss.truncnorm.rvs((-a-mu)/sig_noise, (b-mu)/sig_noise,
                                                    loc=mu, scale=sig_noise,
                                                    size=RF_time_noise[:, k].size)

RF_time += RF_time_noise

tau_PS = (RF_time[:, 2] - RF_time[:, 1])
tau_PpS = (RF_time[:, 5] - RF_time[:, 4])
tau_PsS = (RF_time[:, 8] - RF_time[:, 7])
RF_time[:, -2] = tau_PpS/tau_PS
RF_time[:, -1] = tau_PsS/tau_PS
RF_time_sigma[:, -2] = np.abs(tau_PpS/tau_PS) * ((2*RF_time_sigma[:, 2]/tau_PS) + (2*RF_time_sigma[:, 5]/tau_PpS))
RF_time_sigma[:, -1] = np.abs(tau_PsS/tau_PS) * ((2*RF_time_sigma[:, 2]/tau_PS) + (2*RF_time_sigma[:, 8]/tau_PsS))

save_array_synthe(baz_data, slow_data, RF_time, datadir, header + ['(P0p3S-P0p2S)/(P3S-P2S)', '(P0s3S-P0s2S)/(P3S-P2S)'], 'data_time')
save_array_synthe(baz_data, slow_data, RF_time_sigma, datadir, header + ['(P0p3S-P0p2S)/(P3S-P2S)', '(P0s3S-P0s2S)/(P3S-P2S)'], 'data_time_sigma')
save_array_synthe(baz_data, slow_data, RF_transpol, datadir, header + ['t=0'], 'data_pol_trans')
save_array_synthe(baz_data, slow_data, RF_transamp, datadir, header + ['t=0'], 'data_amp_trans')
save_array_synthe(baz_data, slow_data, RF_transpol_gamma, datadir, header + ['t=0'], 'data_pol_trans_gamma')
save_RF_synthe_SAC(RF_result, dt, npts, datadir)

struct_data.write(datadir + '/structure_synthe_data.txt')


# %% Reduce coverage to observed coverage at SERG

datadir += '_degcov'
if not os.path.isdir(datadir):
    os.makedirs(datadir)

baz = pd.read_csv('/Users/emile/Documents/Etude/2023_2024_Cesure/Stage_Mines/data/receiver_functions_CRL/SERG/data_inversion/data_time.csv', sep=';').baz_degrees.values
p = pd.read_csv('/Users/emile/Documents/Etude/2023_2024_Cesure/Stage_Mines/data/receiver_functions_CRL/SERG/data_inversion/data_time.csv', sep=';')['slowness_s/km'].values

geomray_data = prs.Geometry(baz=baz, slow=p)

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

RF_time, RF_transamp, RF_array, RF_result = predict_RF(struct_data, geomray_data, run_control, type_filt, freq_filt, phaselist_header, invert_tau_ratio, return_rf=True)
RF_transpol = RF_transamp / np.abs(RF_transamp)
RF_time_sigma = np.full(RF_time.shape, sig_time)
RF_transpol_gamma = np.full(RF_transpol.shape, gamma_pol)

RF_time_noise = np.zeros(RF_time.shape)

for k in range(9):

    if k in list(range(0, 9, 3)):
        RF_time_noise[:, k] += ss.norm.rvs(loc=0, scale=sig_noise, size=RF_time_noise[:, k].size)
    
    else:
        a = RF_time[:, k] - ( RF_time[:, k-1] + RF_time_noise[:, k-1])
        b = np.inf
        mu = 0

        RF_time_noise[:, k] += ss.truncnorm.rvs((-a-mu)/sig_noise, (b-mu)/sig_noise,
                                                    loc=mu, scale=sig_noise,
                                                    size=RF_time_noise[:, k].size)

RF_time += RF_time_noise

tau_PS = (RF_time[:, 2] - RF_time[:, 1])
tau_PpS = (RF_time[:, 5] - RF_time[:, 4])
tau_PsS = (RF_time[:, 8] - RF_time[:, 7])
RF_time[:, -2] = tau_PpS/tau_PS
RF_time[:, -1] = tau_PsS/tau_PS
RF_time_sigma[:, -2] = np.abs(tau_PpS/tau_PS) * ((2*RF_time_sigma[:, 2]/tau_PS) + (2*RF_time_sigma[:, 5]/tau_PpS))
RF_time_sigma[:, -1] = np.abs(tau_PsS/tau_PS) * ((2*RF_time_sigma[:, 2]/tau_PS) + (2*RF_time_sigma[:, 8]/tau_PsS))

save_array_synthe(baz, p, RF_time, datadir, header + ['(P0p3S-P0p2S)/(P3S-P2S)', '(P0s3S-P0s2S)/(P3S-P2S)'], 'data_time')
save_array_synthe(baz, p, RF_time_sigma, datadir, header + ['(P0p3S-P0p2S)/(P3S-P2S)', '(P0s3S-P0s2S)/(P3S-P2S)'], 'data_time_sigma')
save_array_synthe(baz, p, RF_transpol, datadir, header + ['t=0'], 'data_pol_trans')
save_array_synthe(baz, p, RF_transamp, datadir, header + ['t=0'], 'data_amp_trans')
save_array_synthe(baz, p, RF_transpol_gamma, datadir, header + ['t=0'], 'data_pol_trans_gamma')
save_RF_synthe_SAC(RF_result, dt, npts, datadir)

struct_data.write(datadir + '/structure_synthe_data.txt')

# %% Deg pick

datadir += '_degpick'
if not os.path.isdir(datadir):
    os.makedirs(datadir)

mask_time = np.logical_not(np.isfinite(pd.read_csv('/Users/emile/Documents/Etude/2023_2024_Cesure/Stage_Mines/data/receiver_functions_CRL/SERG/data_inversion/data_time.csv', sep=';').values[:, 2:]))
RF_time[mask_time] = np.nan
RF_time_sigma[mask_time] = np.nan

mask_amp = np.logical_not(np.isfinite(pd.read_csv('/Users/emile/Documents/Etude/2023_2024_Cesure/Stage_Mines/data/receiver_functions_CRL/SERG/data_inversion/data_amp_trans.csv', sep=';').values[:, 2:]))
RF_transpol[mask_amp] = np.nan
RF_transamp[mask_amp] = np.nan
RF_transpol_gamma[mask_amp] = np.nan

save_array_synthe(baz, p, RF_time, datadir, header + ['(P0p3S-P0p2S)/(P3S-P2S)', '(P0s3S-P0s2S)/(P3S-P2S)'], 'data_time')
save_array_synthe(baz, p, RF_time_sigma, datadir, header + ['(P0p3S-P0p2S)/(P3S-P2S)', '(P0s3S-P0s2S)/(P3S-P2S)'], 'data_time_sigma')
save_array_synthe(baz, p, RF_transpol, datadir, header + ['t=0'], 'data_pol_trans')
save_array_synthe(baz, p, RF_transamp, datadir, header + ['t=0'], 'data_amp_trans')
save_array_synthe(baz, p, RF_transpol_gamma, datadir, header + ['t=0'], 'data_pol_trans_gamma')
save_RF_synthe_SAC(RF_result, dt, npts, datadir)

struct_data.write(datadir + '/structure_synthe_data.txt')

exit(0)