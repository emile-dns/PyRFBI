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
import scipy.interpolate as si
import pandas as pd

import matplotlib.pyplot as plt
from pyraysum import prs

from tools.rfbi_tools import *
from tools.rfbi_tools_plot import *
from tools.rfbi_tools_forward_pb import *

def save_array_synthe(baz, slow, data_array, datadir, header, name):
    idx_sort = np.argsort(baz)
    df = pd.DataFrame(data=np.concatenate((baz[idx_sort][:, np.newaxis], slow[idx_sort][:, np.newaxis], data_array[idx_sort]), axis=1),
                  columns=header)
    df.to_csv(datadir + '/' + name + '.csv', index=False, sep=';')

# %% 

datadir = '/Users/emile/Documents/Etude/2023_2024_Cesure/Stage_Mines/test/data'

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

N_data = 100

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

phaselist_header = gen_phaselist(struct_data.nlay, ['PS', 'PpS', 'PsS'])
RF_data, RF_arrival, RF_pol, RF_amp = predict_RF(struct_data, geomray_data, run_control, freq_filt, phaselist_header)

# %% Save data

header = ['baz_degrees', 'slowness_s/km'] + phaselist_header

sig_time = .15
RF_arrival_sigma = np.full(RF_arrival.shape, sig_time)
gamma_pol = 0.
RF_pol_gamma = np.full(RF_pol.shape, gamma_pol)

tau_PS = (RF_arrival[:, 2] - RF_arrival[:, 1])[..., np.newaxis]
tau_PpS = (RF_arrival[:, 5] - RF_arrival[:, 4])[..., np.newaxis]
tau_PsS = (RF_arrival[:, 8] - RF_arrival[:, 7])[..., np.newaxis]
RF_arrival_sigma[:, -2] = np.sqrt((RF_arrival_sigma[:, 2][..., np.newaxis]/tau_PS)**2 + (RF_arrival_sigma[:, 5][..., np.newaxis]/tau_PpS)**2).T
RF_arrival_sigma[:, -1] = np.sqrt((RF_arrival_sigma[:, 2][..., np.newaxis]/tau_PS)**2 + (RF_arrival_sigma[:, 8][..., np.newaxis]/tau_PsS)**2).T

save_array_synthe(baz_data, slow_data, RF_arrival, datadir, header + ['tau2/tau1', 'tau3/tau1'], 'data_time')
save_array_synthe(baz_data, slow_data, RF_arrival_sigma, datadir, header + ['tau2/tau1', 'tau3/tau1'], 'data_time_sigma')
save_array_synthe(baz_data, slow_data, RF_pol, datadir, header, 'data_pol_trans')
save_array_synthe(baz_data, slow_data, RF_pol_gamma, datadir, header, 'data_pol_trans_gamma')
save_array_synthe(baz_data, slow_data, RF_amp, datadir, header, 'data_amp_trans')

plot_struct(struct_data, 100, 'structure_synthe_data', datadir)
plot_geomray(geomray_data, 'geometry_rays', datadir)
struct_data.write(datadir + '/structure_synthe_data.txt')
geomray_data.write(datadir + '/geometry_rays.txt')

plot_RF_mesh(RF_data, baz_data, slow_data, dt, plotdir=datadir)
plot_RF_plate(RF_data, baz_data, slow_data, dt, plotdir=datadir)

data = [RF_arrival, RF_arrival_sigma, RF_pol, RF_amp]
prediction = [np.full(RF_arrival.shape, np.nan), np.full(RF_pol.shape, np.nan), np.full(RF_amp.shape, np.nan)]

plot_dataVSmodel(data, prediction, baz_data, slow_data, 'data', phaselist_header, datadir, tmax=45)


# %% Noised data

datadir += '_noise'
if not os.path.isdir(datadir):
    os.makedirs(datadir)
sig_noise = .1
RF_arrival_noise = np.zeros(RF_arrival.shape)

for k in range(9):

    if k in list(range(0, 9, 3)):
        RF_arrival_noise[:, k] += ss.norm.rvs(loc=0, scale=sig_noise, size=RF_arrival_noise[:, k].size)
    
    else:
        a = RF_arrival[:, k] - ( RF_arrival[:, k-1] + RF_arrival_noise[:, k-1])
        b = np.inf
        mu = 0

        RF_arrival_noise[:, k] += ss.truncnorm.rvs((-a-mu)/sig_noise, (b-mu)/sig_noise,
                                                    loc=mu, scale=sig_noise,
                                                    size=RF_arrival_noise[:, k].size)

RF_arrival += RF_arrival_noise

tau_PS = (RF_arrival[:, 2] - RF_arrival[:, 1])[..., np.newaxis]
tau_PpS = (RF_arrival[:, 5] - RF_arrival[:, 4])[..., np.newaxis]
tau_PsS = (RF_arrival[:, 8] - RF_arrival[:, 7])[..., np.newaxis]
RF_arrival[:, -2] = (tau_PpS/tau_PS).T
RF_arrival[:, -1] = (tau_PsS/tau_PS).T
RF_arrival_sigma[:, -2] = np.sqrt((RF_arrival_sigma[:, 2][..., np.newaxis]/tau_PS)**2 + (RF_arrival_sigma[:, 5][..., np.newaxis]/tau_PpS)**2).T
RF_arrival_sigma[:, -1] = np.sqrt((RF_arrival_sigma[:, 2][..., np.newaxis]/tau_PS)**2 + (RF_arrival_sigma[:, 8][..., np.newaxis]/tau_PsS)**2).T

save_array_synthe(baz_data, slow_data, RF_arrival, datadir, header + ['tau2/tau1', 'tau3/tau1'], 'data_time')
save_array_synthe(baz_data, slow_data, RF_arrival_sigma, datadir, header + ['tau2/tau1', 'tau3/tau1'], 'data_time_sigma')
save_array_synthe(baz_data, slow_data, RF_pol, datadir, header, 'data_pol_trans')
save_array_synthe(baz_data, slow_data, RF_pol_gamma, datadir, header, 'data_pol_trans_gamma')
save_array_synthe(baz_data, slow_data, RF_amp, datadir, header, 'data_amp_trans')

plot_struct(struct_data, 100, 'structure_synthe_data', datadir)
plot_geomray(geomray_data, 'geometry_rays', datadir)
struct_data.write(datadir + '/structure_synthe_data.txt')
geomray_data.write(datadir + '/geometry_rays.txt')

plot_RF_mesh(RF_data, baz_data, slow_data, dt, plotdir=datadir)
plot_RF_plate(RF_data, baz_data, slow_data, dt, plotdir=datadir)

data = [RF_arrival, RF_arrival_sigma, RF_pol, RF_amp]
prediction = [np.full(RF_arrival.shape, np.nan), np.full(RF_pol.shape, np.nan), np.full(RF_amp.shape, np.nan)]

plot_dataVSmodel(data, prediction, baz_data, slow_data, 'data', phaselist_header, datadir, tmax=45)

# %% Reduce picked data

datadir += '_degpick'
if not os.path.isdir(datadir):
    os.makedirs(datadir)

for i, alpha in enumerate([.95, .65, .25]):
    for j in range(3):

        idx = rd.choice(np.arange(0, N_data), int(N_data*(1-alpha)), replace=False)
        RF_arrival[idx, 3*i+j] = np.nan
        RF_arrival_sigma[idx, 3*i+j] = np.nan
        RF_pol[idx, 3*i+j] = np.nan
        RF_pol_gamma[idx, 3*i+j] = np.nan
        RF_amp[idx, 3*i+j] = np.nan

tau_PS = (RF_arrival[:, 2] - RF_arrival[:, 1])[..., np.newaxis]
tau_PpS = (RF_arrival[:, 5] - RF_arrival[:, 4])[..., np.newaxis]
tau_PsS = (RF_arrival[:, 8] - RF_arrival[:, 7])[..., np.newaxis]
RF_arrival[:, -2] = (tau_PpS/tau_PS).T
RF_arrival[:, -1] = (tau_PsS/tau_PS).T
RF_arrival_sigma[:, -2] = np.sqrt((RF_arrival_sigma[:, 2][..., np.newaxis]/tau_PS)**2 + (RF_arrival_sigma[:, 5][..., np.newaxis]/tau_PpS)**2).T
RF_arrival_sigma[:, -1] = np.sqrt((RF_arrival_sigma[:, 2][..., np.newaxis]/tau_PS)**2 + (RF_arrival_sigma[:, 8][..., np.newaxis]/tau_PsS)**2).T

save_array_synthe(baz_data, slow_data, RF_arrival, datadir, header + ['tau2/tau1', 'tau3/tau1'], 'data_time')
save_array_synthe(baz_data, slow_data, RF_arrival_sigma, datadir, header + ['tau2/tau1', 'tau3/tau1'], 'data_time_sigma')
save_array_synthe(baz_data, slow_data, RF_pol, datadir, header, 'data_pol_trans')
save_array_synthe(baz_data, slow_data, RF_pol_gamma, datadir, header, 'data_pol_trans_gamma')
save_array_synthe(baz_data, slow_data, RF_amp, datadir, header, 'data_amp_trans')

plot_struct(struct_data, 100, 'structure_synthe_data', datadir)
plot_geomray(geomray_data, 'geometry_rays', datadir)
struct_data.write(datadir + '/structure_synthe_data.txt')
geomray_data.write(datadir + '/geometry_rays.txt')

plot_RF_mesh(RF_data, baz_data, slow_data, dt, plotdir=datadir)
plot_RF_plate(RF_data, baz_data, slow_data, dt, plotdir=datadir)

data = [RF_arrival, RF_arrival_sigma, RF_pol, RF_amp]
prediction = [np.full(RF_arrival.shape, np.nan), np.full(RF_pol.shape, np.nan), np.full(RF_amp.shape, np.nan)]

plot_dataVSmodel(data, prediction, baz_data, slow_data, 'data', phaselist_header, datadir, tmax=45)

# %% Reduce coverage to observed coverage at SERG

datadir += '_degcov'
if not os.path.isdir(datadir):
    os.makedirs(datadir)

data = pd.read_csv('/Users/emile/Desktop/list_SERG.txt', header=None, sep=' ', skipinitialspace=True)
baz = data.values[:, 9]
p = data.values[:, 12]
R = 6373.
deg2dist = 180 / np.pi / R
p *= deg2dist

f = si.interp1d(list(range(len(baz))), baz)
baz = f(np.linspace(0, len(baz)-1, N_data))

f = si.interp1d(list(range(len(p))), p)
p = f(np.linspace(0, len(p)-1, N_data))

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

RF_data, RF_arrival, RF_pol, RF_amp = predict_RF(struct_data, geomray_data, run_control, freq_filt, phaselist_header)
RF_arrival_sigma = np.full(RF_arrival.shape, sig_time)
RF_pol_gamma = np.full(RF_pol.shape, gamma_pol)

RF_arrival_noise = np.zeros(RF_arrival.shape)

for k in range(9):

    if k in list(range(0, 9, 3)):
        RF_arrival_noise[:, k] += ss.norm.rvs(loc=0, scale=sig_noise, size=RF_arrival_noise[:, k].size)
    
    else:
        a = RF_arrival[:, k] - ( RF_arrival[:, k-1] + RF_arrival_noise[:, k-1])
        b = np.inf
        mu = 0

        RF_arrival_noise[:, k] += ss.truncnorm.rvs((-a-mu)/sig_noise, (b-mu)/sig_noise,
                                                    loc=mu, scale=sig_noise,
                                                    size=RF_arrival_noise[:, k].size)

RF_arrival += RF_arrival_noise

for i, alpha in enumerate([.95, .65, .25]):
    for j in range(3):

        idx = rd.choice(np.arange(0, N_data), int(N_data*(1-alpha)), replace=False)
        RF_arrival[idx, 3*i+j] = np.nan
        RF_arrival_sigma[idx, 3*i+j] = np.nan
        RF_pol[idx, 3*i+j] = np.nan
        RF_pol_gamma[idx, 3*i+j] = np.nan
        RF_amp[idx, 3*i+j] = np.nan

tau_PS = (RF_arrival[:, 2] - RF_arrival[:, 1])[..., np.newaxis]
tau_PpS = (RF_arrival[:, 5] - RF_arrival[:, 4])[..., np.newaxis]
tau_PsS = (RF_arrival[:, 8] - RF_arrival[:, 7])[..., np.newaxis]
RF_arrival[:, -2] = (tau_PpS/tau_PS).T
RF_arrival[:, -1] = (tau_PsS/tau_PS).T
RF_arrival_sigma[:, -2] = np.sqrt((RF_arrival_sigma[:, 2][..., np.newaxis]/tau_PS)**2 + (RF_arrival_sigma[:, 5][..., np.newaxis]/tau_PpS)**2).T
RF_arrival_sigma[:, -1] = np.sqrt((RF_arrival_sigma[:, 2][..., np.newaxis]/tau_PS)**2 + (RF_arrival_sigma[:, 8][..., np.newaxis]/tau_PsS)**2).T

save_array_synthe(baz, p, RF_arrival, datadir, header + ['tau2/tau1', 'tau3/tau1'], 'data_time')
save_array_synthe(baz, p, RF_arrival_sigma, datadir, header + ['tau2/tau1', 'tau3/tau1'], 'data_time_sigma')
save_array_synthe(baz, p, RF_pol, datadir, header, 'data_pol_trans')
save_array_synthe(baz, p, RF_pol_gamma, datadir, header, 'data_pol_trans_gamma')
save_array_synthe(baz, p, RF_amp, datadir, header, 'data_amp_trans')

plot_struct(struct_data, 100, 'structure_synthe_data', datadir)
plot_geomray(geomray_data, 'geometry_rays', datadir)
struct_data.write(datadir + '/structure_synthe_data.txt')
geomray_data.write(datadir + '/geometry_rays.txt')

plot_RF_mesh(RF_data, baz, p, dt, plotdir=datadir)
plot_RF_plate(RF_data, baz, p, dt, plotdir=datadir)

data = [RF_arrival, RF_arrival_sigma, RF_pol, RF_amp]
prediction = [np.full(RF_arrival.shape, np.nan), np.full(RF_pol.shape, np.nan), np.full(RF_amp.shape, np.nan)]

plot_dataVSmodel(data, prediction, baz, p, 'data', phaselist_header, datadir, tmax=45)

exit(0)