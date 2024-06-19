#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 10:52:15 2023

@author: Emile DENISE
"""

# %% Packages

from tools.rfbi_tools import *
from tools.rfbi_tools_plot import *

# %% Read arguments and check

parser = argparse.ArgumentParser(description='Plot data')
parser.add_argument('config', help='Path to config file.', type=is_file_path)

args = parser.parse_args()
path_config = args.config
config = configparser.ConfigParser()
config.read(path_config)

wkdir = config['INPUT']['wkdir']
datadir = config['INPUT']['datadir']
plotdir = config['INPUT']['plotdir']
outdir = config['INPUT']['outdir']

tmin_plot = config.getfloat('DATA', 'tmin_plot')
tmax_plot = config.getfloat('DATA', 'tmax_plot')
if config.getboolean('DATA', 'rf_plot'):
    npts_data = config.getint('DATA', 'npts_data')
    dt_data = config.getfloat('DATA', 'dt_data')
    tmin_data = config.getfloat('DATA', 'tmin_data')
n_layers = config.getint('STRUCTURE SETUP', 'n_layers')
phase_list = config['INVERSION SETUP']['target_phases'].split(',')
phase_list = gen_phaselist(n_layers, phase_list)
invert_tau_ratio = np.array(list(filter(None, config['INVERSION SETUP']['invert_tau_ratio'].split(',')))).astype(int)

DEG2KM = 111.2

# %% Load data

time_data =  make_masked_array_nan(pd.read_csv(datadir + '/data_time.csv', sep=';').values[:, 2:])
time_data_sig = make_masked_array_nan(pd.read_csv(datadir + '/data_time_sigma.csv', sep=';').values[:, 2:])
transamp_data = make_masked_array_nan(pd.read_csv(datadir + '/data_amp_trans.csv', sep=';').values[:, 2:])
transpol_data_gamma = make_masked_array_nan(pd.read_csv(datadir + '/data_pol_trans_gamma.csv', sep=';').values[:, 2:])

baz_data = pd.read_csv(datadir + '/data_time.csv', sep=';').values[:, 0]
p_data = pd.read_csv(datadir + '/data_time.csv', sep=';').values[:, 1]
geomray_data = prs.Geometry(baz=baz_data, slow=p_data)

if config.getboolean('DATA', 'rf_plot'):
    rf_rad = ob.read(datadir + '/RF/*.r')
    rf_trans = ob.read(datadir + '/RF/*.t')

    baz_RF = np.array([rf_rad[k].stats.sac.baz for k in range(len(rf_rad))])
    p_RF = np.array([rf_rad[k].stats.sac.user3 for k in range(len(rf_rad))]) / DEG2KM

    rf_array = np.empty((geomray_data.ntr, 2, rf_rad[0].stats.npts))
    for k in range(geomray_data.ntr):
        rf_array[k, 0, :] = rf_rad[k].data
        rf_array[k, 1, :] = rf_trans[k].data

# %% Plot

plot_geomray(geomray_data, 'geometry_rays', plotdir)

if config.getboolean('DATA', 'rf_plot'):
    taxis = np.array([tmin_data + dt_data*k for k in range(npts_data)])
    plot_RF_mesh(rf_array, baz_RF, p_RF, taxis=taxis, rot='rtz', show=False, plotdir=plotdir, title='RF_mesh_data', tmin=tmin_plot, tmax=tmax_plot)
    plot_RF_plate(rf_array, baz_RF, p_RF, taxis=taxis, rot='rtz', show=False, plotdir=plotdir, title='RF_plate_data', tmin=tmin_plot, tmax=tmax_plot)

data = [time_data, time_data_sig, transamp_data, transpol_data_gamma]
prediction = [np.full(time_data.shape, np.nan), np.full(transamp_data.shape, np.nan)]

plot_dataVSmodel(data, prediction, baz_data, p_data, phase_list, invert_tau_ratio, show=False, plotdir=plotdir, title='data', tmax=tmax_plot)
