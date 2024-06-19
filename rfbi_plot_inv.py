#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 10:52:15 2023

@author: Emile DENISE
"""

# %% Packages

from tools.rfbi_tools import *
from tools.rfbi_tools_forward_pb import *
from tools.rfbi_tools_plot import *

# %% Read arguments and check

parser = argparse.ArgumentParser(description='Plot inversion results')
parser.add_argument('config', help='Path to config file.', type=is_file_path)

args = parser.parse_args()
path_config = args.config
config = configparser.ConfigParser()
config.read(path_config)

wkdir = config['INPUT']['wkdir']
datadir = config['INPUT']['datadir']
plotdir = config['INPUT']['plotdir']
outdir = config['INPUT']['outdir']

type_filt = config['INVERSION SETUP']['type_filt']
if type_filt == 'bandpass':
    filter_freq = tuple(map(float, config['INVERSION SETUP']['filter_freq'].split(',')))
else:
    filter_freq = config.getfloat('INVERSION SETUP', 'filter_freq')
dt = config.getfloat('INVERSION SETUP', 'dt')
npts = config.getint('INVERSION SETUP', 'npts')
n_params = config.getint('INVERSION SETUP', 'n_params')
n_layers = config.getint('STRUCTURE SETUP', 'n_layers')
tmin_plot = config.getfloat('DATA', 'tmin_plot')
tmax_plot = config.getfloat('DATA', 'tmax_plot')
sampling = config['INVERSION SETUP']['sampling']
phases2extract = np.array(gen_phaselist(n_layers, config['INVERSION SETUP']['target_phases'].split(','))).astype(str)
invert_tau_ratio = np.array(list(filter(None, config['INVERSION SETUP']['invert_tau_ratio'].split(',')))).astype(int)
n_burn = config.getint('INVERSION SETUP', 'n_burn')

# %% Load data

time_data =  make_masked_array_nan(pd.read_csv(datadir + '/data_time.csv', sep=';').values[:, 2:])
time_data_sig = make_masked_array_nan(pd.read_csv(datadir + '/data_time_sigma.csv', sep=';').values[:, 2:])
transamp_data = make_masked_array_nan(pd.read_csv(datadir + '/data_amp_trans.csv', sep=';').values[:, 2:])
# to-do: find a way to represent the uncertainty on the data transverse amplitudes
transpol_data_gamma = make_masked_array_nan(pd.read_csv(datadir + '/data_pol_trans_gamma.csv', sep=';').values[:, 2:])

baz_data = pd.read_csv(datadir + '/data_time.csv', sep=';').values[:, 0]
slow_data = pd.read_csv(datadir + '/data_time.csv', sep=';').values[:, 1]

geomray_data = prs.Geometry(baz=baz_data, slow=slow_data)

# %% Load run conditions

param_set_list, param_set_values, param_inv_list, param_inv_prior, param_same_list, param_same_link = read_csv_struct(wkdir + '/parameters_inversion.csv')

run_control = prs.Control(
    verbose=False,
    rot='RTZ',
    dt=dt,
    npts=npts,
    align=0,
    mults=2
)

# %% Plots

z_accepted = np.load(outdir + '/accepted_models.npy')
z_rejected = np.load(outdir + '/rejected_models.npy')

plot_evol_model(z_accepted, 'accepted', param_inv_list, param_inv_prior, n_layers, 'slateblue', plotdir)
plot_evol_model(z_rejected, 'rejected', param_inv_list, param_inv_prior, n_layers, 'indianred', plotdir)
plot_evol_pdf(z_accepted, z_rejected, plotdir)
plot_marginals(z_accepted[n_burn:, :], param_inv_list, param_inv_prior, plotdir)
plot_confidence_marginals(z_accepted[n_burn:, :], param_inv_list, param_inv_prior, plotdir)

mean_model = np.nanmean(z_accepted[n_burn:, :n_params], axis=0)
median_model = np.nanmedian(z_accepted[n_burn:, :n_params], axis=0)
max_model = z_accepted[np.nanargmax(z_accepted[n_burn:, -1]), :n_params]

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
    
    struct.write(plotdir + '/' + names[i] + '_struct.txt')
    plot_struct(struct, 100, names[i], plotdir)

    RF_arrival, RF_amp, rfarray, _ = predict_RF(struct, geomray_data, run_control, type_filt, filter_freq, phases2extract, invert_tau_ratio, return_rf=True)

    taxis = np.array(range(npts))*dt-npts*dt/2
    plot_RF_mesh(rfarray, baz_data, slow_data, taxis=taxis, rot='rtz', show=False, plotdir=plotdir, title='RF_mesh_' + names[i], tmin=tmin_plot, tmax=tmax_plot)
    plot_RF_plate(rfarray, baz_data, slow_data, taxis=taxis, rot='rtz', show=False, plotdir=plotdir, title='RF_plate_' + names[i], tmin=tmin_plot, tmax=tmax_plot, fact=5)

    data = [time_data, time_data_sig, transamp_data, transpol_data_gamma]
    RF_arrival[np.logical_not(np.isfinite(time_data))] = np.nan
    RF_amp[np.logical_not(np.isfinite(transamp_data))] = np.nan
    prediction = [RF_arrival, RF_amp]

    plot_dataVSmodel(data, prediction, baz_data, slow_data, phases2extract, invert_tau_ratio, show=False, plotdir=plotdir, title=names[i] + '_dataVSpred', tmax=tmax_plot)

if sampling =='adaptative_metropolis':
    C = np.load(outdir + '/covariance_proposal.npy')
    t0 = config.getint('INVERSION SETUP', 't0')
    plot_cov_AMmatrix(C, t0, plotdir)

exit(0)