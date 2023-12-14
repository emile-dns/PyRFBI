#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 10:52:15 2023

@author: Emile DENISE
"""

import os
import argparse
import configparser
from pyraysum import prs

from tools.rfbi_tools import *
from tools.rfbi_tools_plot import *

# %% Read arguments and check

parser = argparse.ArgumentParser(description='Plot data')
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

n_layers = config.getint('STRUCTURE SETUP', 'n_layers')

# %% Load data

time_data =  make_masked_array_nan(pd.read_csv(datadir + '/data_time.csv', sep=';').values[:, 2:])

time_data_sig = make_masked_array_nan(pd.read_csv(datadir + '/data_time_sigma.csv', sep=';').values[:, 2:])

pol_data = make_masked_array_nan(pd.read_csv(datadir + '/data_pol_trans.csv', sep=';').values[:, 2:])

RF_pol_gamma = pd.read_csv(datadir + '/data_pol_trans_gamma.csv', sep=';')
pol_data_gamma = make_masked_array_nan(RF_pol_gamma.values[:, 2:])

amp_data = make_masked_array_nan(pd.read_csv(datadir + '/data_amp_trans.csv', sep=';').values[:, 2:])

baz_data = pd.read_csv(datadir + '/data_time.csv', sep=';').values[:, 0]
slow_data = pd.read_csv(datadir + '/data_time.csv', sep=';').values[:, 1]

geomray_data = prs.Geometry(baz=baz_data, slow=slow_data)

phaselist_header = gen_phaselist(n_layers, ['PS', 'PpS', 'PsS'])

# %% Plot

data = [time_data, time_data_sig, pol_data, amp_data]
prediction = [np.full(time_data.shape, np.nan), np.full(pol_data.shape, np.nan), np.full(amp_data.shape, np.nan)]

plot_dataVSmodel(data, prediction, baz_data, slow_data, 'data', phaselist_header, datadir, tmax=45)
