#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 09:35:53 2023

@author: Emile DENISE
"""

import os
import argparse
import configparser
import pandas as pd
import numpy as np
from tools.rfbi_tools import *

# %% Read arguments and check

parser = argparse.ArgumentParser(description='Prepare inversion')
parser.add_argument('config', help='Path to config file.', type=file_path)
parser.add_argument('sampling', help='Method for the sampling of the a posteriori pdf (among metropolis, adaptative_metropolis)',
                    choices=['metropolis', 'adaptative_metropolis'])

args = parser.parse_args()
path_config = args.config
config = configparser.ConfigParser()
config.read(path_config)
wkdir = config['INPUT']['wkdir']
n_layers = config.getint('STRUCTURE SETUP', 'n_layers')

sampling = args.sampling

# %% Number of inverted parameters

param_list = pd.read_csv(wkdir + '/parameters_inversion.csv', sep=';')
n_params = np.sum(param_list['param_type'] == 'inv')

# %% Update config file

config['INVERSION SETUP'] = {'n_params': '%i' %n_params,
                             'target_phases': 'PS,PpS,PsS',
                             'invert_tau_ratio': ','.join(np.array(range(1, n_layers-1)).astype(str)),
                             'dt': '0.025',
                             'npts': '5000',
                             'type_filt': 'bandpass',
                             'filter_freq': '0.05,0.5',
                             'log_pdf': 'True',
                             'verbose': 'True',
                             'invert_arrival_time': 'True',
                             'invert_polarity': 'True',
                             'weight_polarity': '1',
                             'weight_pol2time': '.5',
                             'sigma_amp': '1E-5',
                             'forward_parallel': 'True',
                             'n_proc': '4'}

config['INVERSION SETUP']['sampling'] = sampling
config['INVERSION SETUP']['n_accepted'] = ''
config['INVERSION SETUP']['n_max'] = ''
config['INVERSION SETUP']['n_burn'] = ''
config['INVERSION SETUP']['denom_proposal'] = '10'

if sampling == 'adaptative_metropolis':
    config['INVERSION SETUP']['sd'] = '%.4f' % (2.4**2/n_params)
    config['INVERSION SETUP']['epsilon'] = ''
    config['INVERSION SETUP']['t0'] = ''

update_config(config, wkdir)

print("################################################################\n" +
      "You need to complete the config file with the desired parameters\n" + 
      "################################################################\n")

exit(0)