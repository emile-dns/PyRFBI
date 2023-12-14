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


# %% Read arguments and check

parser = argparse.ArgumentParser(description='Prepare inversion')

parser.add_argument('config', help='Path to config file.')
parser.add_argument('sampling', help='Method for the sampling of the a posteriori pdf (among metropolis, adaptative_metropolis)',
                    choices=['metropolis', 'adaptative_metropolis', 'grid_search'])

args = parser.parse_args()

path_config = args.config

if not os.path.isfile(path_config):
    msg = "The config file does not exist."
    raise NameError(msg)

config = configparser.ConfigParser()
config.read(path_config)
wkdir = config['INPUT']['wkdir']

sampling = args.sampling

# %% Create output directory

if not os.path.exists(wkdir + "/models"):
    os.makedirs(wkdir + "/models")

# %% Number of inverted parameters

param_list = pd.read_csv(wkdir + '/parameters_inversion.csv', sep=';')
n_params = np.sum(param_list['param_type'] == 'inv')

# %% Update config file

config['INVERSION SETUP'] = {'n_params': '%i' %n_params,
                             'target_phases': 'PS,PpS,PsS',
                             'dt': '0.025',
                             'npts': '5000', 
                             'filter_freq': '0.05,0.5',
                             'log_pdf': 'True',
                             'verbose': 'True',
                             'invert_polarity': 'True',
                             'invert_arrival_time': 'True', 
                             'sigma_amp': '1E-5',
                             'forward_parallel': 'True',
                             'n_proc': '4'}

config['INVERSION SETUP']['sampling'] = sampling

if sampling in ['metropolis', 'adaptative_metropolis']:
    config['INVERSION SETUP']['n_accepted'] = ''
    config['INVERSION SETUP']['n_max'] = ''
    config['INVERSION SETUP']['n_burn'] = ''
    config['INVERSION SETUP']['denom_proposal'] = '10'

if sampling == 'adaptative_metropolis':
    config['INVERSION SETUP']['sd'] = '%.4f' % (2.4**2/n_params)
    config['INVERSION SETUP']['epsilon'] = ''
    config['INVERSION SETUP']['t0'] = ''

if sampling == 'grid_search':
    config['INVERSION SETUP']['n_sample'] = '5'

with open(wkdir + '/rfbi.ini', 'w') as configfile:
    configfile.write("###########################################\n" +
                     "###### Configuration file for pyRFBI ######\n" +
                     "###########################################\n\n")
    config.write(configfile)

print("################################################################\n" +
      "You need to complete the config file with the desired parameters\n" + 
      "################################################################\n")

exit(0)