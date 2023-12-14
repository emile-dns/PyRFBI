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


# %% Read arguments and check

parser = argparse.ArgumentParser(description='Update config file and create .csv file to prepare inversion.')

parser.add_argument('config', help='Relative path to config file.')
parser.add_argument('n_layers', help='Number of layers in the strcuture (integer between 2 and 15).', type=int)
parser.add_argument('target_parameters', help='List of studied parameters (among thickn, rho, vp, vs, vpvs, dip, srike, flag, ani, trend, plunge).'+
                    'thickn, rho and vp are required; if vs or vpvs are not set, 1.73 will be taken as default for vpvs. See PyRaysum documentation for more details.')

args = parser.parse_args()

path_config = args.config

if not os.path.isfile(path_config):
    msg = "The config file does not exist."
    raise NameError(msg)

config = configparser.ConfigParser()
config.read(path_config)
wkdir = config['INPUT']['wkdir']

n_layers = args.n_layers

if n_layers < 2 or n_layers > 15:
    msg = "The number of layers must be between 2 and 15."
    raise ValueError(msg)

target_parameters = args.target_parameters.split(',')

for p in target_parameters:
    if p not in ['thickn', 'rho', 'vp', 'vs', 'flag', 'ani', 'trend', 'plunge', 'strike', 'dip', 'vpvs']:
        msg = "Parameters must be among: thickn, rho, vp, vs, flag, ani, trend, plunge, strike, dip, vpvs"
        raise ValueError(msg)

if not (('thickn' in target_parameters) and ('vp' in target_parameters) and ('rho' in target_parameters)):
    msg = "Thicknesses, P-waves velocities and densities are required."
    raise ValueError(msg)

# %% Update config file

config['STRUCTURE SETUP'] = {'n_layers':  "{n_layers:.0f}".format(n_layers=n_layers),
                             'target_parameters': ','.join(target_parameters)}

with open(wkdir + '/rfbi.ini', 'w') as configfile:
    configfile.write("###########################################\n" +
                     "###### Configuration file for pyRFBI ######\n" +
                     "###########################################\n\n")
    config.write(configfile)

# %% Create .csv file with parameters setup

n_params = len(target_parameters)

empty = np.full(n_layers*n_params, np.nan)

dict = {'param_name': np.repeat(target_parameters, n_layers),
        'nb_layer': np.tile(list(range(n_layers)), n_params),
        'param_type': empty,
        'param_info1': empty,
        'param_info2': empty,
        'param_info3': empty}
df = pd.DataFrame.from_dict(dict, dtype=str)

for p in ['dip', 'strike']:
    df.loc[(df['param_name'] == p) & (df['nb_layer'] == '0'), 'param_type'] = 'set'
    df.loc[(df['param_name'] == p) & (df['nb_layer'] == '0'), 'param_info1'] = '0'

df.loc[(df['param_name'] == 'thickn') & (df['nb_layer'] == '%i' %(n_layers-1)), 'param_type'] = 'set'
df.loc[(df['param_name'] == 'thickn') & (df['nb_layer'] == '%i' %(n_layers-1)), 'param_info1'] = '0'

df.to_csv(wkdir + "/parameters_inversion.csv", index=False, sep=';')

print("##########################################################\n" +
      "You need to complete parameters_inversion.csv             \n\n" +
      "The column param_type specifies the type of the parameter,\n" +
      "whether it is inverted for (inv), set (set) or identical  \n" +
      "to another parameter (same).                              \n\n" +
      "If set, param_info1 contains the value of the parameter.  \n\n" +
      "If same, param_info1 contains the name of the original pa-\n" +
      "rameter and param_info2 its layer number.                 \n\n"
      "If inv, param_info1 contains the type of the prior pdf    \n" +
      "(uniform or gaussian). If uniform, param_info2 contains   \n" +
      "the lower boundary and param_info3 the upper boundary of  \n" +
      "the prior. If gaussian, param_info2 contains the mean and \n" +
      "param_info3 the std of the prior.                         \n"
      "##########################################################")

exit(0)