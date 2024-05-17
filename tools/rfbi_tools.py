#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 10:52:15 2023

@author: Emile DENISE
"""

import shutil
import os
import argparse
import configparser
import obspy as ob
import numpy as np
import pandas as pd
from pyraysum import prs


def dir_path(string):
    if os.path.isdir(string):
        return string
    else:
        raise NotADirectoryError(string)


def file_path(string):
    if os.path.isfile(string):
        return string
    else:
        raise FileNotFoundError(string)


def sec2hours(seconds):
    """
    convert seconds to hours, minutes, seconds
    """
    seconds = seconds % (24 * 3600)
    hour = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60
    return hour, minutes, seconds


def is_number(s):
    """ Returns True if string is a number. """
    try:
        float(s)
        return True
    except ValueError:
        return False


def make_masked_array_nan(array):
    """
    create a numpy.masked_array with NaN being masked
    """
    return np.ma.array(array, mask=np.isnan(array))


def find_idx_nearest(array, values):
    """
    Array: array where to find the nearest values to values (possibly containing nans)
    values : list of values to find the nearest elements in array
    """
    idx_nearest = []
    
    for v in values:
        if np.isfinite(v):
            idx = np.argmin(np.abs(np.array(array)-v))
            idx_nearest.append(idx)
        else:
            idx_nearest.append(np.nan)

    return idx_nearest


def find_values_idx(array, idx):
    """
    Array: array where to find the desired values 
    idx : list of indexes (possibly containing nans) to find elements in array
    """
    values = []

    for i in idx:
        if np.isfinite(i):
            values.append(array[i])
        else:
            values.append(np.nan)
    
    return values


def update_config(config, wkdir):
    with open(wkdir + '/rfbi.ini', 'w') as configfile:
        configfile.write("###########################################\n" +
                        "###### Configuration file for pyRFBI ######\n" +
                        "###########################################\n\n")
        config.write(configfile)


def get_param_unit(params):
    
    ref_params = ['thickn', 'rho', 'vp', 'vs', 'flag', 'ani', 'trend', 'plunge', 'strike', 'dip', 'vpvs']
    ref_params_names = ['Thickness', 'Density', 'V\u209a', 'V\u209b', 'Flag anisotropy', 'Anisotropy', 'Trend', 'Plunge', 'Strike', 'Dip', 'V\u209a/V\u209b']
    ref_params_names_latex = ['Thickness', 'Density', r'$V_P$', r'$V_S$', 'Flag anisotropy', 'Anisotropy', 'Trend', 'Plunge', 'Strike', 'Dip', r'$\frac{V_P}{V_S}$']
    ref_units = ['m', 'kg/m\u00b3', 'm/s', 'm/s', '', '%', '°', '°', '°', '°', '']
    ref_units_latex = [r'$m$', r'$kg.m^{-3}$', r'$m.s^{-1}$', r'$m.s^{-1}$', r'', r'$\%$', r'$°$', r'$°$', r'$°$', r'$°$', r'']

    l = []
    for k in range(len(params)):
        idx = np.where(np.array(ref_params) == params[k])[0][0]
        l.append([params[k], ref_params_names[idx], ref_params_names_latex[idx], ref_units[idx], ref_units_latex[idx]])
    
    return l    


def read_csv_struct(struct_csv):
    """
    """
    param_list = pd.read_csv(struct_csv, sep=';')

    param_inv = {}
    param_set = {}
    param_same = {}

    for k in range(len(param_list)):
    
        line = param_list.iloc[k]
    
        if line['param_type'] == 'inv':
            param_inv[(line['param_name'], line['nb_layer'])] = tuple(line[['param_info1', 'param_info2', 'param_info3']].values)
    
        elif line['param_type'] == 'set':
            param_set[(line['param_name'], line['nb_layer'])] = float(line['param_info1'])
    
        elif line['param_type'] == 'same':
            param_same[(line['param_name'], line['nb_layer'])] = (line['param_info1'], int(line['param_info2']))

    param_set_list = list(param_set.keys())
    param_set_values = [param_set.get(p) for p in param_set_list]

    param_inv_list = list(param_inv.keys())
    param_inv_prior = [param_inv.get(p) for p in param_inv_list]

    param_same_list = list(param_same.keys())
    param_same_link = [param_same.get(p) for p in param_same_list]

    return param_set_list, param_set_values, param_inv_list, param_inv_prior, param_same_list, param_same_link


def check_csv_struct(struct_csv):
    """
    """
    param_list = pd.read_csv(struct_csv, sep=';')

    for k in range(len(param_list)):
    
        line = param_list.iloc[k]

        if line['param_type'] not in ['inv', 'set', 'same']:
            msg = "param_type must be inv, set or same (" + line['param_name'] + " " + str(line['nb_layer']) + ")"
            raise ValueError(msg)
        
        if line['param_type'] == 'set':

            if pd.isna(line['param_info1']) or not is_number(line['param_info1']):
                msg = "param_info1 must be a number for set parameters (" + line['param_name'] + " " + str(line['nb_layer']) + ")"
                raise TypeError(msg)
        
            if np.sum(pd.isna(line[['param_info2', 'param_info3']].values)) != 2:
                msg = "param_info2 and param_info3 must be empty for set parameters (" + line['param_name'] + " " + str(line['nb_layer']) + ")"
                raise ValueError(msg)
        
        if line['param_type'] == 'inv':

            if line['param_info1'] not in ['uniform', 'gaussian']:
                msg = "param_info1 must be uniform or gaussian for inverted parameters (" + line['param_name'] + " " + str(line['nb_layer']) + ")"
                raise ValueError(msg)
            
            if not (np.sum(pd.isna(line[['param_info2', 'param_info3']].values)) == 0 and (is_number(line['param_info2']) and is_number(line['param_info3']))):
                msg = "param_info2 and param_info3 must be numbers for inverted parameters (" + line['param_name'] + " " + str(line['nb_layer']) + ")"
                raise TypeError(msg)
            
            if line['param_info1'] == 'uniform' and float(line['param_info2']) >= float(line['param_info3']):
                msg = "The bounds must be in increasing order for a uniform prior. (" + line['param_name'] + " " + str(line['nb_layer']) + ")"
                raise ValueError(msg)
            
            if line['param_info1'] == 'gaussian' and float(line['param_info3']) <= 0:
                msg = "The std must be positive for a gaussian prior. (" + line['param_name'] + " " + str(line['nb_layer']) + ")"
                raise ValueError(msg)

            
        if line['param_type'] == 'same':

            if not (line['param_info1'] in np.unique(param_list['param_name']) and line['param_info2'] in np.unique(param_list['nb_layer'])):
                msg = "param_info1 and param_info2 must an existing parameter for identical parameters (" + line['param_name'] + " " + str(line['nb_layer']) + ")"
                raise ValueError(msg)
            
            if line['param_info1'] != line['param_name']:
                msg = "The original and the identical parameters must represent the same physical quantity (" + line['param_name'] + " " + str(line['nb_layer']) + ")"
                raise ValueError(msg)

            if param_list[(param_list['param_name']==line['param_info1']) & (param_list['nb_layer']==line['param_info2'])]['param_type'].values == 'same':
                msg = "The original parameter must be a set or an inverted parameter (" + line['param_name'] + " " + str(line['nb_layer']) + ")"
                raise ValueError(msg)
            

            if not pd.isna(line['param_info3']):
                msg = "param_info3 must be empty for identical parameters (" + line['param_name'] + " " + str(line['nb_layer']) + ")"
                raise ValueError(msg)


def gen_struct_from_invsetsame(n_layers, param_set_list, param_set_values, param_inv_list, param_inv_values, param_same_list, param_same_link):

    struct = prs.Model(thickn=np.full(n_layers, np.nan),
                       rho=np.full(n_layers, np.nan),
                       vp=np.full(n_layers, np.nan))

    param_invset_list = np.concatenate((param_set_list, param_inv_list))
    idx = np.argsort(param_invset_list[:, 0])
    param_invset_list = param_invset_list[idx]
    param_invset_values = np.concatenate((param_set_values, param_inv_values))[idx]

    for k in range(len(param_invset_list)):
        struct[int(param_invset_list[k][1]), str(param_invset_list[k][0])] = float(param_invset_values[k])

    # for k in range(len(param_set_list)):
    #     struct[int(param_set_list[k][1]), str(param_set_list[k][0])] = float(param_set_values[k])

    # for k in range(len(param_inv_list)):
    #     struct[int(param_inv_list[k][1]), str(param_inv_list[k][0])] = float(param_inv_values[k])

    for k in range(len(param_same_list)):
        struct[int(param_same_list[k][1]), str(param_same_list[k][0])] = float(struct[param_same_link[k][1], param_same_link[k][0]])

    struct.update()
    return struct


def gen_phaselist(n_layers, waves=['PS', 'PpS', 'PsS']):
    """
    n_layers number of layers
    waves list among ['PS', 'PpS', 'PsS']
    """
    phaselist = []

    if 'PS' in waves:
        phaselist += ['P%iS' %k for k in range(1, n_layers)]
    
    if 'PpS' in waves:
        phaselist += ['P0p%iS' %k for k in range(1, n_layers)]
    
    if 'PsS' in waves:
        phaselist += ['P0s%iS' %k for k in range(1, n_layers)]

    return phaselist
