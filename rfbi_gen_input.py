#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 09:35:53 2023

@author: Emile DENISE
"""

# %% Packages

from tools.rfbi_tools import *

# %% Read arguments and check

parser = argparse.ArgumentParser(description='Generate empty input files.')
parser.add_argument('n_layers', help='Number of layers in the structure (integer between 2 and 15).', type=int, choices=range(2, 16))

args = parser.parse_args()
n_layers = args.n_layers

# %% Construct input files

phaselist = gen_phaselist(n_layers)

dict = {'baz_degrees': pd.Series(dtype='float'),
        'slowness_s/km': pd.Series(dtype='float')}

for phase in phaselist:
    dict[phase] = pd.Series(dtype='float')

# Time
dict_time = dict.copy()
for k in range(1, n_layers-1):
    dict_time['ΔPpS({:})/ΔPS({:})'.format(k, k)] = pd.Series(dtype='float')
    dict_time['ΔPsS({:})/ΔPS({:})'.format(k, k)] = pd.Series(dtype='float')

df_time = pd.DataFrame(dict_time)
df_time.to_csv('./pick_time.csv', index=False)
df_time.to_csv('./pick_time_error.csv', index=False)

# Polarity
dict_pol = dict.copy()
dict_pol['t=0'] = pd.Series(dtype='float')

df_pol = pd.DataFrame(dict_pol)
df_pol.to_csv('./pick_polarity.csv', index=False)
df_pol.to_csv('./pick_polarity_error.csv', index=False)