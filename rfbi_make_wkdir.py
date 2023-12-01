#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 09:35:53 2023

@author: Emile DENISE
"""

import os
import argparse
import configparser


# %% Read arguments and check

parser = argparse.ArgumentParser(description='Generate working directory, config file to complete and symbolic links of the data.')

parser.add_argument('wkdir', help='Absolute path to the desired working directory.')
parser.add_argument('datadir', help='Absolute path to the data directory.')

args = parser.parse_args()

wkdir = args.wkdir
datadir = args.datadir

if not os.path.exists(datadir):
    raise NameError("The data directory does not exist.")

if os.path.exists(wkdir):
    raise NameError("The working directory already exists.")


# %% Create working directory

os.makedirs(wkdir)    


# %% Create config file

config = configparser.ConfigParser()

config['INPUT'] = {'wkdir': wkdir,
                   'datadir': datadir}
 
with open(wkdir + '/rfbi.ini', 'w') as configfile:
    configfile.write("###########################################\n" +
                     "###### Configuration file for pyRFBI ######\n" +
                     "###########################################\n\n")
    config.write(configfile)

# %% Symlink for the data

os.makedirs(wkdir + "/data")

files_data = []

for (_, _, f) in os.walk(datadir):
    files_data.extend(f)

for f in files_data:
    os.symlink(datadir + '/' + f, wkdir + '/data/' + f)

exit(0)
