#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 09:35:53 2023

@author: Emile DENISE
"""

# %% Packages

from tools.rfbi_tools import *


# %% Read arguments and check

parser = argparse.ArgumentParser(description='Generate directories, config file, and copy the data.')
parser.add_argument('wkdir', help='Path to the working directory to be created.')
parser.add_argument('datadir', help='Path to the inversion data directory.', type=is_dir_path)
parser.add_argument('RFdir', help='Path to the RF directory.', type=is_dir_path)

args = parser.parse_args()
wkdir = normalize_path(args.wkdir)
datadir = normalize_path(args.datadir)
RFdir = normalize_path(args.RFdir)


# %% Create working directory

print('Creating directories...')

if os.path.exists(wkdir):
    raise NameError("The working directory already exists.")
os.makedirs(wkdir)
os.makedirs(wkdir + "/data")
os.makedirs(wkdir + "/data/RF")
os.makedirs(wkdir + "/figures")
os.makedirs(wkdir + "/models")


# %% Create config file

config = configparser.ConfigParser()
config['INPUT'] = {'wkdir': wkdir,
                   'datadir': '{:}/data'.format(wkdir),
                   'plotdir': '{:}/figures'.format(wkdir),
                   'outdir': '{:}/models'.format(wkdir)}
update_config(config, wkdir)


# %% Copy the data

print('Copying data...')

files_data = [f for f in os.listdir(datadir) if os.path.isfile(os.path.join(datadir, f))]
for f in files_data:
    shutil.copy(datadir + '/' + f, wkdir + '/data/' + f)

files_data = [f for f in os.listdir(RFdir) if os.path.isfile(os.path.join(RFdir, f))]
for f in files_data:
    shutil.copy(RFdir + '/' + f, wkdir + '/data/RF/' + f)


# %% RF parameters

stream = ob.read('{:}/*.r'.format(RFdir))

config['DATA'] = {'dt_data': '{0:f}'.format(stream[0].stats.delta),
                  'npts_data': '{0:d}'.format(stream[0].stats.npts),
                  'tmin_data': '-5',
                  'tmin_plot': '-1',
                  'tmax_plot': '40'}
update_config(config, wkdir)

print("################################################################\n" +
      "You need to complete the config file with the desired parameters\n" + 
      "################################################################\n")

exit(0)