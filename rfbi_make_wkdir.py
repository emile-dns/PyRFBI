#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 09:35:53 2023

@author: Emile DENISE
"""

from tools.rfbi_tools import *

# %% Read arguments and check

parser = argparse.ArgumentParser(description='Generate working directory, config file to complete and copy the data.')
parser.add_argument('wkdir',help='Absolute path to the working directory to be created.')
parser.add_argument('datadir', help='Absolute path to the inversion data directory.', type=dir_path)
parser.add_argument('RFdir', help='Absolute path to the RF directory.', type=dir_path)

args = parser.parse_args()
wkdir = args.wkdir
datadir = args.datadir
RFdir = args.RFdir

# %% Create working directory

if os.path.exists(wkdir):
    raise NameError("The working directory already exists.")
os.makedirs(wkdir)
os.makedirs(wkdir + "/data")
os.makedirs(wkdir + "/data/RF")
os.makedirs(wkdir + "/figures")
os.makedirs(wkdir + "/models")

# %% Create config file

config = configparser.ConfigParser()
config['INPUT'] = {'wkdir': wkdir + '/',
                   'datadir': wkdir + '/data/',
                   'plotdir': wkdir + '/figures/',
                   'outdir': wkdir + '/models/'}
update_config(config, wkdir)

# %% Copy the data

files_data = [f for f in os.listdir(datadir) if os.path.isfile(os.path.join(datadir, f))]
for f in files_data:
    shutil.copy(datadir + '/' + f, wkdir + '/data/' + f)

files_data = [f for f in os.listdir(RFdir) if os.path.isfile(os.path.join(RFdir, f))]
for f in files_data:
    shutil.copy(RFdir + '/' + f, wkdir + '/data/RF/' + f)

# %% RF parameters

stream = ob.read(RFdir + '*.r')

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