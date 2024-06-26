#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 09:35:53 2023

@author: Emile DENISE
"""

from tools.rfbi_tools import *

# %% Read arguments

parser = argparse.ArgumentParser(description="Generate directories and config file.")
parser.add_argument("wkdir", help="(Absolute or relative) path to the working directory to be created.")
parser.add_argument("datadir", help="(Absolute or relative) path to the directory containing the data files.", type=is_dir_path)

args = parser.parse_args()
wkdir = normalize_path(args.wkdir)
datadir = normalize_path(args.datadir)

# %% Create working directory

print("Creating directories...")

if os.path.exists(wkdir):
    raise NameError("The working directory already exists.")
os.makedirs(wkdir)
os.makedirs("{:}/data".format(wkdir))
os.makedirs("{:}/figures".format(wkdir))
os.makedirs("{:}/models".format(wkdir))

# %% Create config file

print("Creating config file...")

config = configparser.ConfigParser()
config["INPUT"] = {"wkdir": wkdir,
                   "datadir": "{:}/data".format(wkdir),
                   "plotdir": "{:}/figures".format(wkdir),
                   "outdir": "{:}/models".format(wkdir)}
update_config(config, wkdir)

# %% Copy the data

print("Copying data...")

files_data = [f for f in os.listdir(datadir) if os.path.isfile(os.path.join(datadir, f))]
for f in files_data:
    shutil.copy("{:}/{:}".format(datadir, f), "{:}/data/{:}".format(wkdir, f))

exit(0)
