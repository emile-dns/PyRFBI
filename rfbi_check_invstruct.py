#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 10:52:15 2023

@author: Emile DENISE
"""

import os
import argparse
import configparser
from tools.rfbi_tools import *

# %% Read arguments and check

parser = argparse.ArgumentParser(description='Check files with parameters setup for inversion.')
parser.add_argument('config', help='Path to config file.', type=file_path)

args = parser.parse_args()
path_config = args.config
config = configparser.ConfigParser()
config.read(path_config)
wkdir = config['INPUT']['wkdir']

# %% Check csv file

check_csv_struct(wkdir + "/parameters_inversion.csv")

exit(0)