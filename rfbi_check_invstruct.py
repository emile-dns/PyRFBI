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

parser = argparse.ArgumentParser(description='Check files with parameters setup for inversion.')

parser.add_argument('config', help='Path to config file.')

args = parser.parse_args()

path_config = args.config

if not os.path.isfile(path_config):
    msg = "The config file does not exist."
    raise NameError(msg)

config = configparser.ConfigParser()
config.read(path_config)
wkdir = config['INPUT']['wkdir']


# %% Un check sur ce qu'on remplit manuellement, ca coute pas grand chose à faire et c'est pratique mais pour l'instant c'est un peu du détail

exit(0)