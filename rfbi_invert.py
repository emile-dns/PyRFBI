#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 09:35:53 2023

@author: Emile DENISE
"""

import os
import argparse
import configparser

import numpy as np
import numpy.linalg as la
import numpy.random as rd
import scipy.stats as ss
from pyraysum import prs
import pandas as pd
import time
import itertools as iter
import multiprocessing as mp

# %% Read arguments and check

parser = argparse.ArgumentParser(description='Prepare inversion')
parser.add_argument('config', help='Path to config file.')
args = parser.parse_args()

path_config = args.config

if not os.path.isfile(path_config):
    msg = "The config file does not exist."
    raise NameError(msg)

config = configparser.ConfigParser()
config.read(path_config)

wkdir = config['INPUT']['wkdir']
datadir = wkdir + '/data/'
outdir = wkdir + '/models/'

n_layers = config.getint('STRUCTURE SETUP', 'n_layers')

sampling = config['INVERSION SETUP']['sampling']
bool_logpdf = config.getboolean('INVERSION SETUP', 'log_pdf')
verbose = config.getboolean('INVERSION SETUP', 'verbose')

filter_freq = tuple(map(float, config['INVERSION SETUP']['filter_freq'].split(',')))
dt = config.getfloat('INVERSION SETUP', 'dt')
npts = config.getint('INVERSION SETUP', 'npts')
n_params = config.getint('INVERSION SETUP', 'n_params')
invert_polarity = config.getboolean('INVERSION SETUP', 'invert_polarity')
invert_arrival_time = config.getboolean('INVERSION SETUP', 'invert_arrival_time')
sigma_amp = config.getfloat('INVERSION SETUP', 'sigma_amp')
forward_parallel = config.getboolean('INVERSION SETUP', 'forward_parallel')

if forward_parallel:
    n_proc = config.getint('INVERSION SETUP', 'n_proc')

if sampling == 'metropolis':
    n_max = config.getint('INVERSION SETUP', 'n_max')
    n_accepted = config.getint('INVERSION SETUP', 'n_accepted')
    n_burn = config.getint('INVERSION SETUP', 'n_burn')
    denom_proposal = config.getfloat('INVERSION SETUP', 'denom_proposal')

if sampling == 'adaptative_metropolis':
    n_max = config.getint('INVERSION SETUP', 'n_max')
    n_accepted = config.getint('INVERSION SETUP', 'n_accepted')
    n_burn = config.getint('INVERSION SETUP', 'n_burn')
    denom_proposal = config.getfloat('INVERSION SETUP', 'denom_proposal')
    sd = config.getfloat('INVERSION SETUP', 'sd')
    epsilon = config.getfloat('INVERSION SETUP', 'epsilon')
    t0 = config.getint('INVERSION SETUP', 't0')

if sampling == 'grid_search':
    n_sample = config.getint('INVERSION SETUP', 'n_sample')
    grid_search_parallel = config.getboolean('INVERSION SETUP', 'grid_search_parallel')
    if grid_search_parallel:
        n_proc = config.getint('INVERSION SETUP', 'n_proc')


# %% Functions


def sec2hours(seconds):
    seconds = seconds % (24 * 3600)
    hour = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60
    return hour, minutes, seconds


def make_masked_array_nan(array):
    return np.ma.array(array, mask=np.isnan(array))


def read_csv_model(path_csv_model):
    """
    """
    param_list = pd.read_csv(path_csv_model, sep=';')

    n_inv = np.sum(param_list['param_type'] == 'inv')
    n_set = np.sum(param_list['param_type'] == 'set')
    n_same = np.sum(param_list['param_type'] == 'same')

    N = (n_inv, n_set, n_same)

    param_inv = {}
    param_set = {}
    param_same = {}

    for k in range(len(param_list)):
    
        line = param_list.iloc[k]
    
        if line['param_type'] == 'inv':
            param_inv[(line['param_name'], line['nb_layer'])] = (line['param_info1'], tuple(line[['param_info2', 'param_info3']].values))
    
        elif line['param_type'] == 'set':
            param_set[(line['param_name'], line['nb_layer'])] = float(line['param_info1'])
    
        elif line['param_type'] == 'same':
            param_same[(line['param_name'], line['nb_layer'])] = (line['param_info1'], int(line['param_info2']))

        else:
            print('error')

    param_set_list = list(param_set.keys())
    param_set_values = [param_set.get(p) for p in param_set_list]

    param_inv_list = list(param_inv.keys())
    param_inv_bounds = [param_inv.get(p)[1] for p in param_inv_list]

    param_same_list = list(param_same.keys())
    param_same_link = [param_same.get(p) for p in param_same_list]

    return param_set_list, param_set_values, param_inv_list, param_inv_bounds, param_same_list, param_same_link


def gen_struct_from_invsetsame(n_layers, param_set_list, param_set_values, param_inv_list, param_inv_values, param_same_list, param_same_link):

    list_parameters = ['thickn', 'rho', 'vp', 'vpvs', 'dip', 'strike']

    thickn = np.full(n_layers, np.nan)
    rho = np.full(n_layers, np.nan)
    vp = np.full(n_layers, np.nan)
    vpvs = np.full(n_layers, np.nan)
    strike = np.full(n_layers, np.nan)
    dip = np.full(n_layers, np.nan)
     
    struct = prs.Model(thickn=thickn,
                       rho=rho,
                       vp=vp,
                       vpvs=vpvs,
                       strike=strike,
                       dip=dip)

    for k in range(len(param_set_list)):
        struct[int(param_set_list[k][1]), str(param_set_list[k][0])] = float(param_set_values[k])

    for k in range(len(param_inv_list)):
        struct[int(param_inv_list[k][1]), str(param_inv_list[k][0])] = float(param_inv_values[k])

    for k in range(len(param_same_list)):
        struct[int(param_same_list[k][1]), str(param_same_list[k][0])] = float(struct[param_same_link[k][1], param_same_link[k][0]])

    return struct


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


def gen_phaselist2extract(n_layers, waves):
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

def extract_RF_timepolamp(result, phases2extract):
    """
    result is the result of run with run_full (RTZ rotation)
    Returns time peaks, their amplitudes, polarities, and the corresponding phase names with different naming systems.
    """

    n_RF = len(result.rfs)
    t_RF = result.rfs[0][0].stats.taxis

    t_arrival = []
    pol_trans = []
    amp_trans = []

    for k in range(n_RF):

        t_arrival.append([])
        pol_trans.append([])
        amp_trans.append([])

        # Arrival times of converted waves PS, PpS, PsS

        if 'P' in result.rfs[k][0].stats.conversion_names:
        
            t_P = result.rfs[k][0].stats.phase_times[result.rfs[k][0].stats.conversion_names == 'P'][0]

            for phase in phases2extract:

                if phase in result.rfs[k][0].stats.conversion_names:

                    t_arrival[k].append(result.rfs[k][0].stats.phase_times[result.rfs[k][0].stats.conversion_names == phase][0] - t_P)
                
                else:
                    t_arrival[k].append(np.nan)
        
        else: #si on a pas de P, on met des NaN partout

            t_arrival[k].append(len(phases2extract)*[np.nan])

        idx_arrival = find_idx_nearest(t_RF, t_arrival[k])

        # Amplitude of converted waves PS, PpS, PsS on the transverse component
        amp_trans[k] = find_values_idx(result.rfs[k][1].data, idx_arrival)

        # Polarity of converted waves PS, PpS, PsS on the transverse component
        pol_trans[k] = np.array(amp_trans[k]) / np.abs(amp_trans[k])

    return np.array(t_arrival), np.array(pol_trans), np.array(amp_trans)


def predict_RF(struct, geomray, run_control, freq_filt, phases2extract):

    result = prs.run(struct, geomray, run_control, rf=True)

    result.filter('rfs', 'bandpass', freqmin=freq_filt[0], freqmax=freq_filt[1], zerophase=True, corners=2)

    rfarray = np.empty((geomray.ntr, 2, run_control.npts))

    for k in range(geomray.ntr):

        rfarray[k, 0, :] = result.rfs[k][0].data
        rfarray[k, 1, :] = result.rfs[k][1].data

    RF_arrival, RF_pol, RF_amp = extract_RF_timepolamp(result, phases2extract)

    tau_PS = (RF_arrival[:, 2] - RF_arrival[:, 1])[..., np.newaxis]
    tau_PpS = (RF_arrival[:, 5] - RF_arrival[:, 4])[..., np.newaxis]

    if 'PsS' in phases2extract:
        tau_PsS = (RF_arrival[:, 8] - RF_arrival[:, 7])[..., np.newaxis]
        RF_arrival = np.concatenate((RF_arrival, tau_PpS/tau_PS, tau_PsS/tau_PS), axis=1)
    
    else:
        RF_arrival = np.concatenate((RF_arrival, tau_PpS/tau_PS), axis=1)

    return rfarray, RF_arrival, RF_pol, RF_amp


def predict_RF_par(struct, geomray_split, run_control, freq_filt, phases2extract, n_proc):

    with mp.get_context("fork").Pool(n_proc) as pool:
        result = pool.starmap(predict_RF, [(struct, geom, run_control, freq_filt, phases2extract)  for geom in geomray_split])

    rfarray, RF_arrival, RF_pol, RF_amp = result[0][0], result[0][1], result[0][2], result[0][3]
    pool.close()

    for k in range(1, n_proc):

        rfarray = np.vstack((rfarray, result[k][0]))
        RF_arrival = np.vstack((RF_arrival, result[k][1]))
        RF_pol = np.vstack((RF_pol, result[k][2]))
        RF_amp = np.vstack((RF_amp, result[k][3]))

    return rfarray, RF_arrival, RF_pol, RF_amp


def g(struct, geomray, run_control, freq_filt, sigma_amp, phases2extract):
    _, RF_arrival, RF_pol, RF_amp = predict_RF(struct, geomray, run_control, freq_filt, phases2extract)
    RF_sigamp = np.full(RF_amp.shape, sigma_amp)

    RF_arrival = make_masked_array_nan(RF_arrival)
    RF_pol = make_masked_array_nan(RF_pol)
    RF_amp = make_masked_array_nan(RF_amp)
    RF_sigamp = make_masked_array_nan(RF_sigamp)

    return RF_arrival, RF_pol, RF_amp, RF_sigamp


def g_par(struct, geomray, run_control, freq_filt, sigma_amp, phases2extract, n_proc):
    _, RF_arrival, RF_pol, RF_amp = predict_RF_par(struct, geomray, run_control, freq_filt, phases2extract, n_proc)
    RF_sigamp = np.full(RF_amp.shape, sigma_amp)

    RF_arrival = make_masked_array_nan(RF_arrival)
    RF_pol = make_masked_array_nan(RF_pol)
    RF_amp = make_masked_array_nan(RF_amp)
    RF_sigamp = make_masked_array_nan(RF_sigamp)

    return RF_arrival, RF_pol, RF_amp, RF_sigamp


def grid_search(p, z, n_proc, outdir):
    start_time = time.time()

    with mp.get_context("fork").Pool(n_proc) as pool:
        pz = pool.map(p, z)#, chunksize=np.ceil(len(z)/n_proc))

    np.save(outdir + '/grid_search_samples.npy', np.hstack((np.array(z), np.array(pz)[:, -1][:, np.newaxis])))
    pool.close()

    run_time = time.time() - start_time
    h, m, s = sec2hours(run_time)
    print('Runtime: %ih %im %.1fs' % (h, m, s))

    return pz


def metropolis(p, z0, q, n_accepted, n_burn, n_max, outdir, verbose):
    start_time = time.time()

    z_accepted = [z0 + [p(z0)]]
    z_rejected = []

    while len(z_accepted) <= n_accepted+n_burn and len(z_accepted)+len(z_rejected) <= n_max:

        if verbose and (len(z_accepted) % 50 == 0 or (len(z_rejected) + len(z_accepted)) % 50 == 0):
            print('accepted:%i total:%i accept. rate=%.2f' % (len(z_accepted), len(z_rejected) + len(z_accepted), len(z_accepted)/(len(z_rejected) + len(z_accepted))))

        z_candidate = list(q(z0))

        p_zcand = p(z_candidate)
        p_z0 = z_accepted[-1][-1]

        if p_zcand >= p_z0:
            z_accepted.append(z_candidate + [p_zcand])
            z0 = z_candidate
        
        else:
            a = rd.uniform(0., 1.)
            
            if a <= p_zcand / p_z0:
                z_accepted.append(z_candidate + [p_zcand])
                z0 = z_candidate
                
            else:
                z_rejected.append(z_candidate + [p_zcand])
        
        np.save(outdir + '/accepted_models.npy', z_accepted)
        np.save(outdir + '/rejected_models.npy', z_rejected)

    run_time = time.time() - start_time
    h, m, s = sec2hours(run_time)
    print('%i iterations' %(len(z_rejected) + len(z_accepted)))
    print('Runtime: %ih %im %.1fs' % (h, m, s))


def log_metropolis(logp, z0, q, n_accepted, n_burn, n_max, outdir, verbose):
    start_time = time.time()

    z_accepted = [z0 + list(logp(z0))]
    z_rejected = []

    while len(z_accepted) <= n_accepted+n_burn and len(z_accepted)+len(z_rejected) <= n_max:

        if verbose and (len(z_accepted) % 50 == 0 or (len(z_rejected) + len(z_accepted)) % 50 == 0):
            print('accepted:%i total:%i accept. rate=%.2f' % (len(z_accepted), len(z_rejected) + len(z_accepted), len(z_accepted)/(len(z_rejected) + len(z_accepted))))

        z_candidate = list(q(z0))

        logp_zcand = logp(z_candidate)
        logp_z0 = z_accepted[-1][-1]

        if logp_zcand[-1] >= logp_z0:
            z_accepted.append(z_candidate + list(logp_zcand))
            z0 = z_candidate
        
        else:
            a = np.log(rd.uniform(0., 1.))
            print(logp_zcand[-1], logp_z0, logp_zcand[-1] - logp_z0, a)
            
            if a <= logp_zcand[-1] - logp_z0:
                z_accepted.append(z_candidate + list(logp_zcand))
                z0 = z_candidate

            else:
                z_rejected.append(z_candidate + list(logp_zcand))
        
        if (len(z_accepted) + len(z_rejected))% 50 == 0:
            np.save(outdir + '/accepted_models.npy', z_accepted)
            np.save(outdir + '/rejected_models.npy', z_rejected)

    run_time = time.time() - start_time
    h, m, s = sec2hours(run_time)
    print('%i iterations' %(len(z_rejected) + len(z_accepted)))
    print('Runtime: %ih %im %.1fs' % (h, m, s))


def update_mean_AM(t, Xt, Xt_1_mean):

    Xt_mean = (t * Xt_1_mean + Xt) / (t+1)

    return Xt_mean, Xt_1_mean


def update_cov_AM(t, Ct_1, Xt_1, Xt_1_mean, Xt_2_mean, sd, eps, d):

    Ct = ( (t - 1) * Ct_1 + sd * ( t * Xt_2_mean[:, np.newaxis] * Xt_2_mean - (t+1) * Xt_1_mean[:, np.newaxis] * Xt_1_mean + Xt_1[:, np.newaxis] * Xt_1  + eps * np.eye(d) ) ) / t

    return Ct


def log_adaptative_metropolis(logp, z0, C0, d, sd, eps, t0, n_accepted, n_burn, n_max, outdir, verbose):

    start_time = time.time()

    z_accepted = [z0 + list(logp(z0))]
    z_rejected = []
    Xt_mean = np.array(z0)
    Ct = C0.copy()
    save_Ct = Ct.copy()[:, :, np.newaxis]

    while len(z_accepted) <= n_accepted+n_burn and len(z_accepted)+len(z_rejected) <= n_max:

        if verbose and (len(z_accepted) % 50 == 0 or (len(z_rejected) + len(z_accepted)) % 50 == 0):
            print('accepted:%i total:%i accept. rate=%.2f' % (len(z_accepted), len(z_rejected) + len(z_accepted), len(z_accepted)/(len(z_rejected) + len(z_accepted))))

        t = len(z_accepted) - 1

        if t <= t0:
            q = lambda z : ss.multivariate_normal(mean=z, cov=C0, allow_singular=True).rvs()

        else: #t>t0
            q = lambda z : ss.multivariate_normal(mean=z, cov=Ct, allow_singular=True).rvs()

        z_candidate = list(q(z0))

        logp_zcand = logp(z_candidate)
        logp_z0 = z_accepted[-1][-1]

        if logp_zcand[-1] >= logp_z0:
            z_accepted.append(z_candidate + list(logp_zcand))
            z0 = z_candidate

            # update
            t = len(z_accepted)-1
            Xt = np.array(z0)
            Xt_mean, Xt_1_mean = update_mean_AM(t, Xt, Xt_mean)
            if t >= 3:
                Ct = update_cov_AM(t, Ct, Xt, Xt_mean, Xt_1_mean, sd, eps, d)
                save_Ct = np.concatenate((save_Ct, Ct[..., np.newaxis]), axis=2)

        else:
            a = np.log(rd.uniform(0., 1.))
            
            if a <= logp_zcand[-1] - logp_z0:
                z_accepted.append(z_candidate + list(logp_zcand))
                z0 = z_candidate

                # update
                t = len(z_accepted)-1
                Xt = np.array(z0)
                Xt_mean, Xt_1_mean = update_mean_AM(t, Xt, Xt_mean)
                if t >= 3:
                    Ct = update_cov_AM(t, Ct, Xt, Xt_mean, Xt_1_mean, sd, eps, d)
                    save_Ct = np.concatenate((save_Ct, Ct[..., np.newaxis]), axis=2)

            else:
                z_rejected.append(z_candidate + list(logp_zcand))
        
        if (len(z_accepted) + len(z_rejected))% 50 == 0:
            np.save(outdir + '/accepted_models.npy', z_accepted)
            np.save(outdir + '/rejected_models.npy', z_rejected)
            np.save(outdir + '/covariance_proposal.npy', save_Ct)
    
    run_time = time.time() - start_time
    h, m, s = sec2hours(run_time)
    print('%i iterations' %(len(z_rejected) + len(z_accepted)))
    print('Runtime: %ih %im %.1fs' % (h, m, s))


# %% Inversion framework

class Mprior:

    def __init__(self, n_layers, param_set_list, param_set_values, param_inv_list, param_inv_bounds, param_same_list, param_same_link):
        self.n_layers = n_layers
        
        self.param_set_list = param_set_list
        self.param_set_values = param_set_values
        self.n_set = len(self.param_set_list)

        self.param_inv_list = param_inv_list
        self.param_inv_bounds = param_inv_bounds
        self.n_inv = len(self.param_inv_list)

        self.param_same_list = param_same_list
        self.param_same_link = param_same_link
        self.n_same = len(self.param_same_list)

    
    def sample(self):

        mprior_sample = [ss.uniform(loc=b[0], scale=b[1]-b[0]).rvs() for b in self.param_inv_bounds]
        struct = gen_struct_from_invsetsame(self.n_layers,
                                            self.param_set_list, self.param_set_values,
                                            self.param_inv_list, mprior_sample,
                                            self.param_same_list, self.param_same_link)

        return mprior_sample, struct

    def pdf(self, m):

        M = 1
        k = 0

        while k < self.n_inv and self.param_inv_bounds[k][0] <= m[k] and m[k] <= self.param_inv_bounds[k][1]:
            M *= 1 / (self.param_inv_bounds[k][1] - self.param_inv_bounds[k][0])
            k += 1
    
        if k != self.n_inv:
            M = 0

        return M

    def logpdf(self, m):

        M = 0
        k = 0

        while k < self.n_inv and self.param_inv_bounds[k][0] <= m[k] and m[k] <= self.param_inv_bounds[k][1]:
            M -= np.log(self.param_inv_bounds[k][1] - self.param_inv_bounds[k][0])
            k += 1
    
        if k != self.n_inv:
            M = -np.inf

        return M


class Ltime:

    def __init__(self, time_data, cov_data):
        self.time_data = time_data
        self.cov_data = cov_data

    def pdf(self, time_model):
        dt = time_model - self.time_data
        n = np.size(dt) - np.sum(dt.mask)
        return np.exp(-.5 * np.ma.dot(dt, np.ma.dot(la.inv(self.cov_data), dt)) / n)

    def logpdf(self, time_model):
        dt = time_model - self.time_data
        n = np.size(dt) - np.sum(dt.mask)
        return -.5 * np.ma.dot(dt, np.ma.dot(la.inv(self.cov_data), dt)) / n


class Lpolarity:

    def __init__(self, pol_data, gamma_data):
        self.pol_data = pol_data
        self.gamma_data = gamma_data

    def pdf(self, amp_model, sig_model):
        phi = self.gamma_data + (1 - 2 * self.gamma_data) * ss.norm.cdf(amp_model / sig_model)
        L = phi ** ((1 + self.pol_data) / 2) * (1 - phi) ** ((1 - self.pol_data) / 2)
        n = np.size(L) - np.sum(L.mask)
        return np.prod(L) ** (1/n)

    def logpdf(self, amp_model, sig_model):
        phi = self.gamma_data + (1 - 2 * self.gamma_data) * ss.norm.cdf(amp_model / sig_model)
        L = ((1 + self.pol_data) / 2) * np.log(phi) + ((1 - self.pol_data) / 2) * np.log(1 - phi)
        n = np.size(L) - np.sum(L.mask)
        return np.sum(L) / n


class Mposterior:

    def __init__(self,
                 phases2extract,
                 time_data, cov_time_data, pol_data, gamma_pol_data,
                 sigma_amp_model,
                 invert_polarity, invert_arrival_time,
                 n_layers, param_set_list, param_set_values, param_inv_list, param_inv_bounds, param_same_list, param_same_link,
                 geomray, run_control, freq_filt, parallel, n_proc=1, geomray_split=None):

        self.phases2extract = phases2extract
        
        self.time_data = time_data
        self.cov_time_data = cov_time_data
        self.pol_data = pol_data
        self.gamma_pol_data = gamma_pol_data

        self.invert_polarity = invert_polarity
        self.invert_arrival_time = invert_arrival_time

        self.sigma_amp = sigma_amp_model

        self.n_layers = n_layers
        
        self.param_set_list = param_set_list
        self.param_set_values = param_set_values
        self.n_set = len(self.param_set_list)

        self.param_inv_list = param_inv_list
        self.param_inv_bounds = param_inv_bounds
        self.n_inv = len(self.param_inv_list)

        self.param_same_list = param_same_list
        self.param_same_link = param_same_link
        self.n_same = len(self.param_same_list)
        
        self.geomray = geomray
        self.run_control = run_control
        self.freq_filt = freq_filt

        self.parallel = parallel
        self.n_proc = n_proc
        self.geomray_split = geomray_split

        ################

        self.Mprior = Mprior(self.n_layers,
                         self.param_set_list, self.param_set_values, self.param_inv_list, self.param_inv_bounds, self.param_same_list, self.param_same_link)
        self.Ltime = Ltime(self.time_data, self.cov_time_data)
        self.Lpolarity = Lpolarity(self.pol_data, self.gamma_pol_data)

    def pdf(self, m):
        struct = gen_struct_from_invsetsame(self.n_layers,
                                            self.param_set_list, self.param_set_values, self.param_inv_list, m, self.param_same_list, self.param_same_link)
        
        if self.parallel:
            time_model, _, amp_model, sig_amp_model = g_par(struct, self.geomray_split, self.run_control, self.freq_filt, self.sigma_amp, self.phases2extract, self.n_proc)
        else:
            time_model, _, amp_model, sig_amp_model = g(struct, self.geomray, self.run_control, self.freq_filt, self.sigma_amp, self.phases2extract)
        
        time_model = time_model.reshape(time_model.size)

        return self.Mprior.pdf(m) * self.Lpolarity.pdf(amp_model, sig_amp_model) * self.Ltime.pdf(time_model)

    def logpdf(self, m):

        if np.isfinite(self.Mprior.logpdf(m)):

            struct = gen_struct_from_invsetsame(self.n_layers,
                                                self.param_set_list, self.param_set_values, self.param_inv_list, m, self.param_same_list, self.param_same_link)
            
            if self.parallel:
                time_model, _, amp_model, sig_amp_model = g_par(struct, self.geomray_split, self.run_control, self.freq_filt, self.sigma_amp, self.phases2extract, self.n_proc)
            else:
                time_model, _, amp_model, sig_amp_model = g(struct, self.geomray, self.run_control, self.freq_filt, self.sigma_amp, self.phases2extract)

            time_model = time_model.reshape(time_model.size)

            if  np.sum(time_model.mask) >= 5:
                print('model error: %i/%i lacking' % (np.sum(time_model.mask), np.size(time_model)))

            a = self.Mprior.logpdf(m)
            b = self.invert_polarity * self.Lpolarity.logpdf(amp_model, sig_amp_model)
            c = self.invert_arrival_time * self.Ltime.logpdf(time_model)
            
            return a, b, c, a + b + c
        
        else:

            return -np.inf, np.nan, np.nan, -np.inf


# %% Load data

RF_tarrival = pd.read_csv(datadir + '/data_time.csv', sep=';')
time_data =  make_masked_array_nan(RF_tarrival.values[:, 2:].reshape(RF_tarrival.values[:, 2:].size))

RF_tarrival_sig = pd.read_csv(datadir + '/data_time_sigma.csv', sep=';')
time_data_cov = make_masked_array_nan(np.diag(RF_tarrival_sig.values[:, 2:].reshape(RF_tarrival_sig.values[:, 2:].size))**2)

RF_pol = pd.read_csv(datadir + '/data_pol_trans.csv', sep=';')
pol_data = make_masked_array_nan(RF_pol.values[:, 2:])

RF_pol_gamma = pd.read_csv(datadir + '/data_pol_trans_gamma.csv', sep=';')
pol_data_gamma = make_masked_array_nan(RF_pol_gamma.values[:, 2:])

baz_data = RF_tarrival['baz_degrees'].values
slow_data = RF_tarrival['slowness_s/km'].values

geomray_data = prs.Geometry(baz=baz_data, slow=slow_data)

if forward_parallel:

    split_baz = np.array_split(geomray_data.baz, n_proc)
    split_slow = np.array_split(geomray_data.slow, n_proc)
    split_dn = np.array_split(geomray_data.dn, n_proc)
    split_de = np.array_split(geomray_data.de, n_proc)

    geomray_data_split = [prs.Geometry(baz=split_baz[k],
                                       slow=split_slow[k],
                                       dn=split_dn[k],
                                       de=split_de[k]) for k in range(n_proc)]


# %% Load inversion setup

phases2extract = config['INVERSION SETUP']['target_phases'].split(',')
phases2extract = gen_phaselist2extract(n_layers, phases2extract)

param_set_list, param_set_values, param_inv_list, param_inv_bounds, param_same_list, param_same_link = read_csv_model(path_csv_model=wkdir+'/parameters_inversion.csv')

run_control_inv = prs.Control(verbose=False,
                              rot='RTZ',
                              dt=dt,
                              npts=npts,
                              align=0,
                              mults=2)

_, struct_phase = Mprior(n_layers, param_set_list, param_set_values, param_inv_list, param_inv_bounds, param_same_list, param_same_link).sample()

# run avec un tirage du prior pour obtenir la liste des phases à calculer
result = prs.run(struct_phase, geomray_data, run_control_inv)
phase_list = result.descriptors()
run_control_inv.set_phaselist(phase_list, equivalent=True)
result = phase_list = None

# %% Posterior pdf

if forward_parallel:

    Mpost = Mposterior(time_data=time_data,
                       phases2extract=phases2extract,
                    cov_time_data=time_data_cov,
                    pol_data=pol_data,
                    gamma_pol_data=pol_data_gamma,
                    invert_arrival_time=invert_arrival_time,
                    invert_polarity=invert_polarity,
                    sigma_amp_model=sigma_amp,
                    n_layers=n_layers,
                    param_set_list=param_set_list,
                    param_set_values=param_set_values,
                    param_inv_list=param_inv_list,
                    param_inv_bounds=param_inv_bounds,
                    param_same_list=param_same_list,
                    param_same_link=param_same_link,
                    geomray=geomray_data,
                    run_control=run_control_inv,
                    freq_filt=filter_freq,
                    parallel=forward_parallel,
                    n_proc=n_proc,
                    geomray_split=geomray_data_split)

else:

    Mpost = Mposterior(time_data=time_data,
                       phases2extract=phases2extract,
                       cov_time_data=time_data_cov,
                       pol_data=pol_data,
                        gamma_pol_data=pol_data_gamma,
                        invert_arrival_time=invert_arrival_time,
                        invert_polarity=invert_polarity,
                        sigma_amp_model=sigma_amp,
                        n_layers=n_layers,
                        param_set_list=param_set_list,
                        param_set_values=param_set_values,
                        param_inv_list=param_inv_list,
                        param_inv_bounds=param_inv_bounds,
                        param_same_list=param_same_list,
                        param_same_link=param_same_link,
                        geomray=geomray_data,
                        run_control=run_control_inv,
                        freq_filt=filter_freq,
                        parallel=forward_parallel)


# %% Sampling

if __name__ == '__main__':

    if sampling == 'metropolis':

        z0, _ = Mpost.Mprior.sample()
        C = np.diag([((b[1]-b[0])/denom_proposal)**2 for b in param_inv_bounds])
        
        def q(z):
            a = ss.multivariate_normal(mean=z, cov=C, allow_singular=True).rvs()
            if isinstance(a, (np.ndarray, list)):
                return a
            else:
                return [a]

        if bool_logpdf:

            logp = Mpost.logpdf
            log_metropolis(logp, z0, q, n_accepted, n_burn, n_max, outdir, verbose)
            
        else:

            p = Mpost.pdf
            metropolis(p, z0, q, n_accepted, n_burn, n_max, outdir, verbose)

    elif sampling == 'adaptative_metropolis':

        z0, _ = Mpost.Mprior.sample()
        C0 = np.diag([((b[1]-b[0])/denom_proposal)**2 for b in param_inv_bounds])

        if bool_logpdf:
            
            logp = Mpost.logpdf
            log_adaptative_metropolis(logp, z0, C0, n_params, sd, epsilon, t0, n_accepted, n_burn, n_max, outdir, verbose)

        else:

            p = Mpost.pdf
            print('Not yet implemented.')

    elif sampling == 'grid_search':

        z = [np.linspace(param_inv_bounds[k][0] + (param_inv_bounds[k][1] - param_inv_bounds[k][0])/100,
                        param_inv_bounds[k][1] - (param_inv_bounds[k][1] - param_inv_bounds[k][0])/100,
                        n_sample).tolist() for k in range(n_params)]
        z = list(iter.product(*z))
        z = [list(c) for c in z]

        if bool_logpdf:
            p = Mpost.logpdf
        else:
            p = Mpost.pdf

        grid_search(p, z, n_proc, outdir)

exit(0)