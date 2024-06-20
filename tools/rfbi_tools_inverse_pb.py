#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 10:52:15 2023

@author: Emile DENISE
"""

# %% Packages

from tools.rfbi_tools import *
from tools.rfbi_tools_forward_pb import *


# %% Classes

class Mprior:

    def __init__(self, n_layers, param_set_list, param_set_values, param_inv_list, param_inv_prior, param_same_list, param_same_link):
        self.n_layers = n_layers
        
        self.param_set_list = param_set_list
        self.param_set_values = param_set_values
        self.n_set = len(self.param_set_list)

        self.param_inv_list = param_inv_list
        self.param_inv_prior = param_inv_prior
        self.n_inv = len(self.param_inv_list)

        self.param_same_list = param_same_list
        self.param_same_link = param_same_link
        self.n_same = len(self.param_same_list)

        self.rdvar = []

        for b in self.param_inv_prior:
            
            if b[0] == 'uniform':
                self.rdvar.append(ss.uniform(loc=b[1], scale=b[2]-b[1]))
            
            elif b[0] == 'gaussian':
                self.rdvar.append(ss.norm(loc=b[1], scale=b[2]))

    def sample(self):

        mprior_sample = [self.rdvar[k].rvs() for k in range(self.n_inv)]

        struct_sample = gen_struct_from_invsetsame(self.n_layers,
                                                   self.param_set_list, self.param_set_values,
                                                   self.param_inv_list, mprior_sample,
                                                   self.param_same_list, self.param_same_link)

        return mprior_sample, struct_sample
    
    def pdf(self, m):
        M = [self.rdvar[k].pdf(m[k]) for k in range(self.n_inv)]
        return np.nanprod(M)

    def logpdf(self, m):
        logM = [self.rdvar[k].logpdf(m[k]) for k in range(self.n_inv)]
        return np.nansum(logM)


class Ltime:

    def __init__(self, time_data, cov_data):
        diag = np.array([cov_data.data[k, k] for k in range(cov_data.shape[0])])
        self.mask_missing_values = np.isfinite(diag)
        self.cov_data = np.diag(diag[self.mask_missing_values])
        self.time_data = time_data[self.mask_missing_values]

    def pdf(self, time_model):
        dt = time_model[self.mask_missing_values] - self.time_data
        n = np.size(dt) - np.sum(dt.mask)
        if np.all(dt.mask):
            return 1
        else:
            return np.exp(-.5 * np.ma.dot(dt, np.ma.dot(la.inv(self.cov_data), dt)) / n)

    def logpdf(self, time_model):
        dt = time_model[self.mask_missing_values] - self.time_data
        n = np.size(dt) - np.sum(dt.mask)
        if np.all(dt.mask):
            return 0
        else:
            return -.5 * np.ma.dot(dt, np.ma.dot(la.inv(self.cov_data), dt)) / n


class Lpolarity:

    def __init__(self, pol_data, gamma_data):
        self.pol_data = pol_data
        self.gamma_data = gamma_data

    def pdf(self, amp_model, sig_model):
        phi = self.gamma_data + (1 - 2 * self.gamma_data) * ss.norm.cdf(amp_model / sig_model)
        L = phi ** ((1 + self.pol_data) / 2) * (1 - phi) ** ((1 - self.pol_data) / 2)
        n = np.size(L) - np.sum(L.mask)
        if np.all(L.mask):
            return 1
        else:
            return np.nanprod(L) ** (1/n)

    def logpdf(self, amp_model, sig_model):
        phi = self.gamma_data + (1 - 2 * self.gamma_data) * ss.norm.cdf((amp_model) / sig_model)
        L = ((1 + self.pol_data) / 2) * np.log(phi) + ((1 - self.pol_data) / 2) * np.log(1 - phi)
        n = np.size(L) - np.sum(L.mask)
        if np.all(L.mask):
            return 0
        else:
            return np.nansum(L) / n


class Mposterior:

    def __init__(self,
                 phases2extract,
                 invert_tau_ratio,
                 time_data, cov_time_data, pol_data, gamma_pol_data,
                 sigma_amp_model,
                 invert_polarity, invert_arrival_time, weight_polarity, weight_pol2time,
                 n_layers, param_set_list, param_set_values, param_inv_list, param_inv_prior, param_same_list, param_same_link,
                 geomray, run_control, type_filt, freq_filt, parallel, n_proc=1, geomray_split=None):

        self.phases2extract = phases2extract
        self.invert_tau_ratio = invert_tau_ratio
        
        self.time_data = time_data
        self.cov_time_data = cov_time_data
        self.pol_data = pol_data
        self.gamma_pol_data = gamma_pol_data

        self.invert_polarity = invert_polarity
        self.invert_arrival_time = invert_arrival_time
        self.weight_polarity = weight_polarity
        self.weight_pol2time = weight_pol2time

        self.sigma_amp = sigma_amp_model

        self.n_layers = n_layers
        
        self.param_set_list = param_set_list
        self.param_set_values = param_set_values
        self.n_set = len(self.param_set_list)

        self.param_inv_list = param_inv_list
        self.param_inv_prior = param_inv_prior
        self.n_inv = len(self.param_inv_list)

        self.param_same_list = param_same_list
        self.param_same_link = param_same_link
        self.n_same = len(self.param_same_list)
        
        self.geomray = geomray
        self.run_control = run_control
        self.type_filt = type_filt
        self.freq_filt = freq_filt

        self.parallel = parallel
        self.n_proc = n_proc
        self.geomray_split = geomray_split

        ################

        self.Mprior = Mprior(self.n_layers,
                             self.param_set_list, self.param_set_values,
                             self.param_inv_list, self.param_inv_prior,
                             self.param_same_list, self.param_same_link)
        self.Ltime = Ltime(self.time_data, self.cov_time_data)
        self.Lpolarity = Lpolarity(self.pol_data, self.gamma_pol_data)
    
    def pdf(self, m):
        struct = gen_struct_from_invsetsame(self.n_layers,
                                            self.param_set_list, self.param_set_values, self.param_inv_list, m, self.param_same_list, self.param_same_link)
        
        if self.parallel:
            time_model, amp_model, sig_amp_model = g_par(struct, self.geomray_split, self.run_control, self.type_filt, self.freq_filt, self.sigma_amp, self.phases2extract, self.invert_tau_ratio, self.n_proc)
        else:
            time_model, amp_model, sig_amp_model = g(struct, self.geomray, self.run_control, self.type_filt, self.freq_filt, self.sigma_amp, self.phases2extract, self.invert_tau_ratio)
        
        time_model = time_model.reshape(time_model.size)

        success = np.sum(time_model[np.isfinite(self.time_data)].mask)
        tot = np.size(time_model[np.isfinite(self.time_data)])
        if success >= .05*tot:
            print('model error: %i/%i lacking (%.1f %%)' % (success, tot, success/tot*100))

        w = self.weight_pol2time / (1 - self.weight_pol2time)
        
        a = self.Mprior.pdf(m)
        b = self.Lpolarity.pdf(amp_model, sig_amp_model) ** (self.invert_polarity * self.weight_polarity * w)
        c = self.Ltime.pdf(time_model) ** self.invert_arrival_time

        return a, b, c, a * b * c

    def logpdf(self, m):

        if np.isfinite(self.Mprior.logpdf(m)):

            struct = gen_struct_from_invsetsame(self.n_layers,
                                                self.param_set_list, self.param_set_values,
                                                self.param_inv_list, m,
                                                self.param_same_list, self.param_same_link)
            
            if self.parallel:
                time_model, amp_model, sig_amp_model = g_par(struct, self.geomray_split, self.run_control, self.type_filt, self.freq_filt, self.sigma_amp, self.phases2extract, self.invert_tau_ratio, self.n_proc)
            else:
                time_model, amp_model, sig_amp_model = g(struct, self.geomray, self.run_control, self.type_filt, self.freq_filt, self.sigma_amp, self.phases2extract, self.invert_tau_ratio)

            time_model = time_model.reshape(time_model.size)

            success = np.sum(time_model[np.isfinite(self.time_data)].mask)
            tot = np.size(time_model[np.isfinite(self.time_data)])
            if success >= .05*tot:
                print('model error: %i/%i lacking (%.1f %%)' % (success, tot, success/tot*100))

            w = self.weight_pol2time / (1 - self.weight_pol2time)

            a = self.Mprior.logpdf(m)
            b = self.invert_polarity * self.weight_polarity * w * self.Lpolarity.logpdf(amp_model, sig_amp_model)
            c = self.invert_arrival_time * self.Ltime.logpdf(time_model)
    
            return a, b, c, a + b + c
        
        else:

            return -np.inf, np.nan, np.nan, -np.inf
