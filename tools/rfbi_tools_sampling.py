#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 10:52:15 2023

@author: Emile DENISE
"""

import numpy as np
import numpy.random as rd
import scipy.stats as ss
import multiprocessing as mp
import time
from tools.rfbi_tools import *


def grid_search(p, z, n_proc, outdir, verbose):
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

    z_accepted = [z0 + list(p(z0))]
    z_rejected = []

    while len(z_accepted) <= n_accepted+n_burn and len(z_accepted)+len(z_rejected) <= n_max:

        if verbose and (len(z_accepted) % 50 == 0 or (len(z_rejected) + len(z_accepted)) % 50 == 0):
            print('accepted:%i total:%i accept. rate=%.2f' % (len(z_accepted), len(z_rejected) + len(z_accepted), len(z_accepted)/(len(z_rejected) + len(z_accepted))))

        z_candidate = list(q(z0))

        p_zcand = p(z_candidate)
        p_z0 = z_accepted[-1][-1]

        if p_zcand[-1] >= p_z0:
            z_accepted.append(z_candidate + list(p_zcand))
            z0 = z_candidate
        
        else:
            a = rd.uniform(0., 1.)
            
            if a <= p_zcand[-1] / p_z0:
                z_accepted.append(z_candidate + [p_zcand])
                z0 = z_candidate
                
            else:
                z_rejected.append(z_candidate + [p_zcand])
        
        if (len(z_accepted) + len(z_rejected)) % 100 == 0:
            np.save(outdir + '/accepted_models.npy', z_accepted)
            np.save(outdir + '/rejected_models.npy', z_rejected)

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
            
            if a <= logp_zcand[-1] - logp_z0:
                z_accepted.append(z_candidate + list(logp_zcand))
                z0 = z_candidate

            else:
                z_rejected.append(z_candidate + list(logp_zcand))
        
        if (len(z_accepted) + len(z_rejected)) % 100 == 0:
            np.save(outdir + '/accepted_models.npy', z_accepted)
            np.save(outdir + '/rejected_models.npy', z_rejected)

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


def adaptative_metropolis(p, z0, C0, d, sd, eps, t0, n_accepted, n_burn, n_max, outdir, verbose):

    start_time = time.time()

    z_accepted = [z0 + list(p(z0))]
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

        p_zcand = p(z_candidate)
        p_z0 = z_accepted[-1][-1]

        if p_zcand[-1] >= p_z0:
            z_accepted.append(z_candidate + list(p_zcand))
            z0 = z_candidate

            # update
            t = len(z_accepted)-1
            Xt = np.array(z0)
            Xt_mean, Xt_1_mean = update_mean_AM(t, Xt, Xt_mean)
            if t >= 3:
                Ct = update_cov_AM(t, Ct, Xt, Xt_mean, Xt_1_mean, sd, eps, d)
                save_Ct = np.concatenate((save_Ct, Ct[..., np.newaxis]), axis=2)

        else:
            a = rd.uniform(0., 1.)
            
            if a <= p_zcand[-1] / p_z0:
                z_accepted.append(z_candidate + list(p_zcand))
                z0 = z_candidate

                # update
                t = len(z_accepted)-1
                Xt = np.array(z0)
                Xt_mean, Xt_1_mean = update_mean_AM(t, Xt, Xt_mean)
                if t >= 3:
                    Ct = update_cov_AM(t, Ct, Xt, Xt_mean, Xt_1_mean, sd, eps, d)
                    save_Ct = np.concatenate((save_Ct, Ct[..., np.newaxis]), axis=2)

            else:
                z_rejected.append(z_candidate + list(p_zcand))
        
        if (len(z_accepted) + len(z_rejected)) % 100 == 0:
            np.save(outdir + '/accepted_models.npy', z_accepted)
            np.save(outdir + '/rejected_models.npy', z_rejected)
            np.save(outdir + '/covariance_proposal.npy', save_Ct)
    
    np.save(outdir + '/accepted_models.npy', z_accepted)
    np.save(outdir + '/rejected_models.npy', z_rejected)
    np.save(outdir + '/covariance_proposal.npy', save_Ct)
    
    run_time = time.time() - start_time
    h, m, s = sec2hours(run_time)
    print('%i iterations' %(len(z_rejected) + len(z_accepted)))
    print('Runtime: %ih %im %.1fs' % (h, m, s))


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
        
        if (len(z_accepted) + len(z_rejected)) % 100 == 0:
            np.save(outdir + '/accepted_models.npy', z_accepted)
            np.save(outdir + '/rejected_models.npy', z_rejected)
            np.save(outdir + '/covariance_proposal.npy', save_Ct)
    
    np.save(outdir + '/accepted_models.npy', z_accepted)
    np.save(outdir + '/rejected_models.npy', z_rejected)
    np.save(outdir + '/covariance_proposal.npy', save_Ct)
    
    run_time = time.time() - start_time
    h, m, s = sec2hours(run_time)
    print('%i iterations' %(len(z_rejected) + len(z_accepted)))
    print('Runtime: %ih %im %.1fs' % (h, m, s))
