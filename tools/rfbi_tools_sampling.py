#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 10:52:15 2023

@author: Emile DENISE
"""

# %% Packages

from tools.rfbi_tools import *


# %% Metropolis

def metropolis(p, z0, C, param_inv_list, param_inv_prior, n_accepted, n_max, outdir):
    param_list = np.array(param_inv_list)

    start_time = time.time()

    def q(z):
        a = ss.multivariate_normal(mean=z, cov=C, allow_singular=True).rvs()
        if not isinstance(a, (np.ndarray, list)):
            a = [a]
        idx_strike = np.argwhere(param_list[:, 0] == 'strike').T[0]
        for k in idx_strike:
                if param_inv_prior[k][0] == 'uniform' and param_inv_prior[k][1] == 0 and param_inv_prior[k][2] == 360:
                    a[k] = np.mod(a[k], 360)
                if param_inv_prior[k][0] == 'gaussian' and 4*param_inv_prior[k][2] >=360:
                    a[k] = np.mod(a[k], 360)
        return a

    z_accepted = [z0 + list(p(z0))]
    z_rejected = []

    while len(z_accepted) <= n_accepted and len(z_accepted) + len(z_rejected) <= n_max:

        if (len(z_rejected) + len(z_accepted)) % 100 == 0:
            print('accepted:{0:d} total:{1:d} accept. rate={2:.2f}'.format(len(z_accepted), len(z_rejected) + len(z_accepted), len(z_accepted)/(len(z_rejected) + len(z_accepted))))

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

    runtime = time.time() - start_time
    print("Runtime: {:0=2}h {:0=2}m {:02.0f}s".format(*[int(runtime//3600), int(runtime%3600//60), runtime%3600%60]))
    print('%i iterations' %(len(z_rejected) + len(z_accepted)))


def log_metropolis(logp, z0, C, param_inv_list, param_inv_prior, n_accepted, n_max, outdir):
    param_list = np.array(param_inv_list)

    start_time = time.time()

    def q(z):
        a = ss.multivariate_normal(mean=z, cov=C, allow_singular=True).rvs()
        if not isinstance(a, (np.ndarray, list)):
            a = [a]
        idx_strike = np.argwhere(param_list[:, 0] == 'strike').T[0]
        for k in idx_strike:
                if param_inv_prior[k][0] == 'uniform' and param_inv_prior[k][1] == 0 and param_inv_prior[k][2] == 360:
                    a[k] = np.mod(a[k], 360)
                if param_inv_prior[k][0] == 'gaussian' and 4*param_inv_prior[k][2] >=360:
                    a[k] = np.mod(a[k], 360)
        return a

    z_accepted = [z0 + list(logp(z0))]
    z_rejected = []

    while len(z_accepted) <= n_accepted and len(z_accepted) + len(z_rejected) <= n_max:

        if (len(z_rejected) + len(z_accepted)) % 100 == 0:
            print('accepted:{0:d} total:{1:d} accept. rate={2:.2f}'.format(len(z_accepted), len(z_rejected) + len(z_accepted), len(z_accepted)/(len(z_rejected) + len(z_accepted))))

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

    runtime = time.time() - start_time
    print("Runtime: {:0=2}h {:0=2}m {:02.0f}s".format(*[int(runtime//3600), int(runtime%3600//60), runtime%3600%60]))
    print('%i iterations' %(len(z_rejected) + len(z_accepted)))


# %% Adaptative metropolis

def update_mean(t, Xt, Xt_1_mean):
    Xt_mean = (t * Xt_1_mean + Xt) / (t+1)
    return Xt_mean, Xt_1_mean


def update_cov(t, Ct_1, Xt_1, Xt_1_mean, Xt_2_mean, sd, eps, d):
    Ct = ( (t - 1) * Ct_1 + sd * ( t * Xt_2_mean[:, np.newaxis] * Xt_2_mean - (t+1) * Xt_1_mean[:, np.newaxis] * Xt_1_mean + Xt_1[:, np.newaxis] * Xt_1  + eps * np.eye(d) ) ) / t
    return Ct


def adaptative_metropolis(p, z0, C0, param_inv_list, param_inv_prior, d, sd, eps, t0, n_accepted, n_max, outdir):
    param_list = np.array(param_inv_list)

    start_time = time.time()

    z_accepted = [z0 + list(p(z0))]
    z_rejected = []
    Xt_mean = np.array(z0)
    Ct = C0.copy()
    save_Ct = Ct.copy()[:, :, np.newaxis]

    while len(z_accepted) <= n_accepted and len(z_accepted) + len(z_rejected) <= n_max:

        if (len(z_rejected) + len(z_accepted)) % 100 == 0:
            print('accepted:{0:d} total:{1:d} accept. rate={2:.2f}'.format(len(z_accepted), len(z_rejected) + len(z_accepted), len(z_accepted)/(len(z_rejected) + len(z_accepted))))

        t = len(z_accepted) - 1

        if t <= t0:
            def q(z):
                a =  ss.multivariate_normal(mean=z, cov=C0, allow_singular=True).rvs()
                if not isinstance(a, (np.ndarray, list)):
                    a = [a]
                idx_strike = np.argwhere(param_list[:, 0] == 'strike').T[0]
                for k in idx_strike:
                    if param_inv_prior[k][0] == 'uniform' and param_inv_prior[k][1] == 0 and param_inv_prior[k][2] == 360:
                        a[k] = np.mod(a[k], 360)
                    if param_inv_prior[k][0] == 'gaussian' and 4*param_inv_prior[k][2] >=360:
                        a[k] = np.mod(a[k], 360)
                return a

        else: #t>t0
            def q(z):
                a =  ss.multivariate_normal(mean=z, cov=Ct, allow_singular=True).rvs()
                if not isinstance(a, (np.ndarray, list)):
                    a = [a]
                idx_strike = np.argwhere(param_list[:, 0] == 'strike').T[0]
                for k in idx_strike:
                    if param_inv_prior[k][0] == 'uniform' and param_inv_prior[k][1] == 0 and param_inv_prior[k][2] == 360:
                        a[k] = np.mod(a[k], 360)
                    if param_inv_prior[k][0] == 'gaussian' and 4*param_inv_prior[k][2] >=360:
                        a[k] = np.mod(a[k], 360)
                return a

        z_candidate = list(q(z0))

        p_zcand = p(z_candidate)
        p_z0 = z_accepted[-1][-1]

        if p_zcand[-1] >= p_z0:
            z_accepted.append(z_candidate + list(p_zcand))
            z0 = z_candidate

            # update
            t = len(z_accepted)-1
            Xt = np.array(z0)
            Xt_mean, Xt_1_mean = update_mean(t, Xt, Xt_mean)
            if t >= 3:
                Ct = update_cov(t, Ct, Xt, Xt_mean, Xt_1_mean, sd, eps, d)
                save_Ct = np.concatenate((save_Ct, Ct[..., np.newaxis]), axis=2)

        else:
            a = rd.uniform(0., 1.)
            
            if a <= p_zcand[-1] / p_z0:
                z_accepted.append(z_candidate + list(p_zcand))
                z0 = z_candidate

                # update
                t = len(z_accepted)-1
                Xt = np.array(z0)
                Xt_mean, Xt_1_mean = update_mean(t, Xt, Xt_mean)
                if t >= 3:
                    Ct = update_cov(t, Ct, Xt, Xt_mean, Xt_1_mean, sd, eps, d)
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
    
    runtime = time.time() - start_time
    print("Runtime: {:0=2}h {:0=2}m {:02.0f}s".format(*[int(runtime//3600), int(runtime%3600//60), runtime%3600%60]))
    print('%i iterations' %(len(z_rejected) + len(z_accepted)))


def log_adaptative_metropolis(logp, z0, C0, param_inv_list, param_inv_prior, d, sd, eps, t0, n_accepted, n_max, outdir):
    param_list = np.array(param_inv_list)

    start_time = time.time()

    z_accepted = [z0 + list(logp(z0))]
    z_rejected = []
    Xt_mean = np.array(z0)
    Ct = C0.copy()
    save_Ct = Ct.copy()[:, :, np.newaxis]

    while len(z_accepted) <= n_accepted and len(z_accepted) + len(z_rejected) <= n_max:

        if (len(z_rejected) + len(z_accepted)) % 100 == 0:
            print('accepted:{0:d} total:{1:d} accept. rate={2:.2f}'.format(len(z_accepted), len(z_rejected) + len(z_accepted), len(z_accepted)/(len(z_rejected) + len(z_accepted))))

        t = len(z_accepted) - 1

        if t <= t0:
            def q(z):
                a =  ss.multivariate_normal(mean=z, cov=C0, allow_singular=True).rvs()
                if not isinstance(a, (np.ndarray, list)):
                    a = [a]
                idx_strike = np.argwhere(param_list[:, 0] == 'strike').T[0]
                for k in idx_strike:
                    if param_inv_prior[k][0] == 'uniform' and param_inv_prior[k][1] == 0 and param_inv_prior[k][2] == 360:
                        a[k] = np.mod(a[k], 360)
                    if param_inv_prior[k][0] == 'gaussian' and 4*param_inv_prior[k][2] >=360:
                        a[k] = np.mod(a[k], 360)
                return a

        else: #t>t0
            def q(z):
                a =  ss.multivariate_normal(mean=z, cov=Ct, allow_singular=True).rvs()
                if not isinstance(a, (np.ndarray, list)):
                    a = [a]
                idx_strike = np.argwhere(param_list[:, 0] == 'strike').T[0]
                for k in idx_strike:
                    if param_inv_prior[k][0] == 'uniform' and param_inv_prior[k][1] == 0 and param_inv_prior[k][2] == 360:
                        a[k] = np.mod(a[k], 360)
                    if param_inv_prior[k][0] == 'gaussian' and 4*param_inv_prior[k][2] >=360:
                        a[k] = np.mod(a[k], 360)
                return a

        z_candidate = list(q(z0))

        logp_zcand = logp(z_candidate)
        logp_z0 = z_accepted[-1][-1]

        if logp_zcand[-1] >= logp_z0:
            z_accepted.append(z_candidate + list(logp_zcand))
            z0 = z_candidate

            # update
            t = len(z_accepted)-1
            Xt = np.array(z0)
            Xt_mean, Xt_1_mean = update_mean(t, Xt, Xt_mean)
            if t >= 3:
                Ct = update_cov(t, Ct, Xt, Xt_mean, Xt_1_mean, sd, eps, d)
                save_Ct = np.concatenate((save_Ct, Ct[..., np.newaxis]), axis=2)

        else:
            a = np.log(rd.uniform(0., 1.))
            
            if a <= logp_zcand[-1] - logp_z0:
                z_accepted.append(z_candidate + list(logp_zcand))
                z0 = z_candidate

                # update
                t = len(z_accepted)-1
                Xt = np.array(z0)
                Xt_mean, Xt_1_mean = update_mean(t, Xt, Xt_mean)
                if t >= 3:
                    Ct = update_cov(t, Ct, Xt, Xt_mean, Xt_1_mean, sd, eps, d)
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
    
    runtime = time.time() - start_time
    print("Runtime: {:0=2}h {:0=2}m {:02.0f}s".format(*[int(runtime//3600), int(runtime%3600//60), runtime%3600%60]))
    print('%i iterations' %(len(z_rejected) + len(z_accepted)))
