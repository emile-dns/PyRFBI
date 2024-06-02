#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 09:35:53 2023

@author: Emile DENISE
"""

from tools.rfbi_tools import *
from tools.rfbi_tools_sampling import *
from tools.rfbi_tools_inverse_pb import *

# %% Read arguments and check

parser = argparse.ArgumentParser(description='Prepare inversion')
parser.add_argument('config', help='Path to config file.', type=file_path)

args = parser.parse_args()
path_config = args.config
config = configparser.ConfigParser()
config.read(path_config)

wkdir = config['INPUT']['wkdir']
datadir = config['INPUT']['datadir']
outdir = config['INPUT']['outdir']

n_layers = config.getint('STRUCTURE SETUP', 'n_layers')

sampling = config['INVERSION SETUP']['sampling']
bool_logpdf = config.getboolean('INVERSION SETUP', 'log_pdf')
type_filt = config['INVERSION SETUP']['type_filt']
if type_filt == 'bandpass':
    filter_freq = tuple(map(float, config['INVERSION SETUP']['filter_freq'].split(',')))
else:
    filter_freq = config.getfloat('INVERSION SETUP', 'filter_freq')
dt = config.getfloat('INVERSION SETUP', 'dt')
npts = config.getint('INVERSION SETUP', 'npts')
n_params = config.getint('INVERSION SETUP', 'n_params')
invert_polarity = config.getboolean('INVERSION SETUP', 'invert_polarity')
invert_arrival_time = config.getboolean('INVERSION SETUP', 'invert_arrival_time')
weight_polarity = config.getfloat('INVERSION SETUP', 'weight_polarity')
weight_pol2time = config.getfloat('INVERSION SETUP', 'weight_pol2time')
sigma_amp = config.getfloat('INVERSION SETUP', 'sigma_amp')
forward_parallel = config.getboolean('INVERSION SETUP', 'forward_parallel')

n_max = config.getint('INVERSION SETUP', 'n_max')
n_accepted = config.getint('INVERSION SETUP', 'n_accepted')
denom_proposal = config.getfloat('INVERSION SETUP', 'denom_proposal')
if forward_parallel:
    n_proc = config.getint('INVERSION SETUP', 'n_proc')

if sampling == 'adaptative_metropolis':
    sd = config.getfloat('INVERSION SETUP', 'sd')
    epsilon = config.getfloat('INVERSION SETUP', 'epsilon')
    t0 = config.getint('INVERSION SETUP', 't0')


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
phases2extract = np.array(gen_phaselist(n_layers, phases2extract))
invert_tau_ratio = np.array(list(filter(None, config['INVERSION SETUP']['invert_tau_ratio'].split(',')))).astype(int)

param_set_list, param_set_values, param_inv_list, param_inv_prior, param_same_list, param_same_link = read_csv_struct(wkdir + '/parameters_inversion.csv')

run_control_inv = prs.Control(verbose=False,
                              rot='RTZ',
                              dt=dt,
                              npts=npts,
                              align=0,
                              mults=2)

_, struct_phase = Mprior(n_layers, param_set_list, param_set_values, param_inv_list, param_inv_prior, param_same_list, param_same_link).sample()

# run with a structure from the prior to obtain the accurate phase list
result = prs.run(struct_phase, geomray_data, run_control_inv)
phase_list = result.descriptors()
run_control_inv.set_phaselist(phase_list, equivalent=True)
result = phase_list = None

# %% Posterior pdf

Mpost = Mposterior(time_data=time_data,
                   phases2extract=phases2extract,
                   invert_tau_ratio=invert_tau_ratio,
                   cov_time_data=time_data_cov,
                   pol_data=pol_data,
                   gamma_pol_data=pol_data_gamma,
                   invert_arrival_time=invert_arrival_time,
                   invert_polarity=invert_polarity,
                   weight_polarity=weight_polarity,
                   weight_pol2time=weight_pol2time,
                   sigma_amp_model=sigma_amp,
                   n_layers=n_layers,
                   param_set_list=param_set_list,
                   param_set_values=param_set_values,
                   param_inv_list=param_inv_list,
                   param_inv_prior=param_inv_prior,
                   param_same_list=param_same_list,
                   param_same_link=param_same_link,
                   geomray=geomray_data,
                   run_control=run_control_inv,
                   type_filt=type_filt,
                   freq_filt=filter_freq,
                   parallel=forward_parallel)

if forward_parallel:

    Mpost.n_proc = n_proc
    Mpost.geomray_split = geomray_data_split


# %% Sampling

if __name__ == '__main__':

    C0 = []
    for b in param_inv_prior:
        if b[0] == 'uniform':
            C0.append(((b[2]-b[1])/denom_proposal)**2)
        elif b[0] == 'gaussian':
            C0.append((6*b[2]/denom_proposal)**2)
    C0 = np.diag(C0)

    if sampling == 'metropolis':

        z0, _ = Mpost.Mprior.sample()
        
        ### Run to determine the weight for the polarity likelihood (if inverted for)

        if invert_polarity and invert_arrival_time and weight_polarity == 1:

            print("\n##########################################################\n"+
                  "Starting preliminary run to determine the polarity weight.\n"+
                  "##########################################################\n\n")
            
            if not os.path.exists(outdir + "/run_weight_pol"):
                os.makedirs(outdir + "/run_weight_pol")

            if bool_logpdf:
                logp = Mpost.logpdf
                log_metropolis(logp, z0, C0, param_inv_list, param_inv_prior, n_accepted//10, n_max, outdir + "/run_weight_pol/")
            else:
                p = Mpost.pdf
                metropolis(p, z0, C0, param_inv_list, param_inv_prior, n_accepted//10, n_max, outdir + "/run_weight_pol/")

            z_accepted = np.load(outdir + "/run_weight_pol/accepted_models.npy")
            sig_Ltime = np.nanstd(z_accepted[:, -2])
            sig_Lpol = np.nanstd(z_accepted[:, -3])
            Mpost.weight_polarity = sig_Ltime / sig_Lpol

            config['INVERSION SETUP']['weight_polarity'] = '%.4f' % (Mpost.weight_polarity)
            update_config(config, wkdir)
        
        elif invert_polarity and invert_arrival_time and weight_polarity == -1: # invert pol ands time but do not weight pol. likelihood
            Mpost.weight_polarity = 1
            Mpost.weight_pol2time = weight_pol2time
        else:
            Mpost.weight_polarity = weight_polarity
            Mpost.weight_pol2time = weight_pol2time
        
        ### Real run
            
        print("\n########################################\n"+
              "Starting sampling of the posterior pdf.\n"+
              "########################################\n\n")

        if bool_logpdf:

            logp = Mpost.logpdf
            log_metropolis(logp, z0, C0, param_inv_list, param_inv_prior, n_accepted, n_max, outdir)
            
        else:

            p = Mpost.pdf
            metropolis(p, z0, C0, param_inv_list, param_inv_prior, n_accepted, n_max, outdir)

    elif sampling == 'adaptative_metropolis':

        z0, _ = Mpost.Mprior.sample()

        if invert_polarity and invert_arrival_time and weight_polarity == 1: # run to determine weight for the polarity likelihood

            print("\n##########################################################\n"+
                  "Starting preliminary run to determine the polarity weight.\n"+
                  "##########################################################\n\n")
            
            if not os.path.exists(outdir + "/run_weight_pol"):
                os.makedirs(outdir + "/run_weight_pol")

            if bool_logpdf:
                logp = Mpost.logpdf
                log_adaptative_metropolis(logp, z0, C0, param_inv_list, param_inv_prior, n_params, sd, epsilon, t0, n_accepted//10, n_max, outdir + "/run_weight_pol/")
            else:
                p = Mpost.pdf
                adaptative_metropolis(p, z0, C0, param_inv_list, param_inv_prior, n_params, sd, epsilon, t0, n_accepted//10, n_max, outdir + "/run_weight_pol/")

            z_accepted = np.load(outdir + "/run_weight_pol/accepted_models.npy")
            sig_Ltime = np.nanstd(z_accepted[:, -2])
            sig_Lpol = np.nanstd(z_accepted[:, -3])
            Mpost.weight_polarity = sig_Ltime / sig_Lpol
            print(sig_Ltime, sig_Lpol, sig_Ltime / sig_Lpol)

            config['INVERSION SETUP']['weight_polarity'] = '%.4f' % (Mpost.weight_polarity)
            update_config(config, wkdir)
        
        elif invert_polarity and invert_arrival_time and weight_polarity == -1: # invert pol ands time but do not weight pol. likelihood
            Mpost.weight_polarity = 1
            Mpost.weight_pol2time = weight_pol2time

        else:
            Mpost.weight_polarity = weight_polarity
            Mpost.weight_pol2time = weight_pol2time
        
        ### Real run
        
        print("\n########################################\n"+
              "Starting sampling of the posterior pdf.\n"+
              "########################################\n\n")

        if bool_logpdf:
            
            logp = Mpost.logpdf
            log_adaptative_metropolis(logp, z0, C0, param_inv_list, param_inv_prior, n_params, sd, epsilon, t0, n_accepted, n_max, outdir)

        else:

            p = Mpost.pdf
            adaptative_metropolis(p, z0, C0, param_inv_list, param_inv_prior, n_params, sd, epsilon, t0, n_accepted, n_max, outdir)

exit(0)
