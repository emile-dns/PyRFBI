# pyRFBI
Receiver Functions Bayesian Inversion

Bayesian inversion for teleseismic converted waves arrival times and polarities.

Needed:

conda environment with pyraysum + pandas...

export PATH="$PATH:/absolute/path/to/pyRFBI/"

rfbi_make_wkdir.py [-h] wkdir datadir

rfbi_init_invstruct.py [-h] config n_layers target_parameters

rfbi_init_inversion.py [-h] config sampling

rfbi_invert.py [-h] config

rfbi_plot_inv.py [-h] config
