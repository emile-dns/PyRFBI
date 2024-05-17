# pyRFBI

Complete framework for the Bayesian joint inversion of Receiver Functions arrival times and polarities.

Receiver Functions Bayesian Inversion (RFBI) is a tool to invert teleseismic converted waves arrival times and polarities. It uses the package pyraysum

## Documentation

No real documenation for now, please contact me if you have any question.

## Installation and requirements

conda environment with pyraysum + pandas + seaborn + ... (update list)

```
conda activate prs
export PATH="$PATH:/absolute/path/to/pyRFBI/"
xxxxx tests
```

## Usage

```
rfbi_make_wkdir.py [-h] wkdir datadir RFdir

rfbi_init_invstruct.py [-h] config n_layers target_parameters

rfbi_check_invstruct.py [-h] config

rfbi_plot_data.py [-h] config

rfbi_init_inversion.py [-h] config sampling

rfbi_invert.py [-h] config

rfbi_plot_inv.py [-h] config
```

## Contact

Feel free to ask questions and contact me at [emile.denise@ens.psl.eu](mailto:emile.denise@ens.psl.eu)