# PyRFBI

Complete framework for the Bayesian joint inversion of Receiver Functions arrival times and polarities.

Receiver Functions Bayesian Inversion (RFBI) is a tool written in Python to probabilistically invert teleseismic converted waves arrival times and polarities. It uses the Python package [PyRaysum](https://github.com/paudetseis/PyRaysum) to compute the forward model.

## Documentation

No real documentation for now except for this file, please contact me if you have any question. I'll try to reply as best as I can.

## Installation and requirements

You will neeed a Python environment with PyRaySum. You can follow the [PyRaySum documentation](https://paudetseis.github.io/PyRaysum/init.html#installation). PyRFBI also uses pandas, pytest and seaborn which has to be installed. I strongly recommend creating a custom conda environment where PyRaySum can be installed along with its dependencies. I define here a a minimal environment for running PyRFBI.

```
conda create -n prs python=3.8 fortran-compiler obspy pandas seaborn pytest -c conda-forge
```
```
conda activate prs
```
```
pip install pyraysum
```

Otherwise, you can also use the pyrfbi_env.yml file in the repository to install the conda environment:

```
conda env create -f pyrfbi_env.yml
```

Then, you can download PyRFBI code from here and and its location to $PATH:

```
export PATH="$PATH:/absolute/path/to/pyRFBI/"
```

## Usage

Arrival times and polarities needs to be organized in 5 files:
- data_time.csv : arrival times
- data_time_sigma.csv : arrival times uncertainties
- data_amp_trans.csv : amplitudes
- data_pol_trans.csv : polarities
- data_pol_trans_gamma.csv : polarities uncertainties

Create working directories, copy the data and create a config file (rfbi.ini):

```
rfbi_make_wkdir.py [-h] wkdir datadir [RFdir]
```

Initiate the earth model (number of layers and target parameters). You will need to change the parameters_inversion.csv file.

```
rfbi_init_invstruct.py [-h] config n_layers target_parameters
```

Check that parameters_inversion.csv is correct.

```
rfbi_check_invstruct.py [-h] config
```

Select Metropolis or adaptative Metropolis algorithm and add parameters to config file. You will need to change the inversion and sampling parameters in the config file. 

```
rfbi_init_inversion.py [-h] config sampling
```

Plot inversion data:

```
rfbi_plot_data.py [-h] config
```

Run the inversion:

```
rfbi_invert.py [-h] config
```

Plot inversion results:

```
rfbi_plot_inv.py [-h] config
```

## Contact

Feel free to contact me and ask questions at [emile.denise@ens.psl.eu](mailto:emile.denise@ens.psl.eu). I'll try to reply as soon as possible.

## Contributing

All constructive contributions are welcome, e.g. bug reports, discussions or suggestions for new features.