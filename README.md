# PyRFBI

Complete framework for the Bayesian joint inversion of Receiver Functions arrival times and polarities.

Receiver Functions Bayesian Inversion (RFBI) is a tool written in Python to probabilistically invert teleseismic converted waves arrival times and polarities. It uses the Python package [PyRaysum](https://github.com/paudetseis/PyRaysum) to compute the forward model.

## Documentation

No real documentation for now except for this file, please contact me if you have any question. I'll try to reply as best as I can.

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
```

```
rfbi_init_invstruct.py [-h] config n_layers target_parameters
```

```
rfbi_check_invstruct.py [-h] config
```

```
rfbi_plot_data.py [-h] config
```

```
rfbi_init_inversion.py [-h] config sampling
```

```
rfbi_invert.py [-h] config
```

```
rfbi_plot_inv.py [-h] config
```

## Contact

Feel free to contact me and ask questions at [emile.denise@ens.psl.eu](mailto:emile.denise@ens.psl.eu). I'll try to reply as soon as possible.

## Contributing

All constructive contributions are welcome, e.g. bug reports, discussions or suggestions for new features.