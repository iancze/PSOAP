# Processing for setup

All of the scripts necessary to run a typical analysis are located in the `scripts/` directory.

First, create a new directory where you would like to do your analysis. Then, run `psoap_initialze.py` to copy setup files to this directory, like `config.yaml`.

Open your favorite text editor and modify the values in `config.yaml` to suit your particular application.

Then, we will need to generate a `chunks.dat` file and a `masks.dat` file. This is done via

`psoap_generate_chunks.py`
`psoap_generate_mask.py`

The data file is segmented into chunks by running

`psoap_process_chunks.py`

and then the masks are applied by running

`psoap_process_masks.py`

To tweak the calibration (may want to have an orbit in mind, first)

`psoap_process_calibration.py`
`psoap_plot_calibration.py`
`psoap_apply_calibration.py`

To infer parameters,

`psoap_sample_parallel.py`

To reconstruct spectra

`psoap_retrieve_SB2.py`
