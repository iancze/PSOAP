#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import ascii

import psoap.constants as C
from psoap.data import Chunk
from psoap import covariance


import yaml

try:
    f = open("config.yaml")
    config = yaml.load(f)
    f.close()
except FileNotFoundError as e:
    print("You need to copy a config.yaml file to this directory, and then edit the values to your particular case.")
    raise

# read in the chunks.dat file
chunks = ascii.read(config["chunk_file"])
print("Optimizing the calibration for the following chunks of data")
print(chunks)

pars = config["parameters"]

# Go through each chunk and optimize the calibration.
for chunk in chunks:
    order, wl0, wl1 = chunk
    chunkSpec = Chunk.open(order, wl0, wl1)
    print("Optimizing", order, wl0, wl1)

    fl_orig = np.copy(chunkSpec.fl)

    covariance.cycle_calibration_chunk(chunkSpec, pars["amp_f"], pars["l_f"], n_cycles=3, limit_array=5)

    # Make a figure comparing the optimization
    fig, ax = plt.subplots(nrows=3, figsize=(8,6), sharex=True)
    for i in range(chunkSpec.n_epochs):
        # print(np.allclose(fl_orig[i], chunkSpec.fl[i]))
        ax[0].plot(chunkSpec.wl[i], fl_orig[i])
        ax[1].plot(chunkSpec.wl[i], chunkSpec.fl[i])
        ax[2].plot(chunkSpec.wl[i], chunkSpec.fl[i]/fl_orig[i])

    plots_dir = "plots_" + C.chunk_fmt.format(order, wl0, wl1)

    ax[0].set_ylabel("original")
    ax[1].set_ylabel("optimized")
    ax[2].set_ylabel("correction")
    ax[-1].set_xlabel(r"$\lambda [\AA]$")
    fig.savefig(plots_dir + "/optim_cal.png")

    # Save
    chunkSpec.save(order, wl0, wl1)
