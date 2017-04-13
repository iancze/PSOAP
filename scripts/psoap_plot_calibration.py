#!/usr/bin/env python

import argparse

parser = argparse.ArgumentParser(description="Go through the chunks, and plot comparisons relative to the original files.")
parser.add_argument("--chunk_index", type=int, help="Only run the calibration on a specific chunk.")
args = parser.parse_args()


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
for chunk_index,chunk in enumerate(chunks):
    if (args.chunk_index is not None) and (chunk_index != args.chunk_index):
        continue

    order, wl0, wl1 = chunk
    chunk = Chunk.open(order, wl0, wl1)
    plots_dir = "plots_" + C.chunk_fmt.format(order, wl0, wl1)

    # Load the previously corrected flux values.
    fl_cor = np.load(plots_dir + "/fl_cor.npy")

    # Go through and plot the change in each epoch relative to the original (and relative to the highest S/N epoch.)
    wl = chunk.wl
    fl_orig = chunk.fl
    date = chunk.date
    mask = chunk.mask
    # print("date", date)
    date1D = chunk.date1D

    print(wl.shape)
    print(chunk.n_epochs)
    print(fl_cor.shape)

    print("Plotting", order, wl0, wl1)

    # Make a figure comparing the optimization
    fig, ax = plt.subplots(nrows=3, figsize=(8,6), sharex=True)
    for i in range(chunk.n_epochs):
        # print(np.allclose(fl_orig[i], chunkSpec.fl[i]))
        ax[0].plot(chunk.wl[i], fl_orig[i])
        ax[1].plot(chunk.wl[i], fl_cor[i])
        ax[2].plot(chunk.wl[i], fl_cor[i]/fl_orig[i])


    ax[0].set_ylabel("original")
    ax[1].set_ylabel("optimized")
    ax[2].set_ylabel("correction")
    ax[-1].set_xlabel(r"$\lambda [\AA]$")
    fig.savefig(plots_dir + "/optim_cal.png")

    # Go through and re-plot the chunks with highlighted mask points.
    # plot these relative to the highest S/N flux, so we know what looks suspicious, and what to mask.
    for i in range(chunk.n_epochs):
        fig, ax = plt.subplots(nrows=3, sharex=True, figsize=(10,8))
        ax[0].plot(wl[0], fl_orig[0], color="0.5")
        ax[0].plot(wl[i], fl_orig[i], color="b")
        ax[0].plot(wl[i][~mask[i]], fl_cor[i][~mask[i]], color="r")

        ax[1].plot(wl[0], fl_cor[0], color="0.5")
        ax[1].plot(wl[i], fl_cor[i], color="b")
        ax[1].plot(wl[i][~mask[i]], fl_cor[i][~mask[i]], color="r")

        ax[2].plot(wl[i], fl_cor[i]/fl_orig[i], color="0.5")

        ax[2].set_xlabel(r"$\lambda\quad[\AA]$")

        fig.savefig(plots_dir + "/{:.1f}.png".format(date1D[i]))
        plt.close('all')
