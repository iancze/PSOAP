#!/usr/bin/env python

import argparse

parser = argparse.ArgumentParser(description="Go through the chunks.dat file and segment the dataset into chunks.")
parser.add_argument("--plot", action="store_true", help="Make plots of the applied masks.")
args = parser.parse_args()


import os
import psoap.constants as C
from psoap.data import Spectrum, Chunk
from astropy.io import ascii
import matplotlib.pyplot as plt
import multiprocessing as mp

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
print("Processing the following chunks of data")
print(chunks)

# read in the actual dataset
dataset = Spectrum(config["data_file"])
# sort by signal-to-noise
dataset.sort_by_SN()

def process_chunk(chunk):
    order, wl0, wl1 = chunk

    print("Processing order {}, wl0: {:.1f}, wl1: {:.1f}".format(order, wl0, wl1))

    wl = dataset.wl[0, order, :]

    ind = (wl > wl0) & (wl < wl1)

    wl = dataset.wl[:, order, ind]
    fl = dataset.fl[:, order, ind]
    sigma = dataset.sigma[:, order, ind]
    date = dataset.date[:, order, ind]
    date1D = dataset.date1D

    # Stuff this into a chunk object, and save it
    chunkSpec = Chunk(wl, fl, sigma, date)
    chunkSpec.save(order, wl0, wl1)

    if args.plot:
        fig, ax = plt.subplots(nrows=1, sharex=True)

        # Plot in reverse order so that highest S/N spectra are on top
        for i in range(dataset.n_epochs)[::-1]:
            ax.plot(wl[i], fl[i])

        ax.set_xlabel(r"$\lambda\quad[\AA]$")
        fig.savefig(C.chunk_fmt.format(order, wl0, wl1) + ".png", dpi=300)

        # Now make a separate directory with plots labeled by their date so we can mask problem regions
        plots_dir = "plots_" + C.chunk_fmt.format(order, wl0, wl1)
        if not os.path.exists(plots_dir):
            os.makedirs(plots_dir)

        # plot these relative to the highest S/N flux, so we know what looks suspicious, and what to mask.
        for i in range(dataset.n_epochs):
            fig, ax = plt.subplots(nrows=1)
            ax.plot(wl[0], fl[0], color="0.5")
            ax.plot(wl[i], fl[i])
            ax.set_xlabel(r"$\lambda\quad[\AA]$")
            fig.savefig(plots_dir + "/{:.1f}.png".format(date1D[i]))
            plt.close('all')

pool = mp.Pool(mp.cpu_count())
pool.map(process_chunk, chunks)
