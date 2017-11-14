import os
import sys
import argparse

import numpy as np
import matplotlib.pyplot as plt

from astropy.table import Table
from astropy.io import ascii

from .. import constants as C
from ..data import redshift, Spectrum, Chunk
from .. import covariance
from .. import orbit
from .. import utils

import yaml
import multiprocessing as mp

def generate_main():
    parser = argparse.ArgumentParser(description="Auto-generate a comprehensive masks.dat file, which can later be edited by hand if necessary.")
    parser.add_argument("--sigma", type=float, default=7, help="Flag a single epoch of a chunk if it contains a deviant above this level.")

    args = parser.parse_args()

    # First, check to see if masks.dat already exists, if so, print warning and exit.
    if os.path.exists("masks.dat"):
        print("masks.dat already exists in the current directory. This script autogenerates a new masks.dat file, so please either rename or delete it before proceeding.")
        print("Exiting.")
        sys.exit()

    try:
        f = open("config.yaml")
        config = yaml.load(f)
        f.close()
    except FileNotFoundError as e:
        print("You need to copy a config.yaml file to this directory (try psoap-initialize), and then edit the values to your particular case.")
        raise

    # read in the dataset
    dataset = Spectrum(config["data_file"])
    # sort by signal-to-noise
    dataset.sort_by_SN(config.get("snr_order", C.snr_default))

    n_epochs, n_orders, n_pix = dataset.wl.shape

    print("Dataset shape", dataset.wl.shape)

    # Load the chunks file
    chunks = ascii.read(config["chunk_file"])

    data = []

    for chunk in chunks:
        order, wl0, wl1 = chunk

        # Get the indices from the highest signal-to-noise order.
        wl = dataset.wl[0, order, :]
        ind = (wl > wl0) & (wl < wl1)

        # Estimate the per-pixel STD across all epochs.
        fl = dataset.fl[:, order, ind]

        mean = np.mean(fl)
        std = np.std(fl)

        # Figure out if the epoch exceeds this anywhere
        flag = ((fl - mean) > (args.sigma * std))
        epoch_flag = np.any(flag, axis=1)
        print("{} epochs flagged for order {} wl0: {:.1f} wl1: {:.1f}.".format(np.sum(epoch_flag), order, wl0, wl1))

        # Add the masked indices to a masks.dat file.
        # Since date is a 3D array, just take the zeroth pixel of the order.
        flagged_dates = dataset.date[:, order, 0][epoch_flag]

        for date in flagged_dates:
            # Add a small buffer of 1/10th of a day
            t0 = date - 0.1
            t1 = date + 0.1
            data.append([wl0, wl1, t0, t1])


    data = Table(rows=data, names=["wl0", "wl1", "t0", "t1"])

    ascii.write(data, output="masks.dat", formats={"wl0": "%.1f", "wl1": "%.1f", "t0":"%.2f", "t1":"%.2f"})


def process_main():
    parser = argparse.ArgumentParser(description="Apply the masks contained in the masks.dat file to the list of chunks contained in chunks.dat")
    parser.add_argument("--plot", action="store_true", help="Make plots of the applied masks.")
    args = parser.parse_args()

    try:
        f = open("config.yaml")
        config = yaml.load(f)
        f.close()
    except FileNotFoundError as e:
        print("You need to copy a config.yaml file to this directory, and then edit the values to your particular case.")
        raise


    # Read in the masks.dat file and create 2D if statements using wl and date. Save this mask.
    # read in the chunks.dat file
    chunks = ascii.read(config["chunk_file"])
    print("Processing the following chunks of data")
    print(chunks)

    # read in the masks
    masks = ascii.read(config["mask_file"])
    print("Appyling the following masks")
    print(masks)


    # go through each chunk, search all masks to build up a total mask, then re-save that chunk
    def process_chunk(chunk):
        order, wl0, wl1 = chunk
        chunkSpec = Chunk.open(order, wl0, wl1)

        print("Processing order {}, wl0: {:.1f}, wl1: {:.1f}".format(order, wl0, wl1))

        wl = chunkSpec.wl
        fl = chunkSpec.fl
        date = chunkSpec.date
        # print("date", date)
        date1D = chunkSpec.date1D

        # limit?
        limit = config["epoch_limit"]
        if limit > chunkSpec.n_epochs:
            limit = chunkSpec.n_epochs

        # start with a full mask (all points included)
        mask = np.ones_like(chunkSpec.wl, dtype="bool")
        # print("sum mask before", np.sum(mask))

        for region in masks:
            m0, m1, t0, t1 = region
            # First, create a mask which will isolate the bad region. Do things this way because we
            # can uniquely identify the bad regions
            submask = (wl > m0) & (wl < m1) & (date > t0) & (date < t1)

            # Now update the total mask as the previous mask AND (NOT submask)
            mask = mask & ~submask

        # print("sum mask after", np.sum(mask))
        # print()
        chunkSpec.mask = mask

        chunkSpec.save(order, wl0, wl1)

        if args.plot:
            plots_dir = "plots_" + C.chunk_fmt.format(order, wl0, wl1)

            # also redo the plot with every epoch the same
            fig, ax = plt.subplots(nrows=1, sharex=True)

            # Plot in reverse order so that highest S/N spectra are on top
            for i in range(limit)[::-1]:
                ax.plot(wl[i], fl[i])
                ax.plot(wl[i][~mask[i]], fl[i][~mask[i]], color="r")

            ax.set_xlabel(r"$\lambda\quad[\AA]$")
            fig.savefig(C.chunk_fmt.format(order, wl0, wl1) + ".png", dpi=300)

            # Go through and re-plot the chunks with highlighted mask points.
            # plot these relative to the highest S/N flux, so we know what looks suspicious, and what to mask.
            for i in range(chunkSpec.n_epochs):
                fig, ax = plt.subplots(nrows=1)
                ax.plot(wl[0], fl[0], color="0.5")
                ax.plot(wl[i], fl[i], color="b")
                ax.plot(wl[i][~mask[i]], fl[i][~mask[i]], color="r")
                ax.set_xlabel(r"$\lambda\quad[\AA]$")
                fig.savefig(plots_dir + "/{:.1f}.png".format(date1D[i]))
                plt.close('all')

    pool = mp.Pool(mp.cpu_count())
    pool.map(process_chunk, chunks)
