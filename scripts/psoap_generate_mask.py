#!/usr/bin/env python


from astropy.io import ascii
import matplotlib.pyplot as plt
import numpy as np

import psoap.constants as C
from psoap.data import Spectrum, Chunk

import yaml

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
for chunk in chunks:
    order, wl0, wl1 = chunk
    chunkSpec = Chunk.open(order, wl0, wl1)

    wl = chunkSpec.wl
    fl = chunkSpec.fl
    date = chunkSpec.date
    print("date", date)
    date1D = chunkSpec.date1D

    # start with a full mask (all points included)
    mask = np.ones_like(chunkSpec.wl, dtype="bool")
    print("sum mask before", np.sum(mask))

    for region in masks:
        m0, m1, t0, t1 = region
        # First, create a mask which will isolate the bad region. Do things this way because we
        # can uniquely identify the bad regions
        submask = (wl > m0) & (wl < m1) & (date > t0) & (date < t1)

        # Now update the total mask as the previous mask AND (NOT submask)
        mask = mask & ~submask

    print("sum mask after", np.sum(mask))
    print()
    chunkSpec.mask = mask


    chunkSpec.save(order, wl0, wl1)


    plots_dir = "plots_" + C.chunk_fmt.format(order, wl0, wl1)
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
