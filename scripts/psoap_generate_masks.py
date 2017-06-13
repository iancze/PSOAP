#!/usr/bin/env python

# Using a smart estimate of chunk size, create a chunks.dat file.

import argparse

parser = argparse.ArgumentParser(description="Auto-generate comprehensive masks.dat file, which can be later edited by hand.")
parser.add_argument("--sigma", type=float, default=7, help="Flag chunk and date if it contains a deviant of this level.")

args = parser.parse_args()


# First, check to see if chunks.dat or masks.dat already exist, if so, print warning and exit.
import os

if os.path.exists("masks.dat"):
    print("masks.dat already exists in the current directory. Please delete it before proceeding.")
    print("Exiting.")
    import sys
    sys.exit()

import numpy as np
import matplotlib.pyplot as plt

from astropy.table import Table
from astropy.io import ascii
from scipy.linalg import cho_factor, cho_solve

from scipy.stats import norm

from psoap import constants as C
from psoap.data import redshift, Spectrum
from psoap import covariance
from psoap import orbit
from psoap import utils

import yaml

try:
    f = open("config.yaml")
    config = yaml.load(f)
    f.close()
except FileNotFoundError as e:
    print("You need to copy a config.yaml file to this directory, and then edit the values to your particular case.")
    raise

#
# cumul = norm.cdf(args.sigma)
# print("cumul", cumul)
# spread = (1.0 - cumul)/2
# lower = spread * 100.
# upper = (1.0 - spread) * 100.
#
# print("Percentiles for truncation: {:.3f}% {:.3f}%".format(lower, upper))

# Load the HDF5 file
# read in the actual dataset
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
