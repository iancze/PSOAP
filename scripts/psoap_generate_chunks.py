#!/usr/bin/env python

# Using a smart estimate of chunk size, create a chunks.dat file.

import argparse

parser = argparse.ArgumentParser(description="Auto-generate comprehensive chunks.dat file, which can be later edited by hand.")
parser.add_argument("--pixels", type=int, default=80, help="Roughly how many pixels should we keep in each chunk?")
parser.add_argument("--overlap", type=int, default=8, help="How many pixels of overlap to aim for.")
parser.add_argument("--start", type=float, default=3000, help="Starting wavelength.")
parser.add_argument("--end", type=float, default=10000, help="Ending wavelength.")

args = parser.parse_args()


# First, check to see if chunks.dat or masks.dat already exist, if so, print warning and exit.
import os

if os.path.exists("chunks.dat"):
    print("chunks.dat already exists in the current directory. Please delete it before proceeding.")
    print("Exiting.")
    import sys
    sys.exit()

import numpy as np
import matplotlib.pyplot as plt

from astropy.table import Table
from astropy.io import ascii
from scipy.linalg import cho_factor, cho_solve

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

# Load the HDF5 file
# read in the actual dataset
dataset = Spectrum(config["data_file"])
# sort by signal-to-noise
dataset.sort_by_SN()

n_epochs, n_orders, n_pix = dataset.wl.shape

print("Dataset shape", dataset.wl.shape)

# Create the chunks file.

data = []

for order in range(n_orders):
    # Determine the max range of the order
    wl_min = np.min(dataset.wl[:,order])
    wl_max = np.max(dataset.wl[:,order])

    if wl_min < args.start:
        wl_min = args.start
    if wl_max > args.end:
        wl_max = args.end

    # Figure out the stride by dividing n_pix by args.pixels and then modulating to as close to 0 as possible.
    n_chunks_frac = n_pix / args.pixels

    n_chunks = int(np.round(n_chunks_frac))

    print("Order {} ranges from {} to {} AA.".format(order, wl_min, wl_max))

    print("Desired {} pixel chunks would yield {} chunks. Rounding to {} chunks.".format(args.pixels, n_chunks_frac, n_chunks))

    # Assume pixel spacing is linear over this small range
    wl_stride = (wl_max - wl_min)/n_chunks
    wl_overlap = args.overlap/args.pixels * wl_stride
    print("Wavelength stride is {} AA, wl overlap is {} AA.\n".format(wl_stride, wl_overlap))

    # Create a list of strides for this order.
    temp = []
    wl0 = wl_min
    wl1 = wl_min
    while wl1 < wl_max:
        wl1 = wl0 + wl_stride + wl_overlap
        temp.append([order, wl0, wl1])
        wl0 += wl_stride

    data += temp

data = Table(rows=data, names=["order", "wl0", "wl1"], dtype=[np.int, np.float64, np.float64])

ascii.write(data, output="chunks.dat", formats={"order": None, "wl0": "%.1f", "wl1": "%.1f"})


# # Then, go through each epoch and determine the histogram of values within this wavelength range. If there is some major outlier, flag this range of dates and wavelength values into a masks.dat file. Use a small perturbation on the date to make sure we are selecting only this wavelength range.
