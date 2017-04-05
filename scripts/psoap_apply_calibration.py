#!/usr/bin/env python

import argparse

parser = argparse.ArgumentParser(description="Go through the chunks, and replace the raw files with the accepted calibration files.")
# parser.add_argument("--plot", action="store_true", help="Make plots of the applied masks.")
args = parser.parse_args()

import numpy as np
from astropy.io import ascii

import psoap.constants as C
from psoap.data import Chunk

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

# Go through each chunk and optimize the calibration.
for chunk in chunks:
    order, wl0, wl1 = chunk
    print("Applying calibration", order, wl0, wl1)

    chunk = Chunk.open(order, wl0, wl1)

    # Load the previously corrected flux values.
    plots_dir = "plots_" + C.chunk_fmt.format(order, wl0, wl1)
    fl_cor = np.load(plots_dir + "/fl_cor.npy")

    chunk.fl = fl_cor
    chunk.save(order, wl0, wl1)
