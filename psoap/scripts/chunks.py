import os
import numpy as np
import matplotlib.pyplot as plt
import yaml
import argparse
import multiprocessing as mp

from astropy.table import Table
from astropy.io import ascii

from .. import constants as C
from .. import covariance
from .. import orbit
from .. import utils
from ..data import redshift, Spectrum, Chunk

try:
    f = open("config.yaml")
    config = yaml.load(f)
    f.close()
except FileNotFoundError as e:
    print("You need to copy a config.yaml file to this directory (try psoap-initialize), and then edit the values to suit your needs.")
    raise

# Load the HDF5 file
# read in the actual dataset
dataset = Spectrum(config["data_file"])

# sort by signal-to-noise
dataset.sort_by_SN(config.get("snr_order", C.snr_default))

n_epochs, n_orders, n_pix = dataset.wl.shape

print("Dataset shape", dataset.wl.shape)

# limit?
limit = config["epoch_limit"]
if limit > dataset.n_epochs:
    limit = dataset.n_epochs


# Using a smart estimate of chunk size, create a chunks.dat file.

def generate_main():

    parser = argparse.ArgumentParser(description="Auto-generate a chunks.dat file, which can be later edited by hand.")
    parser.add_argument("--pixels", type=int, default=80, help="Roughly how many pixels per epoch should we have in each chunk?")
    parser.add_argument("--overlap", type=int, default=8, help="How many pixels of overlap to aim for between adjacent chunks.")
    parser.add_argument("--start", type=float, default=3000, help="Starting wavelength (AA).")
    parser.add_argument("--end", type=float, default=10000, help="Ending wavelength (AA).")

    args = parser.parse_args()

    # First, check to see if chunks.dat already exists, if so, print warning and exit.

    if os.path.exists("chunks.dat"):
        print("chunks.dat already exists in the current directory. Please delete it before proceeding, since this script will autogenerate a new chunks.dat file.")
        print("Exiting.")
        import sys
        sys.exit()


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

def process_chunk(chunk):
    order, wl0, wl1 = chunk

    print("Processing order {}, wl0: {:.1f}, wl1: {:.1f} and limiting to {:} highest S/N epochs.".format(order, wl0, wl1, limit))

    # higest S/N epoch
    wl = dataset.wl[0, order, :]

    ind = (wl > wl0) & (wl < wl1)

    # limit these to the epochs we want, for computational purposes
    wl = dataset.wl[:limit, order, ind]
    fl = dataset.fl[:limit, order, ind]
    sigma = dataset.sigma[:limit, order, ind]
    date = dataset.date[:limit, order, ind]
    date1D = dataset.date1D[:limit]

    # Stuff this into a chunk object, and save it
    chunkSpec = Chunk(wl, fl, sigma, date)
    chunkSpec.save(order, wl0, wl1)


def plot_chunk(chunk):
    order, wl0, wl1 = chunk

    # Load the chunk from disk
    chunkSpec = Chunk.open(order, wl0, wl1)

    fig, ax = plt.subplots(nrows=1, sharex=True)

    wl = chunkSpec.wl
    fl = chunkSpec.fl

    # Plot in reverse order so that highest S/N spectra are on top
    for i in range(chunkSpec.n_epochs)[::-1]:
        ax.plot(wl[i], fl[i])

    ax.set_xlabel(r"$\lambda\quad[\AA]$")
    fig.savefig(C.chunk_fmt.format(order, wl0, wl1) + ".png", dpi=300)

    # Now make a separate directory with plots labeled by their date so we can mask problem regions
    plots_dir = "plots_" + C.chunk_fmt.format(order, wl0, wl1)
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    # plot these relative to the highest S/N flux, so we know what looks suspicious, and what to mask.
    for i in range(chunkSpec.n_epochs):
        fig, ax = plt.subplots(nrows=1)
        ax.plot(wl[0], fl[0], color="0.5")
        ax.plot(wl[i], fl[i])
        ax.set_xlabel(r"$\lambda\quad[\AA]$")
        fig.savefig(plots_dir + "/{:.1f}.png".format(chunkSpec.date1D[i]))
        plt.close('all')


def process_main():

    parser = argparse.ArgumentParser(description="Use the demarcated chunks in the chunks.dat to segment the dataset into new HDF5 chunks.")
    parser.add_argument("--plot", action="store_true", help="Make plots of the partitioned chunks.")
    args = parser.parse_args()

    # read in the chunks.dat file
    chunks = ascii.read(config["chunk_file"])
    print("Processing the following chunks of data")
    print(chunks)

    pool = mp.Pool(mp.cpu_count())
    pool.map(process_chunk, chunks)

    if args.plot:
        pool.map(plot_chunk, chunks)
