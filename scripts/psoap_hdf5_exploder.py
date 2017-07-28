#!/usr/bin/env python

import argparse

parser = argparse.ArgumentParser(description="Make summary plots for a full HDF5 dataset.")
parser.add_argument("--orders", help="Which orders to plot. By default, all orders are plotted. Can add more than one order in a spaced list, e.g., --orders 22 23 24 but not --orders=22,23,24", default="all", nargs="*")
parser.add_argument("--SNR", action="store_true", help="Plot spectra in order of highest SNR first, instead of by date. Default is by date.")
parser.add_argument("--topo", action="store_true", help="Plot spectra in topocentric frame instead of barycentric frame. Default is barycentric frame.")
parser.add_argument("--spacing", type=float, default=1, help="Multiply the default vertical spacing between epochs by this value")

args = parser.parse_args()

import yaml

try:
    f = open("config.yaml")
    config = yaml.load(f)
    f.close()
except FileNotFoundError as e:
    print("You need to copy a config.yaml file to this directory, and then edit the values to your particular case.")
    raise

import os
import numpy as np
import matplotlib.pyplot as plt
from psoap.data import Spectrum, redshift
import psoap.constants as C


# Load the spectrum
spectrum = Spectrum(config["data_file"])

# Create a plot directory structure
# Depending on command line arguments, options are
#
# plots/sort_date/bary
# plots/sort_date/topo
#
# plots/sort_SNR/bary
# plots/sort_SNR/topo

plot_dir_base = "".join(["plots/", "sort_SNR/" if args.SNR else "sort_date/", "topo/" if args.topo else "bary/"])
print("Saving plots in", plot_dir_base)

# Make plots directory if it doesn't exist
if not os.path.exists(plot_dir_base):
    os.makedirs(plot_dir_base)

# Sort the spectra by SNR if we want
if args.SNR:
    spectrum.sort_by_SN(config.get("snr_order", C.snr_default))

# Read the important quantities
wls = spectrum.wl
fls = spectrum.fl
sigmas = spectrum.sigma
dates = spectrum.date1D
n_epochs = len(wls)

if args.orders == "all":
    orders = np.arange(wls.shape[1])
else:
    orders = np.array(args.orders, dtype=np.int)

# Conversion to topocentric frame
# It is assumed that the spectra are stored in the HDF5 file are already in the barycentric frame.
# To convert back to topocentric frame, we will need to shift them back by the barycentric correction.
if args.topo:
    BCVs = spectrum.BCV
    wls = redshift(wls, -BCVs)


# Set up some plot constants
dy_per_epoch = 0.5 * args.spacing # in

for order in orders:
    # Create a specific director for the order, if it doesn't exist.
    plots_dir = plot_dir_base + "{:0>3d}/".format(order)
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    # Make a plot for all epochs in one diagram. Either in order of decreasing epoch or decreasing SNR.
    # Label each spectrum with date and SNR, in order of which one came first.

    # calculate max and min wl for this order
    wl_min = np.min(wls[:,order])
    wl_max = np.max(wls[:,order])
    fl_min = np.min(fls[:,order])
    fl_max = np.max(fls[:,order])
    wl_range = (wl_max - wl_min)
    fl_range = (fl_max - fl_min)


    x0 = wl_min - 0.09 * wl_range
    x0_annotate = wl_min - 0.045 * wl_range
    x1 = wl_max + 0.09 * wl_range
    x1_annotate = wl_max + 0.045 * wl_range

    y0 = fl_min - 0.09 * fl_range
    y1 = fl_max + 0.09 * fl_range

    fig, ax = plt.subplots(figsize=(10.0, n_epochs * dy_per_epoch))

    ax.set_ylim(1.0, 2.0 + 0.4 * n_epochs)
    ax.annotate("JD - 2,400,000", (x0_annotate, 1.4 + 0.4 * n_epochs), color='k', ha="left", va="center", size=8)
    ax.annotate("SNR", (x1_annotate, 1.4 + 0.4 * n_epochs), color='k', ha="center", va="center", size=8)

    for i in range(n_epochs):
        offset = 0.4 * args.spacing
        pedastal = offset * n_epochs
        p = ax.plot(wls[i,order,:], fls[i,order,:] + (pedastal - offset * i))
        color = p[0].get_color() # the color of the line we just plotted
        # now make labels in this color

        date = dates[i] - 2400000 # JD
        SNR = np.median(fls[i,order,:] / sigmas[i,order,:])

        ax.annotate("{:.1f}".format(date), (x0_annotate, 1 + (pedastal - offset * i)), color=color, ha="center", va="center", size=8)
        ax.annotate("{:.0f}".format(SNR), (x1_annotate, 1 + (pedastal - offset * i)), color=color, ha="center", va="center", size=8)

    ax.set_xlim(x0, x1)
    ax.set_xlabel(r"$\lambda\,[\AA]$")
    ax.set_ylabel("flux + constant")
    fig.subplots_adjust(left=0.05, right=0.95, top=0.99, bottom=(0.5 / (n_epochs * dy_per_epoch)))
    fig.savefig(plots_dir + "all_spectra.png", dpi=300)
    plt.close('all')

    # Now, plot each spectrum individually on it's own figure, with lots of space, and compared to the highest SNR spectrum in the background.
    for i in range(n_epochs):
        fig, ax = plt.subplots(figsize=(10,8))
        ax.plot(wls[i,order,:], fls[i,order,:], color="k")
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        ax.annotate("JD - 2,400,000\n{:.1f}".format(dates[i] - 2400000), (x0_annotate, 1.0), color='k', ha="center", va="center", size=8)
        ax.annotate("SNR\n{:.0f}".format(SNR), (x1_annotate, 1.0), color='k', ha="center", va="center", size=8)

        ax.set_xlabel(r"$\lambda\,[\AA]$")
        ax.set_ylabel("flux")

        fig.subplots_adjust(left=0.05, right=0.95, top=0.99, bottom=0.05)

        if args.SNR:
            fig.savefig(plots_dir + "{:0>3}.png".format(i))
        else:
            fig.savefig(plots_dir + "{:.1f}.png".format(dates[i]))

        plt.close('all')


# plot all orders all epochs separately
# plot all epochs together, offset
# all epochs offset
#     - in order of SNR
#     - in order of date
#     - in BC frame
#     - in topo frame.
