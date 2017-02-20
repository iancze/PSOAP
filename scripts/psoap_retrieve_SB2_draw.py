#!/usr/bin/env python

import argparse

parser = argparse.ArgumentParser(description="Reconstruct the composite spectra for A and B component.")
parser.add_argument("--draws", type=int, default=0, help="In addition to plotting the mean GP, plot several draws of the GP to show the scatter in predicitions.")
args = parser.parse_args()

import numpy as np
import matplotlib.pyplot as plt

from astropy.io import ascii

from psoap import constants as C
from psoap import utils

import yaml

try:
    f = open("config.yaml")
    config = yaml.load(f)
    f.close()
except FileNotFoundError as e:
    print("You need to copy a config.yaml file to this directory, and then edit the values to your particular case.")
    raise

# Load the list of chunks
chunks = ascii.read(config["chunk_file"])

def process_chunk(row):
    print("processing", row)
    order, wl0, wl1 = row



    plots_dir = "plots_" + C.chunk_fmt.format(order, wl0, wl1)
    mu = np.load(plots_dir + "/mu.npy")
    Sigma = np.load(plots_dir + "/Sigma.npy")
    n_pix_predict = int(len(mu)/2)

    wls_A_predict, f = np.load(plots_dir + "/f.npy")
    wls_B_predict, g = np.load(plots_dir + "/g.npy")


    mu_f = mu[0:n_pix_predict]
    mu_g = mu[n_pix_predict:]

    # Make some multivariate draws
    n_draws = args.draws
    mu_draw = np.random.multivariate_normal(mu, Sigma, size=n_draws)

    # Reshape the spectra right here
    mu_draw_f = np.empty((n_draws, n_pix_predict))
    mu_draw_g = np.empty((n_draws, n_pix_predict))

    for j in range(n_draws):
        mu_draw_j = mu_draw[j]

        mu_draw_f[j] = mu_draw_j[0:n_pix_predict]
        mu_draw_g[j] = mu_draw_j[n_pix_predict:]

    np.save(plots_dir + "/f_draws.npy", mu_draw_f)
    np.save(plots_dir + "/g_draws.npy", mu_draw_g)

    # Plot the draws
    fig, ax = plt.subplots(nrows=2, sharex=True)

    for j in range(n_draws):

        ax[0].plot(wls_A_predict, mu_draw_f[j], color="0.2", lw=0.5)
        ax[1].plot(wls_B_predict, mu_draw_g[j], color="0.2", lw=0.5)


    ax[0].plot(wls_A_predict, mu_f, "b")
    ax[0].set_ylabel(r"$f$")
    ax[1].plot(wls_B_predict, mu_g, "g")
    ax[1].set_ylabel(r"$g$")

    ax[-1].set_xlabel(r"$\lambda\,[\AA]$")

    fig.savefig(plots_dir + "/reconstructed.png", dpi=300)
    plt.close("all")


# A laptop (e.g., mine) doesn't have enough memory to do this in parallel, so only serial for now
for chunk in chunks:
    process_chunk(chunk)
