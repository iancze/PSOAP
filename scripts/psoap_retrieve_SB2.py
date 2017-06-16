#!/usr/bin/env python

import argparse

parser = argparse.ArgumentParser(description="Reconstruct the composite spectra for A and B component.")
parser.add_argument("--draws", type=int, default=0, help="In addition to plotting the mean GP, plot several draws of the GP to show the scatter in predicitions.")
args = parser.parse_args()

draws = (args.draws > 0)

import numpy as np
import matplotlib.pyplot as plt

from astropy.io import ascii
from scipy.linalg import cho_factor, cho_solve

from psoap import constants as C
from psoap.data import lredshift, redshift, Chunk
from psoap import covariance
from psoap import orbit
from psoap import utils

import multiprocessing as mp

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
    order, wl0, wl1 = row
    print("Processing order {}, wl0: {:.1f}, wl1: {:.1f}".format(order, wl0, wl1))
    chunk = Chunk.open(order, wl0, wl1, limit=config["epoch_limit"])

    n_epochs = chunk.n_epochs
    n_pix = chunk.n_pix


    # Use the parameters specified in the yaml file to create the spectra
    pars = config["parameters"]

    q = pars["q"]
    K = pars["K"] # km/s
    e = pars["e"] #
    omega = pars["omega"] # deg
    P = pars["P"] # days
    T0 = pars["T0"] # epoch
    gamma = pars["gamma"] # km/s
    amp_f = pars["amp_f"] # flux
    l_f = pars["l_f"] # km/s
    amp_g = pars["amp_g"] # flux
    l_g = pars["l_g"] # km/s

    dates = chunk.date1D
    orb = orbit.SB2(q, K, e, omega, P, T0, gamma, obs_dates=dates)

    # predict velocities for each epoch
    vAs, vBs = orb.get_velocities()

    # shift wavelengths according to these velocities to rest-frame of A component
    # Load the data
    wls = chunk.wl
    lwls = chunk.lwl

    wls_A = redshift(wls, -vAs[:,np.newaxis])
    wls_B = redshift(wls, -vBs[:,np.newaxis])

    lwls_A = lredshift(lwls, -vAs[:,np.newaxis])
    lwls_B = lredshift(lwls, -vBs[:,np.newaxis])


    chunk.apply_mask()
    wls_A = wls_A[chunk.mask]
    wls_B = wls_B[chunk.mask]

    lwls_A = lwls_A[chunk.mask]
    lwls_B = lwls_B[chunk.mask]

    # reload this, including the masked data
    fl = chunk.fl
    sigma = chunk.sigma
    dates = chunk.date1D

    # Spectra onto which we want to predict new spectra.

    # These are 2X finely spaced as the data, and span the maximum range of the spectra at 0
    # velocity (barycentric frame).
    n_pix_predict = 2 * n_pix

    # These are the same input wavelegths.
    lwls_A_predict = np.linspace(np.min(lwls_A), np.max(lwls_A), num=n_pix_predict)
    wls_A_predict = np.exp(lwls_A_predict)

    lwls_B_predict = lwls_A_predict
    wls_B_predict = wls_A_predict

    mu, Sigma = covariance.predict_f_g(lwls_A.flatten(), lwls_B.flatten(), fl.flatten(), sigma.flatten(), lwls_A_predict, lwls_B_predict, mu_f=0.0, mu_g=0.0, amp_f=amp_f, l_f=l_f, amp_g=amp_g, l_g=l_g)

    sigma_diag = np.sqrt(np.diag(Sigma))

    mu_f = mu[0:n_pix_predict]
    sigma_f = sigma_diag[0:n_pix_predict]
    mu_g = mu[n_pix_predict:]
    sigma_g = sigma_diag[n_pix_predict:]


    # Make some multivariate draws
    if draws:
        n_draws = args.draws
        mu_draw = np.random.multivariate_normal(mu, Sigma, size=n_draws)

        print(mu_draw.shape)

        # Reshape the spectra right here
        mu_draw_f = np.empty((n_draws, n_pix_predict))
        mu_draw_g = np.empty((n_draws, n_pix_predict))

        for j in range(n_draws):
            mu_draw_j = mu_draw[j]

            mu_draw_f[j] = mu_draw_j[0:n_pix_predict]
            mu_draw_g[j] = mu_draw_j[n_pix_predict:]


    fig, ax = plt.subplots(nrows=2, sharex=True)

    if draws:
        for j in range(n_draws):

            ax[0].plot(wls_A_predict, mu_draw_f[j], color="0.2", lw=0.5)
            ax[1].plot(wls_B_predict, mu_draw_g[j], color="0.2", lw=0.5)


    ax[0].plot(wls_A_predict, mu_f, "b")
    ax[0].set_ylabel(r"$f$")
    ax[1].plot(wls_B_predict, mu_g, "g")
    ax[1].set_ylabel(r"$g$")

    ax[-1].set_xlabel(r"$\lambda\,[\AA]$")

    plots_dir = "plots_" + C.chunk_fmt.format(order, wl0, wl1)

    fig.savefig(plots_dir + "/reconstructed.png", dpi=300)
    plt.close("all")

    np.save(plots_dir + "/f.npy", np.vstack((wls_A_predict, mu_f, sigma_f)))

    np.save(plots_dir + "/g.npy", np.vstack((wls_B_predict, mu_g, sigma_g)))

    np.save(plots_dir + "/mu.npy", mu)
    np.save(plots_dir + "/Sigma.npy", Sigma)

    if draws:
        np.save(plots_dir + "/f_draws.npy", mu_draw_f)
        np.save(plots_dir + "/g_draws.npy", mu_draw_g)


# A laptop (e.g., mine) doesn't have enough memory to do this in parallel, so only serial for now
for chunk in chunks:
    process_chunk(chunk)
