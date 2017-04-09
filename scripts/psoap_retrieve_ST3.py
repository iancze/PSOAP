#!/usr/bin/env python

import argparse

parser = argparse.ArgumentParser(description="Reconstruct the composite spectra for A and B component.")
args = parser.parse_args()

import numpy as np
import matplotlib.pyplot as plt

from astropy.io import ascii
from scipy.linalg import cho_factor, cho_solve

from psoap import constants as C
from psoap.data import redshift, lredshift, Chunk
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

    q_in = pars["q_in"]
    K_in = pars["K_in"] # km/s
    e_in = pars["e_in"] #
    omega_in = pars["omega_in"] # deg
    P_in = pars["P_in"] # days
    T0_in = pars["T0_in"] # epoch

    q_out = pars["q_out"]
    K_out = pars["K_out"] # km/s
    e_out = pars["e_out"] #
    omega_out = pars["omega_out"] # deg
    P_out = pars["P_out"] # days
    T0_out = pars["T0_out"] # epoch

    gamma = pars["gamma"] # km/s
    amp_f = pars["amp_f"] # flux
    l_f = pars["l_f"] # km/s
    amp_g = pars["amp_g"] # flux
    l_g = pars["l_g"] # km/s
    amp_h = pars["amp_h"] # flux
    l_h = pars["l_h"] # km/s

    dates = chunk.date1D

    orb = orbit.ST3(q_in, K_in, e_in, omega_in, P_in, T0_in, q_out, K_out, e_out, omega_out, P_out, T0_out, gamma, obs_dates=dates)

    # predict velocities for each epoch
    vAs, vBs, vCs = orb.get_component_velocities()

    # shift wavelengths according to these velocities to rest-frame of A component
    wls = chunk.wl
    lwls = chunk.lwl
    lwls_A = lredshift(lwls, -vAs[:,np.newaxis])
    lwls_B = lredshift(lwls, -vBs[:,np.newaxis])
    lwls_C = lredshift(lwls, -vCs[:,np.newaxis])

    chunk.apply_mask()
    lwls_A = lwls_A[chunk.mask]
    lwls_B = lwls_B[chunk.mask]
    lwls_C = lwls_C[chunk.mask]

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

    lwls_C_predict = lwls_A_predict
    wls_C_predict = wls_A_predict

    mu, Sigma = covariance.predict_f_g_h(lwls_A.flatten(), lwls_B.flatten(), lwls_C.flatten(), fl.flatten(), sigma.flatten(), lwls_A_predict, lwls_B_predict, lwls_C_predict, mu_f=0.0, mu_g=0.0, mu_h=0.0, amp_f=amp_f, l_f=l_f, amp_g=amp_g, l_g=l_g, amp_h=amp_h, l_h=l_h)


    mu_f = mu[0:n_pix_predict]
    mu_g = mu[n_pix_predict:2 * n_pix_predict]
    mu_h = mu[2 * n_pix_predict:]


    fig, ax = plt.subplots(nrows=3, sharex=True)

    ax[0].plot(wls_A_predict, mu_f, "b")
    ax[0].set_ylabel(r"$f$")
    ax[1].plot(wls_B_predict, mu_g, "g")
    ax[1].set_ylabel(r"$g$")

    ax[2].plot(wls_C_predict, mu_h, "r")
    ax[2].set_ylabel(r"$h$")

    ax[-1].set_xlabel(r"$\lambda\,[\AA]$")

    plots_dir = "plots_" + C.chunk_fmt.format(order, wl0, wl1)

    fig.savefig(plots_dir + "/reconstructed.png", dpi=300)
    plt.close("all")

    np.save(plots_dir + "/f.npy", np.vstack((wls_A_predict, mu_f)))
    np.save(plots_dir + "/g.npy", np.vstack((wls_B_predict, mu_g)))
    np.save(plots_dir + "/h.npy", np.vstack((wls_C_predict, mu_h)))

    np.save(plots_dir + "/mu.npy", mu)
    np.save(plots_dir + "/Sigma.npy", Sigma)


# A laptop (e.g., mine) doesn't have enough memory to do this in parallel, so only serial for now
for chunk in chunks:
    process_chunk(chunk)
