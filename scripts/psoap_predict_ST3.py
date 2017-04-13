#!/usr/bin/env python

import argparse

parser = argparse.ArgumentParser(description="Measure statistics across multiple chains.")
parser.add_argument("--draws", type=int, default=0, help="In addition to plotting the mean GP, plot several draws of the GP to show the scatter in predicitions.")
args = parser.parse_args()

draws = (args.draws > 0)

import numpy as np
import matplotlib.pyplot as plt

from scipy.linalg import cho_factor, cho_solve

from astropy.io import ascii

from psoap import constants as C
from psoap.data import lredshift, Chunk
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

# Load the list of chunks
chunks = ascii.read(config["chunk_file"])

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


for row in chunks:
    # For now, only use the first chunk.
    order, wl0, wl1 = row
    chunk = Chunk.open(order, wl0, wl1, limit=config["epoch_limit"])

    # Load the data
    wls = chunk.wl
    lwls = chunk.lwl
    fl = chunk.fl
    sigma = chunk.sigma
    dates = chunk.date1D

    orb = orbit.ST3(q_in, K_in, e_in, omega_in, P_in, T0_in, q_out, K_out, e_out, omega_out, P_out, T0_out, gamma, obs_dates=dates)

    # predict velocities for each epoch
    vAs, vBs, vCs = orb.get_component_velocities()

    # shift wavelengths according to these velocities to rest-frame of A component
    lwls_A = lredshift(lwls, -vAs[:,np.newaxis])
    lwls_B = lredshift(lwls, -vBs[:,np.newaxis])
    lwls_C = lredshift(lwls, -vCs[:,np.newaxis])

    n_epochs, n_pix = lwls_A.shape

    # First predict the component spectra as mean 1 GPs
    mu, Sigma = covariance.predict_f_g_h(lwls_A.flatten(), lwls_B.flatten(), lwls_C.flatten(), fl.flatten(), sigma.flatten(), lwls_A.flatten(), lwls_B.flatten(), lwls_C.flatten(), mu_f=0.0, mu_g=0.0, mu_h=0.0, amp_f=amp_f, l_f=l_f, amp_g=amp_g, l_g=l_g, amp_h=amp_h, l_h=l_h)

    mu_sum, Sigma_sum = covariance.predict_f_g_h_sum(lwls_A.flatten(), lwls_B.flatten(), lwls_C.flatten(), fl.flatten(), sigma.flatten(), lwls_A.flatten(), lwls_B.flatten(), lwls_C.flatten(), mu_fgh=1.0, amp_f=amp_f, l_f=l_f, amp_g=amp_g, l_g=l_g, amp_h=amp_h, l_h=l_h)

    mu_f = mu[0:(n_pix * n_epochs)]
    mu_g = mu[(n_pix * n_epochs):2 * (n_pix * n_epochs)]
    mu_h = mu[2 * (n_pix * n_epochs):]

    # Reshape outputs
    mu_f.shape = (n_epochs, -1)
    mu_g.shape = (n_epochs, -1)
    mu_h.shape = (n_epochs, -1)
    mu_sum.shape = (n_epochs, -1)


    # Make some multivariate draws

    if draws:
        n_draws = args.draws
        mu_draw = np.random.multivariate_normal(mu, Sigma, size=n_draws)#, (n_pix * n_epochs)))


    for i in range(n_epochs):
        fig, ax = plt.subplots(nrows=5, sharex=True)

        if draws:
            # Unpack the array
            # mu_std = np.std(mu_draw, axis=0)
            # mu_std_f = mu_std[0:(n_pix * n_epochs)]
            # mu_std_g = mu_std[(n_pix * n_epochs):2 * (n_pix * n_epochs)]
            # mu_std_h = mu_std[2 * (n_pix * n_epochs):]

            # If we've actually made draws, go ahead and plot them.
            for j in range(n_draws):
                mu_draw_j = mu_draw[j]

                mu_draw_f = mu_draw_j[0:(n_pix * n_epochs)]
                mu_draw_g = mu_draw_j[(n_pix * n_epochs):2 * (n_pix * n_epochs)]
                mu_draw_h = mu_draw_j[2 * (n_pix * n_epochs): ]
                mu_draw_f.shape = (n_epochs, -1)
                mu_draw_g.shape = (n_epochs, -1)
                mu_draw_h.shape = (n_epochs, -1)

                ax[1].plot(wls[i], mu_draw_f[i], color="0.3", lw=0.3)
                ax[2].plot(wls[i], mu_draw_g[i], color="0.3", lw=0.3)
                ax[3].plot(wls[i], mu_draw_h[i], color="0.3", lw=0.3)

        ax[0].plot(wls[i], fl[i], ".", color="0.4")
        ax[0].plot(wls[i], mu_sum[i], "b")
        ax[0].plot(wls[i], mu_f[i] + mu_g[i] + mu_h[i] + 1.0, "m", ls="-.")
        ax[0].set_ylabel(r"$f + g + h$")
        ax[1].plot(wls[i], mu_f[i], "b")
        ax[1].set_ylabel(r"$f$")
        ax[2].plot(wls[i], mu_g[i], "g")
        ax[2].set_ylabel(r"$g$")
        ax[3].plot(wls[i], mu_h[i], "r")
        ax[3].set_ylabel(r"$h$")
        ax[4].plot(wls[i], fl[i] - mu_sum[i], ".", color="0.4")
        ax[4].set_ylabel("residuals")

        ax[-1].set_xlabel(r"$\lambda$")

        plots_dir = "plots_" + C.chunk_fmt.format(order, wl0, wl1)

        fig.savefig(plots_dir + "/epoch_{:0>2}.png".format(i), dpi=300)
        plt.close("all")
