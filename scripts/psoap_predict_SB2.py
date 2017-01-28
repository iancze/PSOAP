#!/usr/bin/env python

import argparse

parser = argparse.ArgumentParser(description="Measure statistics across multiple chains.")
parser.add_argument("--draws", type=int, default=0, help="In addition to plotting the mean GP, plot several draws of the GP to show the scatter in predicitions.")
args = parser.parse_args()

draws = (args.draws > 0)

import numpy as np
import matplotlib.pyplot as plt

from astropy.io import ascii
from scipy.linalg import cho_factor, cho_solve

from psoap import constants as C
from psoap.data import redshift, Chunk
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


for row in chunks:
    # For now, only use the first chunk.
    order, wl0, wl1 = row
    chunk = Chunk.open(order, wl0, wl1, limit=config["epoch_limit"])

    # Load the data
    wls = chunk.wl
    fl = chunk.fl
    sigma = chunk.sigma
    dates = chunk.date1D

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

    orb = orbit.SB2(q, K, e, omega, P, T0, gamma, obs_dates=dates)

    # predict velocities for each epoch
    vAs, vBs = orb.get_component_velocities()

    # shift wavelengths according to these velocities to rest-frame of A component
    wls_A = redshift(wls, -vAs[:,np.newaxis])
    wls_B = redshift(wls, -vBs[:,np.newaxis])

    n_epochs, n_pix = wls_A.shape

    # First predict the component spectra as mean 1 GPs
    mu, Sigma = covariance.predict_f_g(wls_A.flatten(), wls_B.flatten(), fl.flatten(), sigma.flatten(), wls_A.flatten(), wls_B.flatten(), mu_f=0.0, mu_g=0.0, amp_f=amp_f, l_f=l_f, amp_g=amp_g, l_g=l_g)

    # Also predict the sum of the spectra
    mu_sum, Sigma_sum = covariance.predict_f_g_sum(wls_A.flatten(), wls_B.flatten(), fl.flatten(), sigma.flatten(), wls_A.flatten(), wls_B.flatten(), mu_fg=1.0, amp_f=amp_f, l_f=l_f, amp_g=amp_g, l_g=l_g)

    mu_f = mu[0:(n_pix * n_epochs)]
    mu_g = mu[(n_pix * n_epochs):]

    # Reshape outputs
    mu_f.shape = (n_epochs, -1)
    mu_g.shape = (n_epochs, -1)
    mu_sum.shape = (n_epochs, -1)


    # Make some multivariate draws
    if draws:
        n_draws = args.ndraws
        mu_draw = np.random.multivariate_normal(mu, Sigma, size=n_draws)#, (n_pix * n_epochs)))

    for i in range(n_epochs):
        fig, ax = plt.subplots(nrows=4, sharex=True)

        # std_envelope
        # mu_std = np.std(mu_draw, axis=0)
        # mu_std_f = mu_std[0:(n_pix * n_epochs)]
        # mu_std_g = mu_std[(n_pix * n_epochs):]
        # print(mu_std_f)
        # print(mu_std_g)
        # ax[1].fill_between(wls[i], mu_f[i] - mu_std_f[i], mu_f[i] + mu_std_f[i], color="0.8")
        # ax[2].fill_between(wls[i], mu_g[i] - mu_std_g[i], mu_g[i] + mu_std_g[i], color="0.8")

        if draws:
            for j in range(5):
                mu_draw_j = mu_draw[j]

                mu_draw_f = mu_draw_j[0:(n_pix * n_epochs)]
                mu_draw_g = mu_draw_j[(n_pix * n_epochs):]
                mu_draw_f.shape = (n_epochs, -1)
                mu_draw_g.shape = (n_epochs, -1)

                ax[1].plot(wls[i], mu_draw_f[i], color="0.2", lw=0.5)
                ax[2].plot(wls[i], mu_draw_g[i], color="0.2", lw=0.5)

        ax[0].plot(wls[i], fl[i], ".", color="0.4")
        ax[0].plot(wls[i], mu_sum[i], "b")
        ax[0].plot(wls[i], mu_f[i] + mu_g[i] + 1.0, "m", ls="-.")
        ax[0].set_ylabel(r"$f + g$")
        ax[1].plot(wls[i], mu_f[i], "b")
        ax[1].set_ylabel(r"$f$")
        ax[2].plot(wls[i], mu_g[i], "g")
        ax[2].set_ylabel(r"$g$")

        residuals = fl[i] - mu_sum[i]
        ax[3].plot(wls[i], residuals, ".", color="0.4")
        ax[3].set_ylabel("residuals")

        ax[-1].set_xlabel(r"$\lambda$")

        plots_dir = "plots_" + C.chunk_fmt.format(order, wl0, wl1)

        fig.savefig(plots_dir + "/epoch_{:0>2}.png".format(i), dpi=300)

        # make a histogram of the residuals
        fig, ax = plt.subplots(figsize=(4,4))
        ax.hist(residuals/sigma[i], normed=True)
        # Plot an actual Gaussian profile
        sig = np.linspace(-4, 4, num=50)
        val = 1/np.sqrt(2 * np.pi) * np.exp(-0.5 * sig**2)
        ax.plot(sig, val)
        ax.set_xlabel("sigma")
        fig.savefig(plots_dir + "/epoch_{:0>2}_hist.png".format(i), dpi=300)

        plt.close("all")
