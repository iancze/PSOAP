#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

from scipy.linalg import cho_factor, cho_solve

from psoap import constants as C
from psoap.data import redshift
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

# Load the data
wls = np.load("fake_SB2_wls.npy")
fl = np.load("fake_SB2_fls.npy")
sigma = np.load("fake_SB2_sigmas.npy")
dates = np.load("fake_SB2_dates.npy")

# Use the parameters specified in the yaml file to create the spectra
pars = config["parameters"]

q = pars["q"]
K = pars["K"] # km/s
e = pars["e"] #
omega = pars["omega"] # deg
P = pars["P"] # days
T0 = pars["T0"] # epoch
gamma = pars["gamma"] # km/s
a_f = pars["a_f"] # flux
l_f = pars["l_f"] # km/s
a_g = pars["a_g"] # flux
l_g = pars["l_g"] # km/s

orb = orbit.SB2(q, K, e, omega, P, T0, gamma, obs_dates=dates)

# predict velocities for each epoch
vAs, vBs = orb.get_component_velocities()

# shift wavelengths according to these velocities to rest-frame of A component
wls_A = redshift(wls, -vAs[:,np.newaxis])
wls_B = redshift(wls, -vBs[:,np.newaxis])

n_epochs, n_pix = wls_A.shape

# First predict the component spectra as mean 1 GPs
mu, Sigma = covariance.predict_f_g(wls_A.flatten(), wls_B.flatten(), wls.flatten(), fl.flatten(), sigma.flatten(), mu_f=0.0, mu_g=0.0, amp_f=a_f, l_f=l_f, amp_g=a_g, l_g=l_g)

mu_sum, Sigma_sum = covariance.predict_f_g_sum(wls_A.flatten(), wls_B.flatten(), fl.flatten(), sigma.flatten(), wls_A.flatten(), wls_B.flatten(), mu_fg=1.0, amp_f=a_f, l_f=l_f, amp_g=a_g, l_g=l_g)

mu_f = mu[0:(n_pix * n_epochs)]
mu_g = mu[(n_pix * n_epochs):]

# Reshape outputs
mu_f.shape = (n_epochs, -1)
mu_g.shape = (n_epochs, -1)
mu_sum.shape = (n_epochs, -1)

for i in range(n_epochs):
    fig, ax = plt.subplots(nrows=4, sharex=True)

    ax[0].plot(wls[i], fl[i], ".", color="0.4")
    ax[0].plot(wls[i], mu_sum[i], "b")
    ax[0].plot(wls[i], mu_f[i] + mu_g[i] + 1.0, "m", ls="-.")
    ax[0].ylabel(r"$f + g$")
    ax[1].plot(wls[i], mu_f[i], "b")
    ax[1].set_ylabel(r"$f$")
    ax[2].plot(wls[i], mu_g[i], "g")
    ax[2].set_ylabel(r"$g$")

    ax[3].plot(wls[i], fl[i] - mu_sum[i], ".", color="0.4")
    ax[3].set_ylabel("residuals")
    
    ax[-1].set_xlabel(r"$\lambda$")

    fig.savefig("epoch_{}.png".format(i), dpi=300)
    plt.close("all")
