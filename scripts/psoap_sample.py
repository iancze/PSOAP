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
from functools import partial

try:
    f = open("config.yaml")
    config = yaml.load(f)
    f.close()
except FileNotFoundError as e:
    print("You need to copy a config.yaml file to this directory, and then edit the values to your particular case.")
    raise

# Load the data
wls = np.load("fake_SB1_wls.npy")
fl = np.load("fake_SB1_fls.npy").flatten()
sigma = np.load("fake_SB1_sigmas.npy").flatten()
dates = np.load("fake_SB1_dates.npy")

pars = config["parameters"]

# Create a partial function which maps a vector of floats to parameters
convert_vector_p = partial(utils.convert_vector, model=config["model"], fix_params=config["fix_params"], **pars)

# Initialize the orbit with some bogus values (or best-guesses, if you have them)
orb = orbit.models[config["model"]](**pars, obs_dates=dates)

N = len(wls.flatten())
V11 = np.empty((N, N), dtype=np.float)

def lnprob_SB1(p):
    # unroll p
    K, e, omega, P, T0, gamma, amp_f, l_f = convert_vector_p(p)

    if K < 0.0 or e < 0.0 or e > 1.0 or P < 0.0 or omega < -180 or omega > 520 or amp_f < 0.0 or l_f < 0.0:
        return -np.inf

    # Update the orbit
    orb.K = K
    orb.e = e
    orb.omega = omega
    orb.P = P
    orb.T0 = T0
    orb.gamma = gamma

    # predict velocities for each epoch
    vAs = orb.get_component_velocities()

    # shift wavelengths according to these velocities to rest-frame of A component
    wls_A = redshift(wls, -vAs[:,np.newaxis])

    # fill out covariance matrix
    lnp = covariance.lnlike_f(V11, wls_A.flatten(), fl, sigma, amp_f, l_f)

    return lnp


def lnprob_SB2(p):
    # unroll p
    q, K, e, omega, P, T0, gamma, amp_f, l_f, amp_g, l_g = convert_vector_p(p)

    if q < 0.0 or q > 1.0 or K < 0.0 or e < 0.0 or e > 1.0 or P < 0.0 or omega < -180 or omega > 520 or amp_f < 0.0 or l_f < 0.0:
        return -np.inf

    # Update the orbit
    orb.q = q
    orb.K = K
    orb.e = e
    orb.omega = omega
    orb.P = P
    orb.T0 = T0
    orb.gamma = gamma

    # predict velocities for each epoch
    vAs, vBs = orb.get_component_velocities()

    # shift wavelengths according to these velocities to rest-frame of A component
    wls_A = redshift(wls, -vAs[:,np.newaxis])
    wls_B = redshift(wls, -vBs[:,np.newaxis])

    # fill out covariance matrix
    lnp = covariance.lnlike_f_g(V11, wls_A.flatten(), wls_B.flatten(), fl, sigma, amp_f, l_f, amp_g, l_g)

    return lnp

def lnprob_ST3(p):
    raise NotImplementedError

lnprobs = {"SB1":lnprob_SB1, "SB2":lnprob_SB2, "ST3":lnprob_ST3}

# Import the Metropolis-hastings sampler
from emcee import MHSampler

# Determine how many parameters we will actually be fitting
# The difference between all of the parameters and the parameters we will be fixing
dim = len(utils.registered_params[config["model"]]) - len(config["fix_params"])

# Read in starting parameters
p0 = utils.convert_dict(config["model"], config["fix_params"], **pars)

try:
    cov = np.load(config["opt_jump"])
    print("using optimal jumps")
except:
    print("using hand-specified jumps")
    cov = utils.convert_dict(config["model"], config["fix_params"], **config["jumps"])**2

sampler = MHSampler(cov, dim, lnprobs[config["model"]])

for i, result in enumerate(sampler.sample(p0, iterations=config["samples"])):
    if (i+1) % 20 == 0:
        print("Iteration", i +1)

# Save the actual chain of samples
print("Acceptance fraction", sampler.acceptance_fraction)
np.save("lnprob.npy", sampler.lnprobability)
np.save("flatchain.npy", sampler.flatchain)
