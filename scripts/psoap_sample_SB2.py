#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

from scipy.linalg import cho_factor, cho_solve

from psoap import constants as C
from psoap.data import redshift
from psoap import covariance
from psoap import orbit


# Infer the orbital parameters of the SB1
# This is a simple test-case, so for now we'll only use one chunk.

# Load the data
wls = np.load("fake_SB2_wls.npy")
fl = np.load("fake_SB2_fls.npy").flatten()
sigma = np.load("fake_SB2_sigmas.npy").flatten()
dates = np.load("fake_SB2_dates.npy")

# Initialize the orbit with some bogus values (or best-guesses, if you have them)
q = 0.2
K = 5.0 # km/s
e = 0.20 #
omega = 12.0 # deg
P = 10.0 # days
T0 = 0.0 # epoch
gamma = 0.0 # km/s

orb = orbit.SB2(q, K, e, omega, P, T0, gamma, obs_dates=dates)

N = len(wls.flatten())

V11 = np.empty((N, N), dtype=np.float)

# Create the likelihood function which generates the orbit.
def lnprob(p):
    # unroll p
    q, K, e, omega, P, T0, amp_f, l_f, amp_g, l_g = p

    if q < 0.0 or q > 1.0 or K < 0.0 or e < 0.0 or e > 1.0 or P < 0.0 or omega < -180 or omega > 520 or amp_f < 0.0 or l_f < 0.0:
        return -np.inf

    # Update the orbit
    orb.q = q
    orb.K = K
    orb.e = e
    orb.omega = omega
    orb.P = P
    orb.T0 = T0

    # predict velocities for each epoch
    vAs, vBs = orb.get_component_velocities()

    # shift wavelengths according to these velocities to rest-frame of A component
    wls_A = redshift(wls, -vAs[:,np.newaxis])
    wls_B = redshift(wls, -vBs[:,np.newaxis])

    # fill out covariance matrix
    lnp = covariance.lnlike_two(V11, wls_A.flatten(), wls_B.flatten(), fl, sigma, amp_f, l_f, amp_g, l_g)

    return lnp


# Import the Metropolis-hastings sampler
from emcee import MHSampler

dim = 10
p0 = np.array([q, K, e, omega, P, T0, 0.2, 6.0, 0.04, 7.0])

try:
    cov = np.load("opt_jump.npy")
    print("using optimal jumps")
except:
    print("using hand-specified jumps")
    cov = np.diag(np.array([0.005, 0.05, 0.01, 0.4, 0.05, 0.05, 0.01, 0.4, 0.01, 0.4])**2)


sampler = MHSampler(cov, dim, lnprob)

nsteps = 6000
for i, result in enumerate(sampler.sample(p0, iterations=nsteps)):
    if (i+1) % 20 == 0:
        print("Iteration", i +1)
        # print("{0:5.1%}".format(float(i) / nsteps))

# Save the actual chain of samples
print("Acceptance fraction", sampler.acceptance_fraction)
np.save("lnprob.npy", sampler.lnprobability)
np.save("flatchain.npy", sampler.flatchain)
