import numpy as np
import matplotlib.pyplot as plt

import psoap.constants as C
from psoap.data import Chunk, lredshift, replicate_wls
from psoap import utils
from psoap import orbit
from psoap import covariance

import yaml
from functools import partial

from scipy.linalg import cho_factor, cho_solve
from numpy.linalg import slogdet

from scipy.sparse.linalg import LinearOperator, cg
from scipy.optimize import minimize
import celerite
from celerite import terms

from astropy.io import ascii

import gc
import logging

import shutil

# Import the Metropolis-hastings sampler
from emcee import MHSampler

# Do all the global setup
try:
    f = open("config.yaml")
    config = yaml.load(f)
    f.close()
except FileNotFoundError as e:
    print("You need to copy a config.yaml file to this directory, and then edit the values to your particular case.")
    raise

pars = config["parameters"]



# Load the list of chunks
chunks = ascii.read(config["chunk_file"])
print("Sampling the first chunk of data.")

order, wl0, wl1 = chunks[0]
data = Chunk.open(order, wl0, wl1, limit=config["epoch_limit"])
data.apply_mask()

# The name of the model
model = config["model"]
pars = config["parameters"]


# Create a partial function which maps a vector of floats to parameters
convert_vector_p = partial(utils.convert_vector, model=config["model"], fix_params=config["fix_params"], **pars)

lwl = data.lwl
fl = data.fl - np.median(data.fl)
sigma = data.sigma * config["soften"]
dates = data.date

# Note that mask is already applied in loading step. This is to transform velocity shifts
# Evaluated off of self.date1D
mask = data.mask
date1D = data.date1D

# Total number of wavelength points (after applying mask)
N = data.N

# Initialize the orbit
orb = orbit.models[model](**pars, obs_dates=date1D)


# term = terms.SHOTerm(log_S0=-7.0, log_omega0=10, log_Q=2.)
term1 = terms.Matern32Term(log_sigma=-4.0, log_rho=-22)# ) #, log_Q=-0.5*np.log(2))
term2 = terms.Matern32Term(log_sigma=-6.0, log_rho=-22)# ) #, log_Q=-0.5*np.log(2))
# term += terms.JitterTerm(log_sigma=np.log(np.median(sigma)))

# term = terms.Matern32Term(log_sigma=-1.53, log_rho=-10.7)
# term += terms.JitterTerm(log_sigma=np.log(np.median(sigma)))


gp1 = celerite.GP(term1)
gp2 = celerite.GP(term2)

# def lnprob(p):
#     '''
#     Unified lnprob interface.
#
#     Args:
#         p (np.float): vector containing the model parameters
#
#     Returns:
#         float : the lnlikelihood of the model parameters.
#     '''
#
#     # separate the parameters into orbital and GP based upon the model type
#     # also backfill any parameters that we have fixed for this analysis
#     p_orb, p_GP = convert_vector_p(p)
#
#     velocities = orbit.models[model](*p_orb, date1D).get_velocities()
#
#     # Make sure none are faster than speed of light
#     if np.any(np.abs(np.array(velocities)) >= C.c_kms):
#         return -np.inf
#
#     # Get shifted wavelengths
#     lwls = replicate_wls(lwl, velocities, mask)
#
#     # Feed velocities and GP parameters to fill out covariance matrix appropriate for this model
#     lnp = covariance.lnlike[model](V11, *lwls, fl, sigma, *p_GP)
#     # lnp = covariance.lnlike_f_g_george(*lwls, self.fl, self.sigma, *p_GP)
#
#     gc.collect()
#
#     return lnp

def lnprob(p):
    '''
    Unified lnprob interface.

    Args:
        p (np.float): vector containing the model parameters

    Returns:
        float : the lnlikelihood of the model parameters.
    '''

    # separate the parameters into orbital and GP based upon the model type
    # also backfill any parameters that we have fixed for this analysis
    p_orb, p_GP = convert_vector_p(p)

    velocities = orbit.models[model](*p_orb, date1D).get_velocities()

    # Make sure none are faster than speed of light
    if np.any(np.abs(np.array(velocities)) >= C.c_kms):
        print("Velocities greater than lightspeed")
        return -np.inf

    # Get shifted wavelengths
    lwla, lwlb = replicate_wls(lwl, velocities, mask)

    # Sort each vector in wavelength
    inds1 = np.argsort(lwla)
    lwla = lwla[inds1]

    inds2 = np.argsort(lwlb)
    lwlb = lwlb[inds2]

    # Define a custom "LinearOperator"
    # Given a vector v, compute K dot v
    def matvec(v):
        a = gp1.dot(v[inds1], lwla, check_sorted=False)
        res = np.empty_like(v)
        res[inds1] = gp1.dot(v[inds1], lwla, check_sorted=False)[:, 0]
        res[inds2] += gp2.dot(v[inds2], lwlb, check_sorted=False)[:, 0]
        res[inds2] += v[inds2] * sigma[inds2]**2
        # res[inds2] += sigma[inds2]**2
        return res
    op = LinearOperator((N, N), matvec=matvec)

    # Solve the system and compute the first term of the log likelihood, (K^-1 fl)
    soln = cg(op, fl, tol=0.01)

    # Then, re-dot the other fl into this
    lnp = 0.5 * np.dot(fl, soln[0])

    return lnp

def prior_SB1(p):
    (K, e, omega, P, T0, gamma), (amp_f, l_f) = convert_vector_p(p)

    if K < 0.0 or e < 0.0 or e > 1.0 or P < 0.0 or omega < -90 or omega > 450 or amp_f < 0.0 or l_f < 0.0:
        return -np.inf

    else:
        return 0.0

def prior_SB2(p):
    (q, K, e, omega, P, T0, gamma), (amp_f, l_f, amp_g, l_g) = convert_vector_p(p)

    if q < 0.0 or K < 0.0 or e < 0.0 or e > 1.0 or P < 0.0 or omega < -90 or omega > 450:
        return -np.inf
    else:
        return 0.0


def prior_ST3(p):
    (q_in, K_in, e_in, omega_in, P_in, T0_in, q_out, K_out, e_out, omega_out, P_out, T0_out, gamma), (amp_f, l_f, amp_g, l_g, amp_h, l_h) = convert_vector_p(p)

    if q_in < 0.0 or K_in < 0.0 or e_in < 0.0 or e_in > 1.0 or P_in < 0.0 or omega_in < -90 or omega_in > 450 or q_out < 0.0 or K_out < 0.0 or e_out < 0.0 or e_out > 1.0 or P_out < 0.0 or omega_out < -90 or omega_out > 450 or amp_f < 0.0 or l_f < 0.0 or amp_g < 0.0 or l_g < 0.0 or amp_h < 0.0 or l_h < 0.0:
        return -np.inf

    else:
        return 0.0

# Optionally load a user-defined prior.
# Check if a file named "prior.py" exists in the local folder
# If so, import it
try:
    from prior import prior
    print("Loaded user defined prior.")
except ImportError:
    print("Using default prior.")
    # Set the default priors.
    priors = {"SB1":prior_SB1, "SB2":prior_SB2, "ST3":prior_ST3}
    prior = priors[model]

def lnp(p):

    lnprior = prior(p)
    if lnprior == -np.inf:
        return -np.inf

    s = lnprob(p)

    # Add any the prior to the total
    return s + lnprior


def main():

    # Do Argparse stuff

    # Load config files
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
        cov = np.diag(utils.convert_dict(config["model"], config["fix_params"], **config["jumps"])**2)

    sampler = MHSampler(cov, dim, lnp)

    for i, result in enumerate(sampler.sample(p0, iterations=config["samples"])):
        if (i+1) % 20 == 0:
            print("Iteration", i +1)

    # Save the actual chain of samples
    print("Acceptance fraction", sampler.acceptance_fraction)
    np.save("lnprob.npy", sampler.lnprobability)
    np.save("flatchain.npy", sampler.flatchain)
