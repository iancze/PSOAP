import numpy as np
import matplotlib.pyplot as plt

from .. import constants as C
from ..data import Chunk, lredshift, replicate_lwls
from .. import utils
from .. import orbit
from .. import covariance

import yaml
from functools import partial
import os

from scipy.linalg import cho_factor, cho_solve
from numpy.linalg import slogdet

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

# Load data and apply masks
chunk_data = []
for chunk in chunks:
    order, wl0, wl1 = chunk
    chunkSpec = Chunk.open(order, wl0, wl1, limit=config["epoch_limit"])
    chunkSpec.apply_mask()
    chunk_data.append(chunkSpec)

# Take the dates of the first chunk for use in the orbit
date1D = chunk_data[0].date1D

# The name of the model
model = config["model"]
pars = config["parameters"]

# Create a partial function which maps a vector of floats to parameters
convert_vector_p = partial(utils.convert_vector, model=config["model"], fix_params=config["fix_params"], **pars)

# Initialize the orbit
orb = orbit.models[model](**pars, obs_dates=date1D)

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
        return -np.inf

    # go through each chunk, evaluate the likelihood, and sum them together at the end
    lnp = 0
    for data in chunk_data:

        # Total number of wavelength points (after applying mask)
        N = data.N
        V11 = np.empty((N, N), dtype=np.float)

        # Get shifted wavelengths
        lwls = replicate_lwls(data.lwl, velocities, data.mask)

        # Feed velocities and GP parameters to fill out covariance matrix appropriate for this model
        lnp += covariance.lnlike[model](V11, *lwls, data.fl, data.sigma, *p_GP)

    gc.collect()

    return lnp


# Optionally load a user-defined prior.
# Check if a file named "prior.py" exists in the local folder
# If so, import it
try:
    from prior import prior
    print("Loaded user defined prior.")
except ImportError:
    print("Using default prior.")
    # Set the default priors.
    prior = utils.priors[model]

def lnp(p):

    # The priors defined in utils.py expect two arguments, p_orb and p_gp, which are each vectors
    # of the full parameter array defined in utils.py
    # this code takes the MCMC proposal, fills it to a 2-tuple of (p_orb, p_gp), and then unpacks it
    # to the call of the prior.
    lnprior = prior(*convert_vector_p(p))
    if lnprior == -np.inf:
        return -np.inf

    s = lnprob(p)

    # Add any the prior to the total
    return s + lnprior


def main():

    import argparse

    parser = argparse.ArgumentParser(description="Sample the distribution across multiple chunks.")
    parser.add_argument("run_index", type=int, default=0, help="Which output subdirectory to save this particular run, in the case you may be running multiple concurrently.")
    args = parser.parse_args()

    # Create an output directory to store the samples from this run
    run_index = args.run_index
    routdir = config["outdir"] + "/run{:0>2}/".format(run_index)
    if os.path.exists(routdir):
        print("Deleting", routdir)
        shutil.rmtree(routdir)

    print("Creating ", routdir)
    os.makedirs(routdir)
    # Copy yaml file from current working directory to routdir for archiving purposes
    shutil.copy("config.yaml", routdir + "config.yaml")

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
    np.save(routdir + "lnprob.npy", sampler.lnprobability)
    np.save(routdir + "flatchain.npy", sampler.flatchain)
