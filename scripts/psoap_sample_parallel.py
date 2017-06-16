#!/usr/bin/env python

import argparse

parser = argparse.ArgumentParser(description="Sample the distribution across multiple chunks.")
parser.add_argument("run_index", type=int, default=0, help="Which output subdirectory to save this particular run, in the case you may be running multiple concurrently.")
parser.add_argument("--debug", action="store_true", help="Print out debug commands to log.log")
args = parser.parse_args()

import yaml
from functools import partial

try:
    f = open("config.yaml")
    config = yaml.load(f)
    f.close()
except FileNotFoundError as e:
    print("You need to copy a config.yaml file to this directory, and then edit the values to your particular case.")
    raise

from multiprocessing import Process, Pipe
import os
import numpy as np
from astropy.io import ascii

# from psoap.samplers import StateSampler
import psoap.constants as C
from psoap.data import Chunk, lredshift
from psoap import utils
from psoap import orbit
from psoap import covariance

from scipy.linalg import cho_factor, cho_solve
from numpy.linalg import slogdet

import gc
import logging

from itertools import chain
from collections import deque
from operator import itemgetter
import shutil

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

# When running a hierarchical model, we'll need to do this.
# # Create subdirectories
# for model_number in range(len(Starfish.data["files"])):
#     for order in Starfish.data["orders"]:
#         order_dir = routdir + Starfish.specfmt.format(model_number, order)
#         print("Creating ", order_dir)
#         os.makedirs(order_dir)

# Load the list of chunks
chunks = ascii.read(config["chunk_file"])
print("Sampling the following chunks of data, one chunk per core.")
print(chunks)

n_chunks = len(chunks)
# list of keys from 0 to (norders - 1)
chunk_keys = np.arange(n_chunks)

# Load data and apply masks
chunk_data = []
for chunk in chunks:
    order, wl0, wl1 = chunk
    chunkSpec = Chunk.open(order, wl0, wl1, limit=config["epoch_limit"])
    chunkSpec.apply_mask()
    chunk_data.append(chunkSpec)

# The name of the model
model = config["model"]
pars = config["parameters"]

# Set up the logger
if args.debug:
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s -  %(message)s", filename="{}log.log".format(routdir), level=logging.DEBUG, filemode="w", datefmt='%m/%d/%Y %I:%M:%S %p')

# Create a partial function which maps a vector of floats to parameters
convert_vector_p = partial(utils.convert_vector, model=model, fix_params=config["fix_params"], **pars)

def info(title):
    '''
    Print process information useful for debugging.
    '''
    print(title)
    print('module name:', __name__)
    if hasattr(os, 'getppid'):  # only available on Unix
        print('parent process:', os.getppid())
    print('process id:', os.getpid())


class Worker:
    def __init__(self, debug=False):
        '''
        This object contains all of the variables necessary for the partial
        lnprob calculation for one chunk. It is designed to first be
        instantiated within the main processes and then forked to other
        subprocesses. Once operating in the subprocess, the variables specific
        to the order are loaded with an `INIT` message call, which tells which key
        to initialize on in the `self.initialize()`.
        '''

        # Choose which lnprob we will be using based off of the model type
        # lnprobs = {"SB1":self.lnprob_SB1, "SB2":self.lnprob_SB2, "ST3":self.lnprob_ST3}
        # self.lnprob = lnprobs[model]

        # The list of possible function calls we can make.
        self.func_dict = {"INIT": self.initialize,
                          "LNPROB": self.lnprob,
                          "FINISH": self.finish
                          }

        self.debug = debug
        if args.debug:
            self.logger = logging.getLogger("{}".format(self.__class__.__name__))

    def initialize(self, key):
        '''
        Initialize to the correct chunk of data.
        :param key: key
        :param type: int
        This method should only be called after all subprocess have been forked.
        '''

        self.key = key

        # Load the proper chunk
        data = chunk_data[self.key]

        self.lwl = data.lwl
        self.fl = data.fl
        self.sigma = data.sigma * config["soften"]
        self.date = data.date

        # Note that mask is already applied in loading step. This is to transform velocity shifts
        # Evaluated off of self.date1D
        self.mask = data.mask
        self.date1D = data.date1D

        # Total number of wavelength points (after applying mask)
        self.N = data.N

        if args.debug:
            self.logger = logging.getLogger("{} {}".format(self.__class__.__name__, self.key))
            if self.debug:
                self.logger.setLevel(logging.DEBUG)
            else:
                self.logger.setLevel(logging.INFO)

            self.logger.info("Initializing model on chunk {}.".format(self.key))

        # Possibly set up temporary holders for V11 matrix.
        # self.N is the length of the masked, flattened wl vector.
        self.V11 = np.empty((self.N, self.N), dtype=np.float64)

        # Create an orbit
        self.orb = orbit.models[model](**pars, obs_dates=self.date1D)

    def lnprob(self, p):
        '''
        Unified lnprob interface.

        Args:
            p (np.float): vector containing the model parameters

        Returns:
            float : the lnlikelihood of the model parameters.
        '''

        # separate the parameters into orbital and GP based upon the model type
        # also backfill any parameters that we have fixed for this analysis
        p_orb, p_GP = convert_separate_p(p)

        velocities = orbit.models[model](*p_orb, self.date1D).get_velocities()

        # Make sure none are faster than speed of light
        if np.any(np.abs(np.array(velocities)) >= C.c_kms):
            return -np.inf

        # Get shifted wavelengths
        lwls = replicate_wls(self.lwl, velocities, self.mask)

        # Feed velocities and GP parameters to fill out covariance matrix appropriate for this model
        lnp = covariance.lnlike[model](self.V11, *lwls, self.fl, self.sigma, *p_GP)

        gc.collect()

        return lnp


    # def lnprob_SB1(self, p):
    #     '''
    #     Update the model to the top level orbital (Theta) parameters and then evaluate the lnprob.
    #     Intended to be called from the master process via the command "LNPROB".
    #     '''
    #
    #     if args.debug:
    #         # Designed to be subclassed based upon what model we want to use.
    #         self.logger.debug("Updating orbital parameters to {}".format(p))
    #
    #     # unroll p
    #     K, e, omega, P, T0, gamma, amp_f, l_f = convert_vector_p(p)
    #
    #     # Update the orbit
    #     self.orb.K = K
    #     self.orb.e = e
    #     self.orb.omega = omega
    #     self.orb.P = P
    #     self.orb.T0 = T0
    #     self.orb.gamma = gamma
    #
    #     # predict velocities for each epoch
    #     vAs = self.orb.get_component_velocities()[0]
    #
    #     # Make sure none are faster than speed of light
    #     if np.any(np.abs(vAs) >= C.c_kms):
    #         return -np.inf
    #
    #
    #     # shift wavelengths according to these velocities to rest-frame of A component
    #     lwls_A = lredshift(self.lwl, (-vAs[:,np.newaxis] * np.ones_like(self.mask))[self.mask])
    #
    #     # fill out covariance matrix
    #     lnp = covariance.lnlike_f(self.V11, lwls_A.flatten(), self.fl, self.sigma, amp_f, l_f)
    #
    #     gc.collect()
    #
    #     return lnp
    #
    # def lnprob_SB2(self, p):
    #     # unroll p
    #     q, K, e, omega, P, T0, gamma, amp_f, l_f, amp_g, l_g = convert_vector_p(p)
    #
    #     # Update the orbit
    #     self.orb.q = q
    #     self.orb.K = K
    #     self.orb.e = e
    #     self.orb.omega = omega
    #     self.orb.P = P
    #     self.orb.T0 = T0
    #     self.orb.gamma = gamma
    #
    #     # predict velocities for each epoch
    #     vAs, vBs = self.orb.get_component_velocities()
    #
    #     # Make sure none are faster than speed of light
    #     if np.any(np.abs(vAs) >= C.c_kms) or np.any(np.abs(vBs) >= C.c_kms):
    #         return -np.inf
    #
    #
    #     # shift wavelengths according to these velocities to rest-frame of A component
    #     lwls_A = lredshift(self.lwl, (-vAs[:,np.newaxis] * np.ones_like(self.mask))[self.mask])
    #     lwls_B = lredshift(self.lwl, (-vBs[:,np.newaxis] * np.ones_like(self.mask))[self.mask])
    #
    #     # fill out covariance matrix
    #     lnp = covariance.lnlike_f_g(self.V11, lwls_A.flatten(), lwls_B.flatten(), self.fl, self.sigma, amp_f, l_f, amp_g, l_g)
    #
    #     if lnp == -np.inf and args.debug:
    #         self.logger.debug("Worker {} evaulated -np.inf".format(self.key))
    #
    #     gc.collect()
    #
    #     return lnp
    #
    # def lnprob_ST3(self, p):
    #     # unroll p
    #     q_in, K_in, e_in, omega_in, P_in, T0_in, q_out, K_out, e_out, omega_out, P_out, T0_out, gamma, amp_f, l_f, amp_g, l_g, amp_h, l_h = convert_vector_p(p)
    #
    #     # Update the orbit
    #     self.orb.q_in = q_in
    #     self.orb.K_in = K_in
    #     self.orb.e_in = e_in
    #     self.orb.omega_in = omega_in
    #     self.orb.P_in = P_in
    #     self.orb.T0_in = T0_in
    #     self.orb.q_out = q_out
    #     self.orb.K_out = K_out
    #     self.orb.e_out = e_out
    #     self.orb.omega_out = omega_out
    #     self.orb.P_out = P_out
    #     self.orb.T0_out = T0_out
    #     self.orb.gamma = gamma
    #
    #     # predict velocities for each epoch
    #     vAs, vBs, vCs = self.orb.get_component_velocities()
    #
    #     # Make sure none are faster than speed of light
    #     if np.any(np.abs(vAs) >= C.c_kms) or np.any(np.abs(vBs) >= C.c_kms) or np.any(np.abs(vCs) >= C.c_kms):
    #         return -np.inf
    #
    #     # shift wavelengths according to these velocities to rest-frame of A component
    #     lwls_A = lredshift(self.lwl, (-vAs[:,np.newaxis] * np.ones_like(self.mask))[self.mask])
    #     lwls_B = lredshift(self.lwl, (-vBs[:,np.newaxis] * np.ones_like(self.mask))[self.mask])
    #     lwls_C = lredshift(self.lwl, (-vCs[:,np.newaxis] * np.ones_like(self.mask))[self.mask])
    #
    #     # fill out covariance matrix
    #     lnp = covariance.lnlike_f_g_h(self.V11, lwls_A.flatten(), lwls_B.flatten(), lwls_C.flatten(), self.fl, self.sigma, amp_f, l_f, amp_g, l_g, amp_h, l_h)
    #
    #     if lnp == -np.inf and args.debug:
    #         self.logger.debug("Worker {} evaulated -np.inf".format(self.key))
    #
    #     gc.collect()
    #
    #     return lnp



    def finish(self, *args):
        '''
        Wrap up the sampling and write the samples to disk.
        '''
        pass

    def brain(self, conn):
        '''
        The infinite loop of the subprocess, which continues to listen for
        messages on the pipe.
        '''
        self.conn = conn
        alive = True
        while alive:
            #Keep listening for messages put on the Pipe
            alive = self.interpret()
            #Once self.interpret() returns `False`, this loop will die.
        self.conn.send("DEAD")

    def interpret(self):
        '''
        Interpret the messages being put into the Pipe, and do something with
        them. Messages are always sent in a 2-arg tuple (fname, arg)
        Right now we only expect one function and one argument but this could
        be generalized to **args.
        '''
        # info("brain")

        fname, arg = self.conn.recv() # Waits here to receive a new message
        if args.debug:
            self.logger.debug("{} received message {}".format(os.getpid(), (fname, arg)))

        func = self.func_dict.get(fname, False)
        if func:
            response = func(arg)
        else:
            if args.debug:
                self.logger.info("Given an unknown function {}, assuming kill signal.".format(fname))
            return False

        # Functions only return a response other than None when they want them
        # communicated back to the master process.
        # Some commands sent to the child processes do not require a response
        # to the main process.
        if response:
            if args.debug:
                self.logger.debug("{} sending back {}".format(os.getpid(), response))
            self.conn.send(response)
        return True

# Moving forward, we have the option to subclass Worker if we want to alter routines.

# We create one Order() in the main process. When the process forks, each
# subprocess now has its own independent OrderModel instance.
# Then, each forked model will be customized using an INIT command passed
# through the PIPE.

def initialize(worker):
    # Fork a subprocess for each key: (spectra, order)
    pconns = {} # Parent connections
    cconns = {} # Child connections
    ps = {} # Process objects
    # Create all of the pipes
    for key in chunk_keys:
        pconn, cconn = Pipe()
        pconns[key], cconns[key] = pconn, cconn
        p = Process(target=worker.brain, args=(cconn,))
        p.start()
        ps[key] = p

    # print("created keys", chunk_keys)
    # print("conns", pconns, cconns)

    # initialize each Model to a specific chunk
    for key, pconn in pconns.items():
        pconn.send(("INIT", key))

    return (pconns, cconns, ps)


def profile_code():
    '''
    Test hook designed to be used by cprofile or kernprof. Does not include any
    network latency from communicating or synchronizing between processes
    because we run on just one process.
    '''

    #Evaluate one complete iteration from delivery of stellar parameters from master process

    #Master proposal
    # stellar_Starting.update({"logg":4.29})
    # model.stellar_lnprob(stellar_Starting)
    #Assume we accepted
    # model.decide_stellar(True)

    #Right now, assumes Kurucz order 23
    pass

def test():

    # Uncomment these lines to profile
    # #Initialize the current model for profiling purposes
    # model.initialize((0, 0))
    # import cProfile
    # cProfile.run("profile_code()", "prof")
    # import sys; sys.exit()

    # Kill all of the orders
    for pconn in pconns.values():
        pconn.send(("FINISH", None))
        pconn.send(("DIE", None))

    # Join on everything and terminate
    for p in ps.values():
        p.join()
        p.terminate()

    import sys;sys.exit()

# All subprocesses will inherit pipe file descriptors created in the master process.
# http://www.pushingbits.net/posts/python-multiprocessing-with-pipes/
# thus, to really close a pipe, you need to close it in every subprocess.

# Create the main sampling loop, which will sample the theta parameters across all chunks
worker = Worker(debug=True)

# Now that the different processes have been forked, initialize them
pconns, cconns, ps = initialize(worker)


def prior_SB1(p):
    K, e, omega, P, T0, gamma, amp_f, l_f = convert_vector_p(p)

    if K < 0.0 or e < 0.0 or e > 1.0 or P < 0.0 or omega < -90 or omega > 450 or amp_f < 0.0 or l_f < 0.0:
        return -np.inf

    else:
        return 0.0

def prior_SB2(p):
    q, K, e, omega, P, T0, gamma, amp_f, l_f, amp_g, l_g = convert_vector_p(p)

    if q < 0.0 or K < 0.0 or e < 0.0 or e > 1.0 or P < 0.0 or omega < -90 or omega > 450 or amp_f < 0.0 or l_f < 0.0 or amp_g < 0.0 or l_g < 0.0:
        return -np.inf
    else:
        return 0.0


def prior_ST3(p):
    q_in, K_in, e_in, omega_in, P_in, T0_in, q_out, K_out, e_out, omega_out, P_out, T0_out, gamma, amp_f, l_f, amp_g, l_g, amp_h, l_h = convert_vector_p(p)

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


def lnprob(p):

    lnprior = prior(p)
    if lnprior == -np.inf:
        return -np.inf

    #Distribute the calculation, one chunk to each process
    for (key, pconn) in pconns.items():
        pconn.send(("LNPROB", p))

    #Collect the answer from each process
    lnps = np.empty(n_chunks)
    for i, pconn in enumerate(pconns.values()):
        lnps[i] = pconn.recv()

    # Calculate the summed lnprob
    s = np.sum(lnps)

    # Add any the prior to the total
    return s + lnprior


# Import the Metropolis-hastings sampler to do the sampling in the top level parameters
from emcee import MHSampler

# Determine how many parameters we will actually be fitting
# The difference between all of the parameters and the parameters we will be fixing
dim = len(utils.registered_params[model]) - len(config["fix_params"])

# Read in starting parameters
p0 = utils.convert_dict(model, config["fix_params"], **pars)

# To check feasibility, evaluate the starting position. If this evaluates to -np.inf, then just
# exit, since we might be wasting our time evaluating the rest.
lnp0 = lnprob(p0)
if lnp0 == -np.inf:
    print("Starting position for Markov Chain evaluates to -np.inf")
    # Kill all of the orders
    for pconn in pconns.values():
        pconn.send(("FINISH", None))
        pconn.send(("DIE", None))

    # Join on everything and terminate
    for p in ps.values():
        p.join()
        p.terminate()

    raise RuntimeError

else:
    print("Starting position good. lnp: {}".format(lnp0))

try:
    cov = np.load(config["opt_jump"])
    print("using optimal jumps")
except:
    print("using hand-specified jumps")
    cov = utils.convert_dict(model, config["fix_params"], **config["jumps"])**2 * np.eye(dim)

sampler = MHSampler(cov, dim, lnprob)

for i, result in enumerate(sampler.sample(p0, iterations=config["samples"])):
    if (i+1) % 20 == 0:
        print("Iteration", i +1)

# Save the actual chain of samples
print("Acceptance fraction", sampler.acceptance_fraction)
np.save(routdir + "lnprob.npy", sampler.lnprobability)
np.save(routdir + "flatchain.npy", sampler.flatchain)

# Kill all of the orders
for pconn in pconns.values():
    pconn.send(("FINISH", None))
    pconn.send(("DIE", None))

# Join on everything and terminate
for p in ps.values():
    p.join()
    p.terminate()

import sys;sys.exit()
