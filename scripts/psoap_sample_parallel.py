#!/usr/bin/env python

import argparse

parser = argparse.ArgumentParser(description="Sample the distribution across multiple chunks.")
parser.add_argument("run_index", type=int, default=0, help="Which output subdirectory to save this particular run, in the case you may be running multiple concurrently.")
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

from psoap.samplers import StateSampler
import psoap.constants as C
from psoap.data import Chunk
from psoap import utils

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
    chunkSpec = Chunk.open(order, wl0, wl1)
    chunkSpec.apply_mask()
    chunk_data.append(chunkSpec)

# Set up the logger
logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s -  %(message)s", filename="{}log.log".format(
    Starfish.routdir), level=logging.DEBUG, filemode="w", datefmt='%m/%d/%Y %I:%M:%S %p')

# Create a partial function which maps a vector of floats to parameters
convert_vector_p = partial(utils.convert_vector, model=config["model"], fix_params=config["fix_params"], **pars)

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
        self.lnprob = -np.inf
        self.lnprob_last = -np.inf

        # The list of possible function calls we can make.
        self.func_dict = {"INIT": self.initialize,
                          "LNPROB": self.lnprob,
                          "FINISH": self.finish
                          }

        self.debug = debug
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

        self.wl = data.wl
        self.fl = data.fl
        self.sigma = data.sigma
        self.date = data.date

        # Note that mask is already applied in loading step. This is to transform velocity shifts
        # Evaluated off of self.date1D
        self.mask = data.mask

        self.logger = logging.getLogger("{} {}".format(self.__class__.__name__, self.order))
        if self.debug:
            self.logger.setLevel(logging.DEBUG)
        else:
            self.logger.setLevel(logging.INFO)

        self.logger.info("Initializing model on Spectrum {}, order {}.".format(self.spectrum_id, self.order_key))

        # Possibly set up temporary holders for V11 matrix.
        # self.N is the length of the masked, flattened wl vector.
        self.V11 = np.empty((self.N, self.N), dtype=np.float64)

        # Create an orbit
        self.orb = orbit.models[config["model"]](**pars, obs_dates=self.date1D)

        # Choose which lnprob we will be using based off of the model type
        lnprobs = {"SB1":self.lnprob_SB1, "SB2":self.lnprob_SB2, "ST3":self.lnprob_ST3}
        self.lnprob = lnprobs[config["model"]]

    def lnprob_SB1(self, p):
        '''
        Update the model to the top level orbital (Theta) parameters and then evaluate the lnprob.
        Intended to be called from the master process via the command "LNPROB".
        '''

        # Designed to be subclassed based upon what model we want to use.
        self.logger.debug("Updating orbital parameters to {}".format(p))

        # unroll p
        K, e, omega, P, T0, gamma, amp_f, l_f = convert_vector_p(p)

        # Update the orbit
        self.orb.K = K
        self.orb.e = e
        self.orb.omega = omega
        self.orb.P = P
        self.orb.T0 = T0
        self.orb.gamma = gamma

        # predict velocities for each epoch
        vAs = self.orb.get_component_velocities()

        # shift wavelengths according to these velocities to rest-frame of A component
        wls_A = redshift(self.chunk.wl, -vAs[:,np.newaxis][self.chunk.mask])

        # fill out covariance matrix
        lnp = covariance.lnlike_f(self.V11, wls_A.flatten(), self.fl, self.sigma, amp_f, l_f)

        gc.collect()

        return lnp

    def lnprob_SB2(self, p):
        # unroll p
        q, K, e, omega, P, T0, gamma, amp_f, l_f, amp_g, l_g = convert_vector_p(p)

        # Update the orbit
        self.orb.q = q
        self.orb.K = K
        self.orb.e = e
        self.orb.omega = omega
        self.orb.P = P
        self.orb.T0 = T0
        self.orb.gamma = gamma

        # predict velocities for each epoch
        vAs, vBs = self.orb.get_component_velocities()

        # shift wavelengths according to these velocities to rest-frame of A component
        wls_A = redshift(self.chunk.wl, -vAs[:,np.newaxis][self.chunk.mask])
        wls_B = redshift(self.chunk.wl, -vBs[:,np.newaxis][self.chunk.mask])

        # fill out covariance matrix
        lnp = covariance.lnlike_f_g(self.V11, wls_A.flatten(), wls_B.flatten(), self.chunk.fl, self.chunk.sigma, amp_f, l_f, amp_g, l_g)

        gc.collect()

        return lnp

    def lnprob_ST3(self, p):
        raise NotImplementedError

    def finish(self, *args):
        '''
        Wrap up the sampling and write the samples to disk.
        '''
        self.logger.debug("Finishing")

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
        #info("brain")

        fname, arg = self.conn.recv() # Waits here to receive a new message
        self.logger.debug("{} received message {}".format(os.getpid(), (fname, arg)))

        func = self.func_dict.get(fname, False)
        if func:
            response = func(arg)
        else:
            self.logger.info("Given an unknown function {}, assuming kill signal.".format(fname))
            return False

        # Functions only return a response other than None when they want them
        # communicated back to the master process.
        # Some commands sent to the child processes do not require a response
        # to the main process.
        if response:
            self.logger.debug("{} sending back {}".format(os.getpid(), response))
            self.conn.send(response)
        return True

class FlatWorker(Worker):
    '''
    Sample the theta parameters, while parallelizing the chunks.
    '''
    def initialize(self, key):
        super().initialize(key)

        # Set up p0 and the independent sampler, if necessary

    def finish(self, *args):
        super().finish(*args)
        self.sampler.write(self.noutdir)


# We create one Order() in the main process. When the process forks, each
# subprocess now has its own independent OrderModel instance.
# Then, each forked model will be customized using an INIT command passed
# through the PIPE.

def initialize(model):
    # Fork a subprocess for each key: (spectra, order)
    pconns = {} # Parent connections
    cconns = {} # Child connections
    ps = {} # Process objects
    # Create all of the pipes
    for key in chunk_keys:
        pconn, cconn = Pipe()
        pconns[key], cconns[key] = pconn, cconn
        p = Process(target=model.brain, args=(cconn,))
        p.start()
        ps[key] = p

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
    stellar_Starting.update({"logg":4.29})
    model.stellar_lnprob(stellar_Starting)
    #Assume we accepted
    model.decide_stellar(True)

    #Right now, assumes Kurucz order 23

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
model = FlatWorker(debug=True)

# Now that the different processes have been forked, initialize them
pconns, cconns, ps = parallel.initialize(model)

# Optionally load a user-defined prior

def lnprob(p):

    # TODO Optionally apply a user-defined prior here

    # Assume p is [K, e, omega, etc...]

    #Distribute the calculation, one chunk to each process
    for (key, pconn) in pconns.items():
        pconn.send(("LNPROB", p))

    #Collect the answer from each process
    lnps = np.empty(config["nchunks"])
    for i, pconn in enumerate(pconns.values()):
        lnps[i] = pconn.recv()

    # Calculate the summed lnprob
    s = np.sum(lnps)

    # Add any priors
    return s


# lnprobs = {"SB1":lnprob_SB1, "SB2":lnprob_SB2, "ST3":lnprob_ST3}

# Import the Metropolis-hastings sampler to do the sampling in the top level parameters
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

sampler = MHSampler(cov, dim, lnprob)

for i, result in enumerate(sampler.sample(p0, iterations=config["samples"])):
    if (i+1) % 20 == 0:
        print("Iteration", i +1)

# Save the actual chain of samples
print("Acceptance fraction", sampler.acceptance_fraction)
np.save("lnprob.npy", sampler.lnprobability)
np.save("flatchain.npy", sampler.flatchain)

# Kill all of the orders
for pconn in pconns.values():
    pconn.send(("FINISH", None))
    pconn.send(("DIE", None))

# Join on everything and terminate
for p in ps.values():
    p.join()
    p.terminate()

import sys;sys.exit()
