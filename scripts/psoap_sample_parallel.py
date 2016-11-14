#!/usr/bin/env python

import argparse

parser = argparse.ArgumentParser(description="Sample the distribution across multiple chunks.")
parser.add_argument("run_index", type=int, default=0, help="How many different orbital draws.")
args = parser.parse_args()


from multiprocessing import Process, Pipe
import os
import numpy as np

from psoap.samplers import StateSampler
import psoap.constants as C

from scipy.linalg import cho_factor, cho_solve
from numpy.linalg import slogdet

import gc
import logging

from itertools import chain
from collections import deque
from operator import itemgetter
import shutil

import yaml
from functools import partial

try:
    f = open("config.yaml")
    config = yaml.load(f)
    f.close()
except FileNotFoundError as e:
    print("You need to copy a config.yaml file to this directory, and then edit the values to your particular case.")
    raise

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

config["nchunks"]

# list of keys from 0 to (norders - 1)
chunk_keys = np.arange(config["nchunks"])

# Load data and apply masks

# Set up the logger
logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s -  %(message)s", filename="{}log.log".format(
    Starfish.routdir), level=logging.DEBUG, filemode="w", datefmt='%m/%d/%Y %I:%M:%S %p')


def info(title):
    '''
    Print process information useful for debugging.
    '''
    print(title)
    print('module name:', __name__)
    if hasattr(os, 'getppid'):  # only available on Unix
        print('parent process:', os.getppid())
    print('process id:', os.getpid())


class Chunk:
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
                          "DECIDE": self.decide_Theta,
                          "INST": self.instantiate,
                          "LNPROB": self.lnprob_Theta,
                          "GET_LNPROB": self.get_lnprob,
                          "FINISH": self.finish,
                          "SAVE": self.save,
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
        data = Data[self.key]

        self.wl = data.wl
        self.fl = data.fl
        self.sigma = data.sigma
        self.date = data.date
        self.mask = data.mask

        #TODO Apply mask

        self.logger = logging.getLogger("{} {}".format(self.__class__.__name__, self.order))
        if self.debug:
            self.logger.setLevel(logging.DEBUG)
        else:
            self.logger.setLevel(logging.INFO)

        self.logger.info("Initializing model on Spectrum {}, order {}.".format(self.spectrum_id, self.order_key))

        # Possibly set up temporary holders for V11 matrix.
        # self.V11 =

        self.lnprior = 0.0 # Modified and set by NuisanceSampler.lnprob

        # New outdir based upon id
        self.noutdir = routdir + "{}/".format(self.key)

    def get_lnprob(self, *args):
        '''
        Return the *current* value of lnprob.
        Intended to be called from the master process to
        query the child processes for their current value of lnprob.
        '''
        return self.lnprob

    def lnprob_Theta(self, p):
        '''
        Update the model to the top level orbital (Theta) parameters and then evaluate the lnprob.
        Intended to be called from the master process via the command "LNPROB".
        '''
        try:
            self.update_Theta(p)
            lnp = self.evaluate() # Also sets self.lnprob to new value
            return lnp
        except C.ChunkError:
            self.logger.debug("ChunkError in Theta parameters for chunk {}, sending back -np.inf {}".format(self.key, p))
            return -np.inf

    def evaluate(self):
        '''
        Return the lnprob using the current version of the V11 matrix.
        '''

        self.lnprob_last = self.lnprob

        # TODO change to appropriate model
        # fill out covariance matrix
        self.lnprob = covariance.lnlike_f(V11, wls_A.flatten(), fl, sigma, amp_f, l_f)


    def update_Theta(self, p):
        '''
        Update the model to the current Theta parameters.
        :param p: parameters to update model to
        :type p: model.ThetaParam
        '''

        # Designed to be subclassed based upon what model we want to use.
        self.logger.debug("Updating Theta parameters to {}".format(p))

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

        # Helps keep memory usage low, seems like the numpy routine is slow
        # to clear allocated memory for each iteration.
        gc.collect()

    def revert_Theta(self):
        '''
        Revert the status of the model from a rejected Theta proposal.
        '''

        self.logger.debug("Reverting Theta parameters")
        self.lnprob = self.lnprob_last
        # self.V11 = self.V11_last


    def decide_Theta(self, yes):
        '''
        Interpret the decision from the master process to either accept the parameters and move on OR reject the parameters and revert the Theta model.
        :param yes: if True, accept stellar parameters.
        :type yes: boolean
        '''
        if yes:
            # accept and move on
            self.logger.debug("Deciding to accept Theta parameters")
        else:
            # revert and move on
            self.logger.debug("Deciding to revert Theta parameters")
            self.revert_Theta()

        # Proceed with independent sampling for this chunk, if applicable.
        self.independent_sample(1)

    def update_Phi(self, p):
        '''
        Update the Phi parameters (chunk-level parameters) and data covariance matrix.
        :param params: large dictionary containing cheb, cov, and regions
        '''

        raise NotImplementedError

    def revert_Phi(self, *args):
        '''
        Revert all products from the nuisance parameters, including the data
        covariance matrix.
        '''

        self.logger.debug("Reverting Phi parameters")
        self.lnprob = self.lnprob_last
        # self.V11 = self.V11_last


    def independent_sample(self, niter):
        '''
        Do the independent sampling specific to this echelle order, using the
        attached self.sampler (NuisanceSampler).
        :param niter: number of iterations to complete before returning to master process.
        '''

        self.logger.debug("Beginning independent sampling on Phi parameters")

        if self.lnprob:
            # If we have a current value, pass it to the sampler
            self.p0, self.lnprob, state = self.sampler.run_mcmc(pos0=self.p0, N=niter, lnprob0=self.lnprob)
        else:
            # Otherwise, start from the beginning
            self.p0, self.lnprob, state = self.sampler.run_mcmc(pos0=self.p0, N=niter)

        self.logger.debug("Finished independent sampling on Phi parameters")
        # Don't return anything to the master process.

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

class SampleFlat(Chunk):
    '''
    Sample the theta parameters, while parallelizing the chunks.
    '''
    def initialize(self, key):
        super().initialize(key)

        # Set up p0 and the independent sampler, if necessary

    def finish(self, *args):
        super().finish(*args)
        self.sampler.write(self.noutdir)


# class SampleThetaCheb(Order):
#     def initialize(self, key):
#         super().initialize(key)
#
#         # for now, just use white noise
#         self.data_mat = self.sigma_mat.copy()
#         self.data_mat_last = self.data_mat.copy()
#
#         #Set up p0 and the independent sampler
#         fname = Starfish.specfmt.format(self.spectrum_id, self.order) + "phi.json"
#         phi = PhiParam.load(fname)
#         self.p0 = phi.cheb
#         cov = np.diag(Starfish.config["cheb_jump"]**2 * np.ones(len(self.p0)))
#
#         def lnfunc(p):
#             # turn this into pars
#             self.update_Phi(p)
#             lnp = self.evaluate()
#             self.logger.debug("Evaluated Phi parameters: {} {}".format(p, lnp))
#             return lnp
#
#         def rejectfn():
#             self.logger.debug("Calling Phi revertfn.")
#             self.revert_Phi()
#
#         self.sampler = StateSampler(lnfunc, self.p0, cov, query_lnprob=self.get_lnprob, rejectfn=rejectfn, debug=True)
#
#     def update_Phi(self, p):
#         '''
#         Update the Chebyshev coefficients only.
#         '''
#         self.chebyshevSpectrum.update(p)
#
#     def finish(self, *args):
#         super().finish(*args)
#         fname = Starfish.routdir + Starfish.specfmt.format(self.spectrum_id, self.order) + "/mc.hdf5"
#         self.sampler.write(fname=fname)

# class SampleThetaPhi(Order):
#
#     def initialize(self, key):
#         # Run through the standard initialization
#         super().initialize(key)
#
#         # for now, start with white noise
#         self.data_mat = self.sigma_mat.copy()
#         self.data_mat_last = self.data_mat.copy()
#
#         #Set up p0 and the independent sampler
#         fname = Starfish.specfmt.format(self.spectrum_id, self.order) + "phi.json"
#         phi = PhiParam.load(fname)
#
#         # Set the regions to None, since we don't want to include them even if they
#         # are there
#         phi.regions = None
#
#         #Loading file that was previously output
#         # Convert PhiParam object to an array
#         self.p0 = phi.toarray()
#
#         jump = Starfish.config["Phi_jump"]
#         cheb_len = (self.npoly - 1) if self.chebyshevSpectrum.fix_c0 else self.npoly
#         cov_arr = np.concatenate((Starfish.config["cheb_jump"]**2 * np.ones((cheb_len,)), np.array([jump["sigAmp"], jump["logAmp"], jump["l"]])**2 ))
#         cov = np.diag(cov_arr)
#
#         def lnfunc(p):
#             # Convert p array into a PhiParam object
#             ind = self.npoly
#             if self.chebyshevSpectrum.fix_c0:
#                 ind -= 1
#
#             cheb = p[0:ind]
#             sigAmp = p[ind]
#             ind+=1
#             logAmp = p[ind]
#             ind+=1
#             l = p[ind]
#
#             par = PhiParam(self.spectrum_id, self.order, self.chebyshevSpectrum.fix_c0, cheb, sigAmp, logAmp, l)
#
#             self.update_Phi(par)
#
#             # sigAmp must be positive (this is effectively a prior)
#             # See https://github.com/iancze/Starfish/issues/26
#             if not (0.0 < sigAmp):
#                 self.lnprob_last = self.lnprob
#                 lnp = -np.inf
#                 self.logger.debug("sigAmp was negative, returning -np.inf")
#                 self.lnprob = lnp # Same behavior as self.evaluate()
#             else:
#                 lnp = self.evaluate()
#                 self.logger.debug("Evaluated Phi parameters: {} {}".format(par, lnp))
#
#             return lnp
#
#         def rejectfn():
#             self.logger.debug("Calling Phi revertfn.")
#             self.revert_Phi()
#
#         self.sampler = StateSampler(lnfunc, self.p0, cov, query_lnprob=self.get_lnprob, rejectfn=rejectfn, debug=True)
#
#     def update_Phi(self, p):
#         self.logger.debug("Updating nuisance parameters to {}".format(p))
#
#         # Read off the Chebyshev parameters and update
#         self.chebyshevSpectrum.update(p.cheb)
#
#         # Check to make sure the global covariance parameters make sense
#         #if p.sigAmp < 0.1:
#         #   raise C.ModelError("sigAmp shouldn't be lower than 0.1, something is wrong.")
#
#         max_r = 6.0 * p.l # [km/s]
#
#         # Create a partial function which returns the proper element.
#         k_func = make_k_func(p)
#
#         # Store the previous data matrix in case we want to revert later
#         self.data_mat_last = self.data_mat
#         self.data_mat = get_dense_C(self.wl, k_func=k_func, max_r=max_r) + p.sigAmp*self.sigma_mat
#
#     def finish(self, *args):
#         super().finish(*args)
#         fname = Starfish.routdir + Starfish.specfmt.format(self.spectrum_id, self.order) + "/mc.hdf5"
#         self.sampler.write(fname=fname)

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
model = SampleFlat(debug=True)

# Now that the different processes have been forked, initialize them
pconns, cconns, ps = parallel.initialize(model)

def lnprob(p):

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
