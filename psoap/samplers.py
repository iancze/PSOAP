#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
A Metropolis-Hastings and Gibbs samplers for a state-ful model.
"""

import numpy as np
from emcee import autocorr
from emcee.sampler import Sampler
import logging
import h5py

class StateSampler(Sampler):
    """
    The most basic possible Metropolis-Hastings style MCMC sampler, but with
    optional callbacks for acceptance and rejection.
    :param cov:
        The covariance matrix to use for the proposal distribution.
    :param dim:
        Number of dimensions in the parameter space.
    :param lnpostfn:
        A function that takes a vector in the parameter space as input and
        returns the natural logarithm of the posterior probability for that
        position.
    :param query_lnprob:
        A function which queries the model for the current lnprob value. This is
        because an intermediate sampler might have intervened.
    :param rejectfn: (optional)
        A function to be executed if the proposal is rejected. Generally this
        should be a function which reverts the model to the previous parameter
        values. Might need to be a closure.
    :param acceptfn: (optional)
        A function to be executed if the proposal is accepted.
    :param args: (optional)
        A list of extra positional arguments for ``lnpostfn``. ``lnpostfn``
        will be called with the sequence ``lnpostfn(p, *args, **kwargs)``.
    :param kwargs: (optional)
        A list of extra keyword arguments for ``lnpostfn``. ``lnpostfn``
        will be called with the sequence ``lnpostfn(p, *args, **kwargs)``.
    """
    def __init__(self, lnprob, p0, cov, query_lnprob=None, rejectfn=None,
        acceptfn=None, debug=False, outdir="", *args, **kwargs):
        dim = len(p0)
        super().__init__(dim, lnprob, *args, **kwargs)
        self.cov = cov
        self.p0 = p0
        self.query_lnprob = query_lnprob
        self.rejectfn = rejectfn
        self.acceptfn = acceptfn
        self.logger = logging.getLogger(self.__class__.__name__)
        self.outdir = outdir
        self.debug = debug
        if self.debug:
            self.logger.setLevel(logging.DEBUG)
        else:
            self.logger.setLevel(logging.INFO)

    def reset(self):
        super().reset()
        self._chain = np.empty((0, self.dim))
        self._lnprob = np.empty(0)

    def sample(self, p0, lnprob0=None, randomstate=None, thin=1,
               storechain=True, iterations=1, incremental_save=0, **kwargs):
        """
        Advances the chain ``iterations`` steps as an iterator
        :param p0:
            The initial position vector.
        :param lnprob0: (optional)
            The log posterior probability at position ``p0``. If ``lnprob``
            is not provided, the initial value is calculated.
        :param rstate0: (optional)
            The state of the random number generator. See the
            :func:`random_state` property for details.
        :param iterations: (optional)
            The number of steps to run.
        :param thin: (optional)
            If you only want to store and yield every ``thin`` samples in the
            chain, set thin to an integer greater than 1.
        :param storechain: (optional)
            By default, the sampler stores (in memory) the positions and
            log-probabilities of the samples in the chain. If you are
            using another method to store the samples to a file or if you
            don't need to analyse the samples after the fact (for burn-in
            for example) set ``storechain`` to ``False``.
        At each iteration, this generator yields:
        * ``pos`` - The current positions of the chain in the parameter
          space.
        * ``lnprob`` - The value of the log posterior at ``pos`` .
        * ``rstate`` - The current state of the random number generator.
        """

        self.random_state = randomstate

        p = np.array(p0)
        if lnprob0 is None:
            # See if there's something there
            lnprob0 = self.query_lnprob()

            # If not, we're on the first iteration
            if lnprob0 is None:
                lnprob0 = self.get_lnprob(p)

        # Resize the chain in advance.
        if storechain:
            N = int(iterations / thin)
            self._chain = np.concatenate((self._chain,
                                          np.zeros((N, self.dim))), axis=0)
            self._lnprob = np.append(self._lnprob, np.zeros(N))

        i0 = self.iterations

        # Use range instead of xrange for python 3 compatability
        for i in range(int(iterations)):
            self.iterations += 1

            # Since the independent nuisance sampling may have changed parameters,
            # query each process for the current lnprob
            lnprob0 = self.query_lnprob()
            self.logger.debug("Queried lnprob: {}".format(lnprob0))

            # Calculate the proposal distribution.
            if self.dim == 1:
                q = self._random.normal(loc=p[0], scale=self.cov[0], size=(1,))
            else:
                q = self._random.multivariate_normal(p, self.cov)

            newlnprob = self.get_lnprob(q)
            diff = newlnprob - lnprob0
            self.logger.debug("old lnprob: {}".format(lnprob0))
            self.logger.debug("proposed lnprob: {}".format(newlnprob))

            # M-H acceptance ratio
            if diff < 0:
                diff = np.exp(diff) - self._random.rand()
                if diff < 0:
                    #Reject the proposal and revert the state of the model
                    self.logger.debug("Proposal rejected")
                    if self.rejectfn is not None:
                        self.rejectfn()

            if diff > 0:
                #Accept the new proposal
                self.logger.debug("Proposal accepted")
                p = q
                lnprob0 = newlnprob
                self.naccepted += 1
                if self.acceptfn is not None:
                    self.acceptfn()

            if storechain and i % thin == 0:
                ind = i0 + int(i / thin)
                self._chain[ind, :] = p
                self._lnprob[ind] = lnprob0

            # The default of 0 evaluates to False
            if incremental_save:
                if (((i+1) % incremental_save) == 0) & (i > 0):
                    np.save('chain_backup.npy', self._chain)

            # Heavy duty iterator action going on right here...
            yield p, lnprob0, self.random_state

    @property
    def acor(self):
        """
        An estimate of the autocorrelation time for each parameter (length:
        ``dim``).
        """
        return self.get_autocorr_time()

    def get_autocorr_time(self, window=50):
        """
        Compute an estimate of the autocorrelation time for each parameter
        (length: ``dim``).
        :param window: (optional)
            The size of the windowing function. This is equivalent to the
            maximum number of lags to use. (default: 50)
        """
        return autocorr.integrated_time(self.chain, axis=0, window=window)

    def write(self, fname="mc.hdf5"):
        '''
        Write the samples to an HDF file.
        flatchain
        acceptance fraction
        Everything can be saved in the dataset self.fname
        '''

        filename = self.outdir + fname
        self.logger.debug("Opening {} for writing HDF5 flatchains".format(filename))
        hdf5 = h5py.File(filename, "w")
        samples = self.flatchain

        dset = hdf5.create_dataset("samples", samples.shape, compression='gzip', compression_opts=9)
        dset[:] = samples
        dset.attrs["acceptance"] = "{}".format(self.acceptance_fraction)
        hdf5.close()
