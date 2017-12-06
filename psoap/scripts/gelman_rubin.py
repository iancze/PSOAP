#!/usr/bin/env python

import argparse

parser = argparse.ArgumentParser(description="Measure statistics across multiple chains.")
parser.add_argument("glob", help="Do something on this glob. Must be given as a quoted expression to avoid shell expansion.")

parser.add_argument("--burn", type=int, default=0, help="How many samples to discard from the beginning of the chain for burn in.")

args = parser.parse_args()

import numpy as np
import sys
from astropy.table import Table
from astropy.io import ascii


def gelman_rubin(samplelist):
    '''
    Given a list of flatchains from separate runs (that already have burn in cut
    and have been trimmed, if desired), compute the Gelman-Rubin statistics in
    Bayesian Data Analysis 3, pg 284. If you want to compute this for fewer
    parameters, then slice the list before feeding it in.
    '''

    full_iterations = len(samplelist[0])
    assert full_iterations % 2 == 0, "Number of iterations must be even. Try cutting off a different number of burn in samples."
    shape = samplelist[0].shape
    #make sure all the chains have the same number of iterations
    for flatchain in samplelist:
        assert len(flatchain) == full_iterations, "Not all chains have the same number of iterations!"
        assert flatchain.shape == shape, "Not all flatchains have the same shape!"

    #make sure all chains have the same number of parameters.

    #Following Gelman,
    # n = length of split chains
    # i = index of iteration in chain
    # m = number of split chains
    # j = index of which chain
    n = full_iterations//2
    m = 2 * len(samplelist)
    nparams = samplelist[0].shape[-1] #the trailing dimension of a flatchain

    #Block the chains up into a 3D array
    chains = np.empty((n, m, nparams))
    for k, flatchain in enumerate(samplelist):
        chains[:,2*k,:] = flatchain[:n]  #first half of chain
        chains[:,2*k + 1,:] = flatchain[n:] #second half of chain

    #Now compute statistics
    #average value of each chain
    avg_phi_j = np.mean(chains, axis=0, dtype="f8") #average over iterations, now a (m, nparams) array
    #average value of all chains
    avg_phi = np.mean(chains, axis=(0,1), dtype="f8") #average over iterations and chains, now a (nparams,) array

    B = n/(m - 1.0) * np.sum((avg_phi_j - avg_phi)**2, axis=0, dtype="f8") #now a (nparams,) array

    s2j = 1./(n - 1.) * np.sum((chains - avg_phi_j)**2, axis=0, dtype="f8") #now a (m, nparams) array

    W = 1./m * np.sum(s2j, axis=0, dtype="f8") #now a (nparams,) arary

    var_hat = (n - 1.)/n * W + B/n #still a (nparams,) array
    std_hat = np.sqrt(var_hat)

    R_hat = np.sqrt(var_hat/W) #still a (nparams,) array


    data = Table({   "Value": avg_phi,
                     "Uncertainty": std_hat},
                 names=["Value", "Uncertainty"])

    print(data)

    ascii.write(data, sys.stdout, Writer = ascii.Latex, formats={"Value":"%0.3f", "Uncertainty":"%0.3f"}) #

    #print("Average parameter value: {}".format(avg_phi))
    #print("std_hat: {}".format(np.sqrt(var_hat)))
    print("R_hat: {}".format(R_hat))

    if np.any(R_hat >= 1.1):
        print("You might consider running the chain for longer. Not all R_hats are less than 1.1.")


def cat_list(file, flatchainList):
    '''
    Given a list of flatchains, concatenate all of these and write them to a
    single HDF5 file.
    '''
    #Write this out to the new file
    print("Opening", file)
    hdf5 = h5py.File(file, "w")

    cat = np.concatenate(flatchainList, axis=0)

    # id = flatchainList[0].id
    # param_tuple = flatchainList[0].param_tuple

    dset = hdf5.create_dataset("samples", cat.shape, compression='gzip',
        compression_opts=9)
    dset[:] = cat
    # dset.attrs["parameters"] = "{}".format(param_tuple)

    hdf5.close()


#Now that all of the structures have been declared, do the initialization stuff.
from glob import glob
files = glob(args.glob)

#Because we are impatient and want to compute statistics before all the jobs are finished,
# there may be some directories that do not have a flatchains.hdf5 file
flatchainList = []
for f in files:
    try:
        flatchainList.append(np.load(f)[args.burn::]) # thin by 100
    except OSError as e:
        print("{} does not exist, skipping. Or error {}".format(f, e))

print("Using a total of {} flatchains".format(len(flatchainList)))

assert len(flatchainList) > 1, "If running Gelman-Rubin test, must provide more than one flatchain"
gelman_rubin(flatchainList)

# Cat together all of the flatchains and save as one big flatchain
combined = np.concatenate(flatchainList, axis=0)
print(combined.shape)
np.save("flatchain_combined.npy", combined)
