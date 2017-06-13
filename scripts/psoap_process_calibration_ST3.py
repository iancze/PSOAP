#!/usr/bin/env python

import argparse

parser = argparse.ArgumentParser(description="Go through the chunks and try to optimize the calibration, assuming some orbital configuration.")
parser.add_argument("--chunk_index", type=int, help="Only run the calibration on a specific chunk.")
parser.add_argument("--nref", type=int, default=3, help="The number of epochs to select as a 'reference.' ")
parser.add_argument("--ncycles", type=int, default=3, help="The number of cycles to go through optimizing.")
args = parser.parse_args()


import numpy as np
from numpy.polynomial import Chebyshev as Ch
from astropy.io import ascii

import psoap.constants as C
from psoap.data import Chunk, lredshift
from psoap import covariance
from psoap import matrix_functions
from psoap import orbit

import yaml

try:
    f = open("config.yaml")
    config = yaml.load(f)
    f.close()
except FileNotFoundError as e:
    print("You need to copy a config.yaml file to this directory, and then edit the values to your particular case.")
    raise

# read in the chunks.dat file
chunks = ascii.read(config["chunk_file"])
print("Optimizing the calibration for the following chunks of data")
print(chunks)

assert config["model"] == "ST3", "Calibration only implemented for ST3 model for now."

pars = config["parameters"]
q_in = pars["q_in"]
K_in = pars["K_in"] # km/s
e_in = pars["e_in"] #
omega_in = pars["omega_in"] # deg
P_in = pars["P_in"] # days
T0_in = pars["T0_in"] # epoch

q_out = pars["q_out"]
K_out = pars["K_out"] # km/s
e_out = pars["e_out"] #
omega_out = pars["omega_out"] # deg
P_out = pars["P_out"] # days
T0_out = pars["T0_out"] # epoch

gamma = pars["gamma"] # km/s
amp_f = pars["amp_f"] # flux
l_f = pars["l_f"] # km/s
amp_g = pars["amp_g"] # flux
l_g = pars["l_g"] # km/s
amp_h = pars["amp_h"] # flux
l_h = pars["l_h"] # km/s

# Sometimes this is necessary to ensure the matrix stays semi-positive definite.
soften = config["soften"]

limit_array = args.nref


# Go through each chunk and optimize the calibration.
for chunk_index,chunk in enumerate(chunks):
    if (args.chunk_index is not None) and (chunk_index != args.chunk_index):
        continue

    order, wl0, wl1 = chunk
    lwl0 = np.log(wl0)
    lwl1 = np.log(wl1)
    chunk = Chunk.open(order, wl0, wl1)
    print("Optimizing", order, wl0, wl1)

    # Load the data
    wls = chunk.wl
    lwls = chunk.lwl
    mask = chunk.mask
    dates = chunk.date1D
    # Soften the sigmas a little bit to prevent inversion errors.
    sigma = soften * chunk.sigma

    # Applying masks is somewhat tricky, because we need to respect the dimensions of the data in order to apply the
    # velocity shifts on an epoch-by-epoch data.

    # On the other hand, we want to make sure that masked lines do not appear in the least-squares solution for calibration

    # Temporary copy, so that we can do multiple cycle corrections.
    fl_out = np.copy(chunk.fl)
    # And so that we can plot the original to compare, later
    fl_orig = np.copy(chunk.fl)

    # Create an orbit using this chunk
    orb = orbit.models["ST3"](**pars, obs_dates=chunk.date1D)

    # predict velocities for each epoch
    vAs, vBs, vCs = orb.get_component_velocities()

    # shift wavelengths according to these velocities to rest-frame of each component
    # We will be assuming the same orbit throughout, so these will not change
    lwls_A = lredshift(lwls, -vAs[:,np.newaxis])
    lwls_B = lredshift(lwls, -vBs[:,np.newaxis])
    lwls_C = lredshift(lwls, -vCs[:,np.newaxis])

    for cycle in range(args.ncycles):
        print("Optimization cycle:", cycle)
        # Just do all the epochs for now, later leave one out?
        for i in range(chunk.n_epochs):
            lwl_tweak = lwls[i]
            lwl_tweak_unmasked = lwls[i]
            lwl_A_tweak = lwls_A[i]
            lwl_B_tweak = lwls_B[i]
            lwl_C_tweak = lwls_C[i]

            fl_tweak = fl_out[i]
            fl_tweak_unmasked = fl_out[i]
            sigma_tweak = sigma[i]
            mask_tweak = mask[i]

            fl_remain = np.delete(fl_out, i, axis=0)[0:limit_array]
            sigma_remain = np.delete(sigma, i, axis=0)[0:limit_array]
            mask_remain = np.delete(mask, i, axis=0)[0:limit_array]

            # Temporary arrays without the epoch we just chose, always selecting the highest S/N epochs
            lwl_A_remain = np.delete(lwls_A, i, axis=0)[0:limit_array]
            lwl_B_remain = np.delete(lwls_B, i, axis=0)[0:limit_array]
            lwl_C_remain = np.delete(lwls_C, i, axis=0)[0:limit_array]

            # Apply the masks to everything
            lwl_tweak = lwl_tweak[mask_tweak]
            fl_tweak = fl_tweak[mask_tweak]
            sigma_tweak = sigma_tweak[mask_tweak]
            lwl_A_tweak = lwl_A_tweak[mask_tweak]
            lwl_B_tweak = lwl_B_tweak[mask_tweak]
            lwl_C_tweak = lwl_C_tweak[mask_tweak]

            fl_remain = fl_remain[mask_remain]
            sigma_remain = sigma_remain[mask_remain]
            lwl_A_remain = lwl_A_remain[mask_remain]
            lwl_B_remain = lwl_B_remain[mask_remain]
            lwl_C_remain = lwl_C_remain[mask_remain]

            # Make the covariance matrices
            M = len(lwl_A_tweak)
            N = len(lwl_A_remain.flatten())

            V11_f = np.empty((M, M), dtype=np.float)
            V11_g = np.empty((M, M), dtype=np.float)
            V11_h = np.empty((M, M), dtype=np.float)

            matrix_functions.fill_V11_f(V11_f, lwl_A_tweak, amp_f, l_f)
            matrix_functions.fill_V11_f(V11_g, lwl_B_tweak, amp_g, l_g)
            matrix_functions.fill_V11_f(V11_h, lwl_C_tweak, amp_h, l_h)

            V11 = V11_f + V11_g + V11_h
            V11[np.diag_indices_from(V11)] += sigma_tweak**2

            V12_f = np.empty((M, N), dtype=np.float64)
            V12_g = np.empty((M, N), dtype=np.float64)
            V12_h = np.empty((M, N), dtype=np.float64)
            matrix_functions.fill_V12_f(V12_f, lwl_A_tweak, lwl_A_remain, amp_f, l_f)
            matrix_functions.fill_V12_f(V12_g, lwl_B_tweak, lwl_B_remain, amp_g, l_g)
            matrix_functions.fill_V12_f(V12_h, lwl_C_tweak, lwl_C_remain, amp_h, l_h)
            V12 = V12_f + V12_g + V12_h

            V22_f = np.empty((N,N), dtype=np.float)
            V22_g = np.empty((N,N), dtype=np.float)
            V22_h = np.empty((N,N), dtype=np.float)

            # It's a square matrix, so we can just reuse fil_V11_f
            matrix_functions.fill_V11_f(V22_f, lwl_A_remain, amp_f, l_f)
            matrix_functions.fill_V11_f(V22_g, lwl_B_remain, amp_g, l_g)
            matrix_functions.fill_V11_f(V22_h, lwl_C_remain, amp_h, l_h)
            V22 = V22_f + V22_g + V22_h
            V22[np.diag_indices_from(V22)] += sigma_remain**2


            # optimize the calibration of "tweak" with respect to all other orders
            fl_cor, X = covariance.optimize_calibration(lwl0, lwl1, lwl_tweak, fl_tweak, fl_remain, V11, V22, V12, order=config["order_cal"])

            # since fl_cor may have actually have fewer pixels than originally, we can't just
            # stuff the corrected fluxes directly back into the array.
            # Instead, we need to re-evaluate the line on all the wavelengths
            # (including those that may have been masked)
            # using the chebyshev coefficients, and apply this.

            # Here is where we need to make sure that wl0 and wl1 are the same.
            T = []
            for k in range(0, config["order_cal"] + 1):
                pass
                coeff = [0 for j in range(k)] + [1]
                Chtemp = Ch(coeff, domain=[lwl0, lwl1])
                Ttemp = Chtemp(lwl_tweak_unmasked)
                T += [Ttemp]

            T = np.array(T)

            Q = fl_tweak_unmasked[:,np.newaxis] * T.T

            # Apply the correction
            fl_cor = np.dot(Q, X)

            # replace this epoch with the corrected fluxes
            fl_out[i] = fl_cor


    # Save the corrected fluxes
    np.save("plots_" + C.chunk_fmt.format(order, wl0, wl1) + "/fl_cor.npy", fl_out)
