#!/usr/bin/env python

import argparse

parser = argparse.ArgumentParser(description="Measure statistics across multiple chains.")
parser.add_argument("--draws", type=int, default=10, help="How many different orbital draws.")
parser.add_argument("--burn", type=int, default=0, help="How many samples to discard from the beginning of the chain for burn in.")
parser.add_argument("--thin", type=int, default=1, help="How many samples to skip (stride-wise) so to gain independent samples.")
args = parser.parse_args()


import numpy as np
import matplotlib.pyplot as plt
import os

from psoap import constants as C
from psoap import orbit
from psoap import utils
from psoap.data import Chunk
from astropy.io import ascii

import yaml
from functools import partial

try:
    f = open("config.yaml")
    config = yaml.load(f)
    f.close()
except FileNotFoundError as e:
    print("You need to copy a config.yaml file to this directory, and then edit the values to your particular case.")
    raise

# Load the list of chunks
chunks = ascii.read("../../" + config["chunk_file"])

# Load data and apply masks
order, wl0, wl1 = chunks[0]
chunkSpec = Chunk.open(order, wl0, wl1, limit=config["epoch_limit"], prefix="../../")

dates = chunkSpec.date1D

dates_fine = np.linspace(np.min(dates), np.max(dates), num=300)

pars = config["parameters"]

# Create a partial function which maps a vector of floats to parameters
convert_vector_p = partial(utils.convert_vector, model=config["model"], fix_params=config["fix_params"], **pars)


# Choose the orbital model
orb = orbit.models[config["model"]](**pars, obs_dates=dates)


def get_orbit_SB1(p):
    (K, e, omega, P, T0, gamma), (amp_f, l_f) = convert_vector_p(p)

    if K < 0.0 or e < 0.0 or e > 1.0 or P < 0.0 or omega < -180 or omega > 520 or amp_f < 0.0 or l_f < 0.0:
        raise RuntimeError

    # Update the orbit
    orb.K = K
    orb.e = e
    orb.omega = omega
    orb.P = P
    orb.T0 = T0
    orb.gamma = gamma

    # predict velocities for each epoch
    vAs = orb.get_component_velocities()
    vAs_fine = orb.get_component_velocities(dates_fine)

    return (vAs, vAs_fine)

def get_orbit_SB2(p):
    # unroll p
    (q, K, e, omega, P, T0, gamma), (amp_f, l_f, amp_g, l_g) = convert_vector_p(p)

    if q < 0.0 or K < 0.0 or e < 0.0 or e > 1.0 or P < 0.0 or omega < -180 or omega > 520 or amp_f < 0.0 or l_f < 0.0 or amp_g < 0.0 or l_g < 0.0 :
        raise RuntimeError

    # Update the orbit
    orb.q = q
    orb.K = K
    orb.e = e
    orb.omega = omega
    orb.P = P
    orb.T0 = T0
    orb.gamma = gamma

    # predict velocities for each epoch
    vAs, vBs = orb.get_velocities()
    vAs_fine, vBs_fine = orb.get_velocities(dates_fine)

    return (vAs, vAs_fine, vBs, vBs_fine)

def get_orbit_ST3(p):
    q_in, K_in, e_in, omega_in, P_in, T0_in, q_out, K_out, e_out, omega_out, P_out, T0_out, gamma, amp_f, l_f, amp_g, l_g, amp_h, l_h = convert_vector_p(p)

    # Update the orbit
    orb.q_in = q_in
    orb.K_in = K_in
    orb.e_in = e_in
    orb.omega_in = omega_in
    orb.P_in = P_in
    orb.T0_in = T0_in
    orb.q_out = q_out
    orb.K_out = K_out
    orb.e_out = e_out
    orb.omega_out = omega_out
    orb.P_out = P_out
    orb.T0_out = T0_out
    orb.gamma = gamma

    # predict velocities for each epoch
    vAs, vBs, vCs = orb.get_component_velocities()
    vAs_fine, vBs_fine, vCs_fine = orb.get_component_velocities(dates_fine)

    return (vAs, vAs_fine, vBs, vBs_fine, vCs, vCs_fine)


# Read config, data, and samples. Create a set of finely spaced dates, and for each (independent) sample, draw points and plot an orbit.

flatchain = np.load("flatchain.npy")[args.burn::args.thin]
indexes = np.random.choice(np.arange(len(flatchain)), size=args.draws)
flatchain = flatchain[indexes]

if config["model"] == "SB1":
    fig, ax = plt.subplots(figsize=(8,5))
    ax.axhline(pars["gamma"], color="0.4", ls="-.")

    for p in flatchain:
        vAs, vAs_fine = get_orbit_SB1(p)
        ax.plot(dates_fine, vAs_fine, color="0.4", lw=0.5)
        ax.plot(dates, vAs, "o", color="0.4")

    # Save the velocities from a random draw.
    np.save("vA_model.npy", vAs)

    fig.savefig("orbits.png", dpi=300)

elif config["model"] == "SB2":
    fig, ax = plt.subplots(figsize=(8,5))
    ax.axhline(pars["gamma"], color="0.4", ls="-.")

    for p in flatchain:
        vAs, vAs_fine, vBs, vBs_fine = get_orbit_SB2(p)
        ax.plot(dates_fine, vAs_fine, color="b", lw=0.5, alpha=0.3)
        ax.plot(dates_fine, vBs_fine, color="g", lw=0.5, alpha=0.3)
        ax.plot(dates, vAs, ".", color="b")
        ax.plot(dates, vBs, ".", color="g")

    # Save the velocities from a random draw.
    np.save("vA_model.npy", vAs)
    np.save("vB_model.npy", vBs)

    ax.set_xlabel("Julian Date")

    fig.savefig("orbits.png", dpi=300)

    # Now make an orbital phase plot
    fig, ax = plt.subplots(figsize=(8,5))
    ax.axhline(pars["gamma"], color="0.4", ls="-.")

    for p in flatchain:
        vAs, vAs_fine, vBs, vBs_fine = get_orbit_SB2(p)
        (q, K, e, omega, P, T0, gamma), (amp_f, l_f, amp_g, l_g) = convert_vector_p(p)

        phase = ((dates - T0) % P) / P
        phase_fine = ((dates_fine - T0) % P) / P

        indsort = np.argsort(phase)
        phase = phase[indsort]
        vAs = vAs[indsort]
        vBs = vBs[indsort]

        indsort = np.argsort(phase_fine)
        phase_fine = phase_fine[indsort]
        vAs_fine = vAs_fine[indsort]
        vBs_fine = vBs_fine[indsort]


        ax.plot(phase_fine, vAs_fine, color="b", lw=0.5, alpha=0.3)
        ax.plot(phase_fine, vBs_fine, color="g", lw=0.5, alpha=0.3)
        ax.plot(phase, vAs, ".", color="b")
        ax.plot(phase, vBs, ".", color="g")

    ax.set_xlabel(r"$\phi$")
    fig.savefig("orbits_phase.png", dpi=300)

elif config["model"] == "ST3":

    # Make a full orbital figure, then make two figures for sampling as a function of orbital phase

    fig, ax = plt.subplots(nrows=4, figsize=(8,5), sharex=True)
    ax[0].axhline(pars["gamma"], color="0.4", ls="-.")


    # If we have the actual true velocities, (fake data), plot them
    if os.path.exists("vAs_relative.npy"):
        vAs_relative = np.load("vAs_relative.npy")
        vBs_relative = np.load("vBs_relative.npy")
        vCs_relative = np.load("vCs_relative.npy")

        ax[1].plot(dates, vAs_relative, "ko")
        ax[2].plot(dates, vBs_relative, "ko")
        ax[3].plot(dates, vCs_relative, "ko")


    # find the index that corresponds to the minimum date
    ind0 = np.argmin(dates)

    for p in flatchain:
        vAs, vAs_fine, vBs, vBs_fine, vCs, vCs_fine = get_orbit_ST3(p)

        ax[0].plot(dates_fine, vAs_fine, color="b", lw=0.5, alpha=0.3)
        ax[0].plot(dates_fine, vBs_fine, color="g", lw=0.5, alpha=0.3)
        ax[0].plot(dates_fine, vCs_fine, color="r", lw=0.5, alpha=0.3)
        ax[0].plot(dates, vAs, ".", color="b")
        ax[0].plot(dates, vBs, ".", color="g")
        ax[0].plot(dates, vCs, ".", color="r")

        ax[1].plot(dates, vAs - vAs[ind0], "b.")

        ax[2].plot(dates, vBs - vBs[ind0], "g.")

        ax[3].plot(dates, vCs - vCs[ind0], "r.")

    ax[0].set_ylabel(r"$v$ [km/s]")
    ax[1].set_ylabel(r"$v_A$ relative")
    ax[2].set_ylabel(r"$v_B$ relative")
    ax[3].set_ylabel(r"$v_C$ relative")
    ax[-1].set_xlabel("date [day]")

    # Save the velocities from a random draw.
    np.save("vA_model.npy", vAs)
    np.save("vB_model.npy", vBs)
    np.save("vC_model.npy", vCs)

    fig.savefig("orbits.png", dpi=300)

    # Now make the orbital phase plots


    # Convert dates to orbital phase

    fig, ax = plt.subplots(ncols=2, figsize=(8,6))

    for p in flatchain:
        vAs, vAs_fine, vBs, vBs_fine, vCs, vCs_fine = get_orbit_ST3(p)

        q_in, K_in, e_in, omega_in, P_in, T0_in, q_out, K_out, e_out, omega_out, P_out, T0_out, gamma, amp_f, l_f, amp_g, l_g, amp_h, l_h = convert_vector_p(p)

        phase_inner = (dates - T0_in) % P_in
        phase_inner_fine = (dates_fine - T0_in) % P_in

        phase_outer = (dates - T0_out) % P_out
        phase_outer_fine = (dates_fine - T0_in) % P_out

        # plot the outer orbit on the left
        ax[0].plot(phase_outer_fine, vAs_fine, color="b", lw=0.5, alpha=0.3)
        ax[0].plot(phase_outer_fine, vBs_fine, color="g", lw=0.5, alpha=0.3)
        ax[0].plot(phase_outer_fine, vCs_fine, color="r", lw=0.5, alpha=0.3)
        ax[0].plot(phase_outer, vAs, ".", color="b")
        ax[0].plot(phase_outer, vBs, ".", color="g")
        ax[0].plot(phase_outer, vCs, ".", color="r")

        # plot the inner orbit on the right
        ax[1].plot(phase_inner_fine, vAs_fine, color="b", lw=0.5, alpha=0.3)
        ax[1].plot(phase_inner_fine, vBs_fine, color="g", lw=0.5, alpha=0.3)
        ax[1].plot(phase_inner_fine, vCs_fine, color="r", lw=0.5, alpha=0.3)
        ax[1].plot(phase_inner, vAs, ".", color="b")
        ax[1].plot(phase_inner, vBs, ".", color="g")
        ax[1].plot(phase_inner, vCs, ".", color="r")

    ax[0].set_ylabel(r"$v$ [km/s]")
    ax[1].set_ylabel(r"$v$ [km/s]")

    ax[0].set_xlabel("outer phase")
    ax[1].set_xlabel("inner phase")

    fig.savefig("orbits_phase.png", dpi=300)


else:
    print("model not implemented yet")
