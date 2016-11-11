#!/usr/bin/env python

import argparse

parser = argparse.ArgumentParser(description="Measure statistics across multiple chains.")
parser.add_argument("--draws", type=int, default=10, help="How many different orbital draws.")
parser.add_argument("--burn", type=int, default=0, help="How many samples to discard from the beginning of the chain for burn in.")
parser.add_argument("--thin", type=int, default=0, help="How many samples to skip (stride-wise) so to gain independent samples.")
args = parser.parse_args()


import numpy as np
import matplotlib.pyplot as plt

from psoap import constants as C
from psoap import orbit
from psoap import utils

import yaml
from functools import partial

try:
    f = open("config.yaml")
    config = yaml.load(f)
    f.close()
except FileNotFoundError as e:
    print("You need to copy a config.yaml file to this directory, and then edit the values to your particular case.")
    raise


# Load the data
dates = np.load("fake_SB2_dates.npy")

dates_fine = np.linspace(np.min(dates), np.max(dates), num=100)

pars = config["parameters"]

# Create a partial function which maps a vector of floats to parameters
convert_vector_p = partial(utils.convert_vector, model=config["model"], fix_params=config["fix_params"], **pars)


# Choose the orbital model
orb = orbit.models[config["model"]](**pars, obs_dates=dates)


def get_orbit_SB1(p):
    K, e, omega, P, T0, gamma, amp_f, l_f = convert_vector_p(p)

    if K < 0.0 or e < 0.0 or e > 1.0 or P < 0.0 or omega < -180 or omega > 520 or amp_f < 0.0 or l_f < 0.0:
        return -np.inf

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
    q, K, e, omega, P, T0, gamma, amp_f, l_f, amp_g, l_g = convert_vector_p(p)

    if q < 0.0 or q > 1.0 or K < 0.0 or e < 0.0 or e > 1.0 or P < 0.0 or omega < -180 or omega > 520 or amp_f < 0.0 or l_f < 0.0:
        return -np.inf

    # Update the orbit
    orb.q = q
    orb.K = K
    orb.e = e
    orb.omega = omega
    orb.P = P
    orb.T0 = T0
    orb.gamma = gamma

    # predict velocities for each epoch
    vAs, vBs = orb.get_component_velocities()
    vAs_fine, vBs_fine = orb.get_component_velocities(dates_fine)

    return (vAs, vAs_fine, vBs, vBs_fine)

def get_orbit_ST3(p):
    raise NotImplementedError


# Read config, data, and samples. Create a set of finely spaced dates, and for each (independent) sample, draw points and plot an orbit.

flatchain = np.load("flatchain.npy")[args.burn::args.thin]
indexes = np.random.choice(np.arange(len(flatchain)), size=args.draws)
flatchain = flatchain[indexes]

fig, ax = plt.subplots(figsize=(8,5))
ax.axhline(pars["gamma"], color="0.4", ls="-.")
if config["model"] == "SB1":
    for p in flatchain:
        vAs, vAs_fine = get_orbit_SB1(p)
        ax.plot(dates_fine, vAs_fine, color="0.4", lw=0.5)
        ax.plot(dates, vAs, "o", color="0.4")
elif config["model"] == "SB2":
    for p in flatchain:
        vAs, vAs_fine, vBs, vBs_fine = get_orbit_SB2(p)
        ax.plot(dates_fine, vAs_fine, color="b", lw=0.5, alpha=0.3)
        ax.plot(dates_fine, vBs_fine, color="g", lw=0.5, alpha=0.3)
        ax.plot(dates, vAs, ".", color="b")
        ax.plot(dates, vBs, ".", color="g")

else:
    print("model not implemented yet")

ax.set_xlabel("date [day]")
ax.set_ylabel(r"$v$ [km/s]")
fig.savefig("orbits.png")
