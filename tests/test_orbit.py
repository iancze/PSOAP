import pytest

import numpy as np
from psoap import orbit
import matplotlib.pyplot as plt


# Create plots of all of the orbits

# Binary orbit parameters
dates = np.linspace(0, 20, num=200)

q = 0.4
K = 10.0
e = 0.1
omega = 20.0
P = 5.0
T0 = 2.0
gamma = 10.0

# Triple orbit parameters
q_in = 0.4
K_in = 10.0
e_in = 0.1
omega_in = 20.0
P_in = 3.0
T0_in = 2.0
q_out = 0.2
K_out = 4.0
e_out = 0.2
omega_out = 80.0
P_out = 10.0
T0_out = 4.0

def test_SB1():
    orb = orbit.SB1(K, e, omega, P, T0, gamma, dates)

    vels = orb.get_velocities()

    fig,ax = plt.subplots(nrows=1)
    ax.axhline(gamma, color="0.5", ls=":")
    ax.plot(dates, vels[0])
    ax.set_xlabel("JD")
    ax.set_ylabel(r"$v_A\,\mathrm{km/s}$")
    fig.savefig("plots/SB1.png", dpi=300)

def test_SB2():
    orb = orbit.SB2(q, K, e, omega, P, T0, gamma, dates)

    vels = orb.get_velocities()

    fig,ax = plt.subplots(nrows=1)
    ax.axhline(gamma, color="0.5", ls=":")
    ax.plot(dates, vels[0])
    ax.plot(dates, vels[1])
    ax.set_xlabel("JD")
    ax.set_ylabel(r"$v\,\mathrm{km/s}$")
    fig.savefig("plots/SB2.png", dpi=300)

def test_ST1():
    orb = orbit.ST1(K_in, e_in, omega_in, P_in, T0_in, K_out, e_out, omega_out, P_out, T0_out, gamma, dates)

    vels = orb.get_velocities()

    fig,ax = plt.subplots(nrows=1)
    ax.axhline(gamma, color="0.5", ls=":")
    ax.plot(dates, vels[0])
    ax.set_xlabel("JD")
    ax.set_ylabel(r"$v\,\mathrm{km/s}$")
    fig.savefig("plots/ST1.png", dpi=300)

def test_ST2():
    orb = orbit.ST2(q_in, K_in, e_in, omega_in, P_in, T0_in, K_out, e_out, omega_out, P_out, T0_out, gamma, dates)

    vels = orb.get_velocities()

    fig,ax = plt.subplots(nrows=1)
    ax.axhline(gamma, color="0.5", ls=":")
    ax.plot(dates, vels[0])
    ax.plot(dates, vels[1])
    ax.set_xlabel("JD")
    ax.set_ylabel(r"$v\,\mathrm{km/s}$")
    fig.savefig("plots/ST2.png", dpi=300)


def test_ST3():
    orb = orbit.ST3(q_in, K_in, e_in, omega_in, P_in, T0_in, q_out, K_out, e_out, omega_out, P_out, T0_out, gamma, dates)

    vels = orb.get_velocities()

    fig,ax = plt.subplots(nrows=1)
    ax.axhline(gamma, color="0.5", ls=":")
    ax.plot(dates, vels[0])
    ax.plot(dates, vels[1])
    ax.plot(dates, vels[2])
    ax.set_xlabel("JD")
    ax.set_ylabel(r"$v\,\mathrm{km/s}$")
    fig.savefig("plots/ST3.png", dpi=300)
