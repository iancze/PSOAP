import pytest

import os
import pkg_resources
import numpy as np
from psoap import orbit_astrometry
from psoap import constants as C
import matplotlib.pyplot as plt
from astropy.time import Time
import matplotlib


# Create plots of all of the orbits
from astropy.io import ascii


# Create plots of all of the orbits
# If it doesn't already exist, create a "plots/basic/" directory
outdir = "tests/plots/Gl417BC/"
if not os.path.exists(outdir):
    print("Creating ", outdir)
    os.makedirs(outdir)

# Load the Dupuy et al. data sets for astrometry
data_fname = pkg_resources.resource_filename("psoap", "data/Gl417BC/data.txt")
data = ascii.read(data_fname, format="csv")

# convert UT date to JD
astro_dates = Time(data["date"])
astro_dates.format = 'jd'
astro_dates = astro_dates.value


theta_data = data["PA"]
theta_err = data["PA_err"]

rho_data = data["rho"]
rho_err = data["rho_err"]

def test_data():
    # Make a plot of the astrometric data on the sky
    fig, ax = plt.subplots(nrows=1)

    xs = rho_data * np.cos(theta_data * np.pi/180)
    ys = rho_data * np.sin(theta_data * np.pi/180)
    ax.plot(xs, ys, ".")
    ax.set_xlabel("North")
    ax.set_ylabel("East")
    ax.plot(0,0, "k*")
    ax.set_aspect("equal", "datalim")
    fig.savefig(outdir + "data_astro.png")
    plt.close('all')

dpc = 21.93 # pc

# Orbital elements
a = 0.130 * dpc # [AU]
e = 0.106
i = 102.9 # [deg]
# omega =  # omega_1
# omega = 0
omega_2 = 348
omega = omega_2 + 180

# omega_2 = omega + 180
# omega = omega_2 - 180# [deg] # we actua
Omega = 101.03 # [deg]
# Omega = 20 # [deg]
T0 = 2456664.5 # [Julian Date]
# M_2 = 1.20 # [M_sun]
M_2 = 0.04
M_tot = 0.0949 # [M_sun]
gamma = 0.0 # [km/s]

P = np.sqrt(4 * np.pi**2 / (C.G * M_tot * C.M_sun) * (a * C.AU)**3) / (24 * 3600) # [day]

# Pick a span of dates for the observations

# Pick a span of dates for one period
dates = np.linspace(T0, T0 + P, num=600)

# Initialize the orbit
orb = orbit_astrometry.Binary(a, e, i, omega, Omega, T0, M_tot, M_2, gamma, obs_dates=dates)

full_dict = orb.get_full_orbit()

vAs, vBs, XYZ_As, XYZ_Bs, XYZ_ABs, xy_As, xy_Bs, xy_ABs = [full_dict[key] for key in ("vAs", "vBs", "XYZ_As", "XYZ_Bs", "XYZ_ABs", "xy_As", "xy_Bs", "xy_ABs")]

polar_dict = orb.get_orbit()

vAs, vBs, rho_ABs, theta_ABs = [polar_dict[key] for key in ("vAs", "vBs", "rhos", "thetas")]

# Convert to sky coordinates, using distance
alpha_dec_As = XYZ_As/dpc # [arcsec]
alpha_dec_Bs = XYZ_Bs/dpc # [arcsec]
alpha_dec_ABs = XYZ_ABs/dpc # [arcsec]
rho_ABs = rho_ABs/dpc # [arcsec]


peri_A = orb._get_periastron_A()/dpc
peri_B = orb._get_periastron_B()/dpc
peri_BA = orb._get_periastron_BA()/dpc

asc_A = orb._get_node_A()/dpc
asc_B = orb._get_node_B()/dpc
asc_BA = orb._get_node_BA()/dpc

# Since we are plotting vs one date, we need to plot the dots using a color scale so we can figure them out along the orbit.

# Set a colorscale for the lnprobs
cmap_primary = matplotlib.cm.get_cmap("Blues")
cmap_secondary = matplotlib.cm.get_cmap("Oranges")

norm = matplotlib.colors.Normalize(vmin=np.min(dates), vmax=np.max(dates))

# Determine colors based on the ending lnprob of each walker
def plot_points(ax, dates, xs, ys, primary):
    for date, x, y in zip(dates, xs, ys):
        if primary:
            c = cmap_primary(norm(date))
        else:
            c = cmap_secondary(norm(date))
        ax.plot(x, y, "o", color=c, mew=0.1, ms=3, mec="k")


# Then, we will make 3D plots of the orbit so that we can square with what we think is happening.

# The final crowning grace will be a 3D matplotlib plot of the orbital path.

def test_B_rel_A():
    # Plot the Orbits
    fig, ax = plt.subplots(nrows=1, figsize=(5,5))
    plot_points(ax, dates, alpha_dec_ABs[:,0], alpha_dec_ABs[:,1], False)
    ax.plot(0,0, "*k", ms=2)
    # ax.plot(peri_BA[0], peri_BA[1], "ko", ms=3)
    # ax.plot(asc_BA[0], asc_BA[1], "o", color="C2", ms=3)
    ax.set_xlabel(r"$\Delta \alpha \cos \delta$ mas")
    ax.set_ylabel(r"$\Delta \delta$ mas")
    ax.invert_xaxis()
    ax.set_aspect("equal", "datalim")
    fig.savefig(outdir + "orbit_B_rel_A.png")

def test_AB_Z():
    # Now plot A and B together, viewed from the -Z axis
    fig, ax = plt.subplots(nrows=1, figsize=(5,5))
    ax.plot(0,0, "ok", ms=2)
    plot_points(ax, dates, alpha_dec_As[:,0], alpha_dec_As[:,1], True)
    plot_points(ax, dates, alpha_dec_Bs[:,0], alpha_dec_Bs[:,1], False)
    # ax.plot(peri_A[0], peri_A[1], "ko", ms=3)
    # ax.plot(peri_B[0], peri_B[1], "ko", ms=3)
    # ax.plot(asc_A[0], asc_A[1], "^", color="C0", ms=3)
    # ax.plot(asc_B[0], asc_B[1], "^", color="C1", ms=3)
    ax.set_xlabel(r"$\Delta \alpha \cos \delta$ mas")
    ax.set_ylabel(r"$\Delta \delta$ mas")
    ax.set_aspect("equal", "datalim")
    ax.invert_xaxis()
    fig.subplots_adjust(left=0.15, right=0.85, bottom=0.15, top=0.85)
    # Plot A and B together, viewed from the observer's perspective
    fig.savefig(outdir + "orbit_AB_-Z.png")

def test_AB_X():
    # Now plot A and B together, viewed from the X axis
    # This means Y will form the "X" axis, or North
    # And Z will form the Y axis, or towards observer
    fig, ax = plt.subplots(nrows=1, figsize=(5,5))
    ax.plot(0,0, "ok", ms=2)
    plot_points(ax, dates, alpha_dec_As[:,1], alpha_dec_As[:,2], True)
    plot_points(ax, dates, alpha_dec_Bs[:,1], alpha_dec_Bs[:,2], False)
    ax.plot(peri_A[1], peri_A[2], "ko", ms=3)
    ax.plot(peri_B[1], peri_B[2], "ko", ms=3)
    ax.plot(asc_A[1], asc_A[2], "^", color="C0", ms=3)
    ax.plot(asc_B[1], asc_B[2], "^", color="C1", ms=3)
    ax.set_xlabel(r"$\Delta \alpha$ mas")
    ax.set_ylabel(r"$\Delta Z$ mas (towards observer)")
    ax.axhline(0, ls=":", color="k")
    ax.set_aspect("equal", "datalim")
    # ax.set_xlim(-0.02, 0.02)
    # ax.set_ylim(-0.02, 0.02)
    fig.savefig(outdir + "orbit_AB_X.png")

def test_AB_Y():
    # Now plot A and B together, viewed from the Y axis
    # This means Z will form the "X" axis, or towards the observer
    # And X will form the Y axis, or East
    fig, ax = plt.subplots(nrows=1, figsize=(5,5))
    ax.plot(0,0, "ok", ms=2)
    plot_points(ax, dates, alpha_dec_As[:,2], alpha_dec_As[:,0], True)
    plot_points(ax, dates, alpha_dec_Bs[:,2], alpha_dec_Bs[:,0], False)
    ax.plot(peri_A[2], peri_A[0], "ko", ms=3)
    ax.plot(peri_B[2], peri_B[0], "ko", ms=3)
    ax.plot(asc_A[2], asc_A[0], "^", color="C0", ms=3)
    ax.plot(asc_B[2], asc_B[0], "^", color="C1", ms=3)
    ax.axvline(0, ls=":", color="k")
    ax.set_xlabel(r"$\Delta Z$ mas (towards observer)")
    ax.set_ylabel(r"$\Delta \delta $ mas")
    ax.set_aspect("equal", "datalim")
    # ax.set_xlim(-0.02, 0.02)
    # ax.set_ylim(-0.02, 0.02)
    fig.savefig(outdir + "orbit_AB_Y.png")

def test_vel_rho_theta_one_period():
    # Plot velocities, rho, and theta as function of time for one period
    fig, ax = plt.subplots(nrows=4, sharex=True, figsize=(8,8))

    ax[0].plot(dates, vAs)
    # ax[0].errorbar(rv_jds_A, vAs_data, yerr=vAs_err, ls="")
    # ax[0].plot(rv_jds_A, vAs_data, "k.")
    ax[0].set_ylabel(r"$v_A$ km/s")

    ax[1].plot(dates, vBs)
    # ax[1].errorbar(rv_jds_B, vBs_data, yerr=vBs_err, ls="")
    # ax[1].plot(rv_jds_B, vBs_data, "k.")
    ax[1].set_ylabel(r"$v_B$ km/s")

    ax[2].plot(dates, rho_ABs)
    # ax[2].errorbar(astro_dates, rho_data, yerr=rho_err, ls="")
    # ax[2].plot(astro_dates, rho_data, "k.")
    ax[2].set_ylabel(r"$\rho_\mathrm{AB}$ [mas]")

    ax[3].plot(dates, theta_ABs)
    # ax[3].errorbar(astro_dates, theta_data, yerr=theta_err, ls="")
    # ax[3].plot(astro_dates, theta_data, "k.")
    ax[3].set_ylabel(r"$\theta$ [deg]")

    ax[-1].set_xlabel("date")
    fig.savefig(outdir + "orbit_vel_rho_theta_one_period.png", dpi=400)


# Now make a 3D Orbit and pop it up

def test_B_rel_A_plane():
    # Plot the orbits in the plane
    fig, ax = plt.subplots(nrows=1, figsize=(5,5))
    plot_points(ax, dates, xy_ABs[:,0], xy_ABs[:,1], False)
    ax.plot(0,0, "*k", ms=10)
    ax.set_xlabel(r"$X$ [AU]")
    ax.set_ylabel(r"$Y$ [AU]")
    ax.set_aspect("equal", "datalim")
    fig.savefig(outdir + "orbit_B_rel_A_plane.png")

def test_AB_plane():
    fig, ax = plt.subplots(nrows=1, figsize=(5,5))
    plot_points(ax, dates, xy_As[:,0], xy_As[:,1], True)
    plot_points(ax, dates, xy_Bs[:,0], xy_Bs[:,1], False)
    ax.plot(0,0, "ko", ms=10)
    ax.set_xlabel(r"$X$ [AU]")
    ax.set_ylabel(r"$Y$ [AU]")
    ax.set_aspect("equal", "datalim")
    fig.savefig(outdir + "orbit_AB_plane.png")
    plt.close('all')

# Redo this using a finer space series of dates spanning the full series of observations.

# Pick a span of dates for the observations
dates = np.linspace(2451940, 2456790, num=300) # [day]
orb = orbit_astrometry.Binary(a, e, i, omega, Omega, T0, M_tot, M_2, gamma, obs_dates=dates)

full_dict = orb.get_full_orbit()

vAs, vBs, XYZ_As, XYZ_Bs, XYZ_ABs, xy_As, xy_Bs, xy_ABs = [full_dict[key] for key in ("vAs", "vBs", "XYZ_As", "XYZ_Bs", "XYZ_ABs", "xy_As", "xy_Bs", "xy_ABs")]

polar_dict = orb.get_orbit()

vAs, vBs, rho_ABs, theta_ABs = [polar_dict[key] for key in ("vAs", "vBs", "rhos", "thetas")]

# Convert to sky coordinates, using distance
alpha_dec_As = XYZ_As/dpc # [arcsec]
alpha_dec_Bs = XYZ_Bs/dpc # [arcsec]
alpha_dec_ABs = XYZ_ABs/dpc # [arcsec]
rho_ABs = rho_ABs/dpc # [arcsec]

def test_vel_rho_theta():
    # Plot velocities, rho, and theta as function of time
    fig, ax = plt.subplots(nrows=4, sharex=True, figsize=(12,8))

    ax[0].plot(dates, vAs)
    # ax[0].errorbar(rv_jds_A, vAs_data, yerr=vAs_err, ls="")
    # ax[0].plot(rv_jds_A, vAs_data, "k.")
    ax[0].set_ylabel(r"$v_A$ km/s")

    ax[1].plot(dates, vBs)
    # ax[1].errorbar(rv_jds_B, vBs_data, yerr=vBs_err, ls="")
    # ax[1].plot(rv_jds_B, vBs_data, "k.")
    ax[1].set_ylabel(r"$v_B$ km/s")

    print(rho_data)
    print(rho_err.data)

    ax[2].plot(dates, rho_ABs)
    ax[2].errorbar(astro_dates, 1e-3 * rho_data.data, yerr= 1e-3 * rho_err.data, ls="")
    ax[2].plot(astro_dates, 1e-3 *  rho_data, "k.")
    ax[2].set_ylabel(r"$\rho_\mathrm{AB}$ [mas]")

    ax[3].plot(dates, theta_ABs)
    ax[3].errorbar(astro_dates, theta_data.data, yerr=theta_err.data, ls="")
    ax[3].plot(astro_dates, theta_data, "k.")
    ax[3].set_ylabel(r"$\theta$ [deg]")

    ax[-1].set_xlabel("date")
    fig.savefig(outdir + "orbit_vel_rho_theta.png", dpi=400)
    plt.close('all')

plt.close('all')
