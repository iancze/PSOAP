import pytest

import numpy as np
from psoap import orbit_astrometry
from psoap import constants as C
import matplotlib.pyplot as plt

# Create plots of all of the orbits
from astropy.io import ascii

# Load the Tokovinin data sets for radial velocity and astrometry
rv_data = ascii.read(C.PSOAP_dir + "data/41Dra/rv.dat")

ind_A = (rv_data["comp"] == "a")
ind_B = (rv_data["comp"] == "b")

rv_jds_A = rv_data["JD"][ind_A]
vAs_data = rv_data["RV"][ind_A]
vAs_err = rv_data["err"][ind_A]

rv_jds_B = rv_data["JD"][ind_B]
vBs_data = rv_data["RV"][ind_B]
vBs_err = rv_data["err"][ind_B]

# Sort to separate vA and vB from each other
astro_data = ascii.read(C.PSOAP_dir + "data/41Dra/astro.dat")

rho_data = astro_data["rho"]
rho_err = 0.003

theta_data = astro_data["theta"]
theta_err = 0.1

astro_jds = astro_data["JD"]

dpc = 44.6 # pc

# Orbital elements for 41 Dra
a = 0.0706 * dpc # [AU]
e = 0.9754
i = 49.7 # [deg]
omega = 127.31 # [deg]
Omega = 1.9 # [deg]
T0 = 2449571.037 # [Julian Date]
M_2 = 1.20 # [M_sun]
M_tot = 1.28 + M_2 # [M_sun]
gamma = 5.76 # [km/s]


# Pick a span of dates
dates = np.linspace(2446630, 2452010, num=3000) # [day]


# Initialize the orbit
orb = orbit_astrometry.Binary(a, e, i, omega, Omega, T0, M_tot, M_2, gamma, obs_dates=dates)

vAs, vBs, XY_As, XY_Bs, XY_ABs, xy_As, xy_Bs, xy_ABs = orb.get_full_orbit()

vAs, vBs, rho_ABs, theta_ABs = orb.get_orbit()

# Convert to sky coordinates, using distance
alpha_dec_As = XY_As/dpc # [arcsec]
alpha_dec_Bs = XY_Bs/dpc # [arcsec]
alpha_dec_ABs = XY_ABs/dpc # [arcsec]
rho_ABs = rho_ABs/dpc # [arcsec]

# Plot the Orbits
fig, ax = plt.subplots(nrows=1, figsize=(5,5))

ax.plot(alpha_dec_ABs[:,1], alpha_dec_ABs[:,0])
ax.plot(0,0, "*k", ms=10)
ax.set_xlabel(r"$\Delta \alpha$ mas")
ax.set_ylabel(r"$\Delta \delta$ mas")
# Flip the X axis for RA.
ax.set_xlim(0.080, -0.080) # reverse to put RA increasing to left (east)
ax.set_ylim(-0.080, 0.080)
fig.savefig("plots/orbit_B_rel_A.png")

# Now plot A and B together
fig, ax = plt.subplots(nrows=1, figsize=(5,5))
ax.plot(alpha_dec_As[:,1], alpha_dec_As[:,0], color="b")
ax.plot(alpha_dec_Bs[:,1], alpha_dec_Bs[:,0], color="g")
ax.plot(0,0, "ok", ms=10)
ax.set_xlabel(r"$\Delta \alpha$ mas")
ax.set_ylabel(r"$\Delta \delta$ mas")
ax.set_xlim(0.080, -0.080) # reverse to put RA increasing to left (east)
ax.set_ylim(-0.080, 0.080)

# Flip the X axis for RA.
fig.savefig("plots/orbit_AB.png")


# Plot the orbits in the plane
# Plot the Orbits
fig, ax = plt.subplots(nrows=1, figsize=(5,5))

ax.plot(xy_ABs[:,0], xy_ABs[:,1])
ax.plot(0,0, "*k", ms=10)
ax.set_xlabel(r"$X$ [AU]")
ax.set_ylabel(r"$Y$ [AU]")
# ax.set_xlim(80, -80) # reverse to put RA increasing to left (east)
# ax.set_ylim(-40, 120)

# Flip the X axis for RA.
fig.savefig("plots/orbit_B_rel_A_plane.png")



fig, ax = plt.subplots(nrows=1, figsize=(5,5))

ax.plot(xy_As[:,0], xy_As[:,1], color="b")
ax.plot(xy_Bs[:,0], xy_Bs[:,1], color="g")

ax.plot(0,0, "ko", ms=10)
ax.set_xlabel(r"$X$ [AU]")
ax.set_ylabel(r"$Y$ [AU]")
# ax.set_xlim(80, -80) # reverse to put RA increasing to left (east)
# ax.set_ylim(-40, 120)

# Flip the X axis for RA.
fig.savefig("plots/orbit_AB_plane.png")


# Plot velocities, rho, and theta as function of time
fig, ax = plt.subplots(nrows=4, sharex=True, figsize=(12,8))

ax[0].plot(dates, vAs)
ax[0].errorbar(rv_jds_A, vAs_data, yerr=vAs_err, ls="")
ax[0].plot(rv_jds_A, vAs_data, "k.")
ax[0].set_ylabel(r"$v_A$ km/s")

ax[1].plot(dates, vBs)
ax[1].errorbar(rv_jds_B, vBs_data, yerr=vBs_err, ls="")
ax[1].plot(rv_jds_B, vBs_data, "k.")
ax[1].set_ylabel(r"$v_B$ km/s")

ax[2].plot(dates, rho_ABs)
ax[2].errorbar(astro_jds, rho_data, yerr=rho_err, ls="")
ax[2].plot(astro_jds, rho_data, "k.")
ax[2].set_ylabel(r"$\rho_\mathrm{AB}$ [mas]")

ax[3].plot(dates, theta_ABs)
ax[3].errorbar(astro_jds, theta_data, yerr=theta_err, ls="")
ax[3].plot(astro_jds, theta_data, "k.")
ax[3].set_ylabel(r"$\theta$ [deg]")

ax[-1].set_xlabel("date")
fig.savefig("plots/orbit_vel_rho_theta.png", dpi=400)

# Plot the phase-folded RVs
fig, ax = plt.subplots(nrows=1, figsize=(8,8))

P = orb.P
T0 = orb.T0

print("P", P)
print("K_A", orb.K)
print("q", orb.q)


phase = ((dates - T0) % (2*P)) / P

rv_phase_A = ((rv_jds_A - T0) % P) / P
rv_phase_B = ((rv_jds_B - T0) % P) / P

# sort according to phase
ind = np.argsort(phase)
ax.plot(phase[ind], vAs[ind], color="k")
ax.plot(phase[ind], vBs[ind], color="r", ls="--")

ax.plot(rv_phase_A, vAs_data, "k.")

ax.plot(rv_phase_B, vBs_data, "r.")

ax.set_xlabel("phase")
ax.set_ylabel(r"$v$ [km/s]")
ax.set_xlim(-0.2, 1.2)
fig.savefig("plots/orbit_rv_phase.png", dpi=300)


# remake a zoomed version of this figure to compare to the plot in Tokovinin.

fig, ax = plt.subplots(nrows=1, figsize=(8,8))

ax.plot(phase[ind], vAs[ind], color="k")
ax.plot(phase[ind], vBs[ind], color="r", ls="--")

indA = rv_phase_A < 0.05
indB = rv_phase_B < 0.05

# Shift the points so they show up on our plot
rv_phase_A[indA] = 1.0 + rv_phase_A[indA]
rv_phase_B[indB] = 1.0 + rv_phase_B[indB]

ax.plot(rv_phase_A, vAs_data, "k.")
ax.plot(rv_phase_B, vBs_data, "r.")

ax.set_xlabel("phase")
ax.set_ylabel(r"$v$ [km/s]")

ax.set_xlim(0.98, 1.03)

fig.savefig("plots/orbit_rv_phase_zoom.png", dpi=300)
