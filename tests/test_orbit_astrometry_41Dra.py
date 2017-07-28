import pytest

import numpy as np
from psoap import orbit_astrometry
from psoap import constants as C
import matplotlib.pyplot as plt

import matplotlib


# Create plots of all of the orbits
from astropy.io import ascii


plot_dir = "plots/41Dra/"

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

# Make a plot of the astrometric data on the sky
fig, ax = plt.subplots(nrows=1)

xs = rho_data * np.cos(theta_data * np.pi/180)
ys = rho_data * np.sin(theta_data * np.pi/180)
ax.plot(xs, ys, ".")
ax.set_xlabel("North")
ax.set_ylabel("East")
ax.plot(0,0, "k*")
ax.set_aspect("equal", "datalim")
fig.savefig(plot_dir + "data_astro.png")

dpc = 44.6 # pc

# Orbital elements for 41 Dra
a = 0.0706 * dpc # [AU]
e = 0.9754
i = 49.7 # [deg]
omega = 127.31 # omega_1
Omega = 1.9 + 180 # [deg]
T0 = 2449571.037 # [Julian Date]
M_2 = 1.20 # [M_sun]
M_tot = 1.28 + M_2 # [M_sun]
gamma = 5.76 # [km/s]

P = np.sqrt(4 * np.pi**2 / (C.G * M_tot * C.M_sun) * (a * C.AU)**3) / (24 * 3600) # [day]

# Pick a span of dates for the observations
dates_obs = np.linspace(2446630, 2452010, num=300) # [day]

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


peri_A = orb.get_periastron_A()/dpc
peri_B = orb.get_periastron_B()/dpc
peri_BA = orb.get_periastron_BA()/dpc

asc_A = orb.get_node_A()/dpc
asc_B = orb.get_node_B()/dpc
asc_BA = orb.get_node_BA()/dpc

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

# Plot the Orbits
fig, ax = plt.subplots(nrows=1, figsize=(5,5))
plot_points(ax, dates, alpha_dec_ABs[:,0], alpha_dec_ABs[:,1], False)
ax.plot(0,0, "*k", ms=2)
ax.plot(peri_BA[0], peri_BA[1], "ko", ms=3)
ax.plot(asc_BA[0], asc_BA[1], "o", color="C2", ms=3)
ax.set_xlabel(r"$\Delta \delta$ mas")
ax.set_ylabel(r"$\Delta \alpha \cos \delta $ mas")
ax.set_aspect("equal", "datalim")
fig.savefig(plot_dir + "orbit_B_rel_A.png")


# Make a series of astrometric plots from different angles.

# Now plot A and B together, viewed from the Z axis
fig, ax = plt.subplots(nrows=1, figsize=(5,5))
ax.plot(0,0, "ok", ms=2)
plot_points(ax, dates, alpha_dec_As[:,0], alpha_dec_As[:,1], True)
plot_points(ax, dates, alpha_dec_Bs[:,0], alpha_dec_Bs[:,1], False)
ax.plot(peri_A[0], peri_A[1], "ko", ms=3)
ax.plot(peri_B[0], peri_B[1], "ko", ms=3)
ax.plot(asc_A[0], asc_A[1], "^", color="C0", ms=3)
ax.plot(asc_B[0], asc_B[1], "^", color="C1", ms=3)
ax.set_xlabel(r"$\Delta \delta$ mas")
ax.set_ylabel(r"$\Delta \alpha \cos \delta$ mas")
ax.set_aspect("equal", "datalim")

fig.subplots_adjust(left=0.15, right=0.85, bottom=0.15, top=0.85)
# Plot A and B together, viewed from the observer (along -Z axis).
fig.savefig(plot_dir + "orbit_AB_Z.png")


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
ax.set_xlabel(r"$\Delta \alpha \cos \delta$ mas")
ax.set_ylabel(r"$\Delta Z$ mas (towards observer)")
ax.axhline(0, ls=":", color="k")
ax.set_aspect("equal", "datalim")
fig.savefig(plot_dir + "orbit_AB_X.png")

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
ax.set_ylabel(r"$\Delta \delta$ mas")
ax.set_aspect("equal", "datalim")
fig.savefig(plot_dir + "orbit_AB_Y.png")


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
# ax[2].errorbar(astro_jds, rho_data, yerr=rho_err, ls="")
# ax[2].plot(astro_jds, rho_data, "k.")
ax[2].set_ylabel(r"$\rho_\mathrm{AB}$ [mas]")

ax[3].plot(dates, theta_ABs)
# ax[3].errorbar(astro_jds, theta_data, yerr=theta_err, ls="")
# ax[3].plot(astro_jds, theta_data, "k.")
ax[3].set_ylabel(r"$\theta$ [deg]")

ax[-1].set_xlabel("date")
fig.savefig(plot_dir + "orbit_vel_rho_theta_one_period.png", dpi=400)


# Now make a 3D Orbit and pop it up


# Plot the orbits in the plane
fig, ax = plt.subplots(nrows=1, figsize=(5,5))
plot_points(ax, dates, xy_ABs[:,0], xy_ABs[:,1], False)
ax.plot(0,0, "*k", ms=10)
ax.set_xlabel(r"$X$ [AU]")
ax.set_ylabel(r"$Y$ [AU]")
ax.set_aspect("equal", "datalim")
fig.savefig(plot_dir + "orbit_B_rel_A_plane.png")


fig, ax = plt.subplots(nrows=1, figsize=(5,5))
plot_points(ax, dates, xy_As[:,0], xy_As[:,1], True)
plot_points(ax, dates, xy_Bs[:,0], xy_Bs[:,1], False)
ax.plot(0,0, "ko", ms=10)
ax.set_xlabel(r"$X$ [AU]")
ax.set_ylabel(r"$Y$ [AU]")
ax.set_aspect("equal", "datalim")
fig.savefig(plot_dir + "orbit_AB_plane.png")

# Redo this using a finer space series of dates spanning the full series of observations.

# Pick a span of dates for the observations
dates = np.linspace(2446630, 2452010, num=3000) # [day]
orb = orbit_astrometry.Binary(a, e, i, omega, Omega, T0, M_tot, M_2, gamma, obs_dates=dates)

polar_dict = orb.get_orbit()
vAs, vBs, rho_ABs, theta_ABs = [polar_dict[key] for key in ("vAs", "vBs", "rhos", "thetas")]
# Convert to sky coordinates, using distance
rho_ABs = rho_ABs/dpc # [arcsec]

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
fig.savefig(plot_dir + "orbit_vel_rho_theta.png", dpi=400)

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
fig.savefig(plot_dir + "orbit_rv_phase.png", dpi=300)


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

fig.savefig(plot_dir + "orbit_rv_phase_zoom.png", dpi=300)
