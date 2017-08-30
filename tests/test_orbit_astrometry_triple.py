import pytest

import numpy as np
from psoap import orbit_astrometry
from psoap import constants as C
import matplotlib.pyplot as plt
import matplotlib

# Create plots of all of the orbits

plot_dir = "plots/triple/"


dpc = 388 # pc

# Orbital elements for GW Ori
# a_in = 1.2 # [AU]
# e_in = 0.2
# i_in = 135.0 # [deg]
# omega_in = 45.0 # omega_1
# Omega_in = 30. # [deg]
# T0_in = 2450000.0 # [Julian Date]
#
# a_out = 8.0 # [AU]
# e_out = 0.2
# i_out = 135.0 # [deg]
# omega_out = 45.0 # omega_1
# Omega_out = 30. # [deg]
# T0_out = 2450000.0 # [Julian Date]
#
# M_1 = 3.0
# M_2 = 1.50 # [M_sun]
# M_3 = 1.0 # M_sun
#
# gamma = 27.0 # [km/s]

a_in = 10**(0.127) # [AU]
e_in = 0.074
i_in = 152.35 # [deg]
omega_in = 200.8 # [deg]
Omega_in = 275.8 # [deg]
T0_in = 2451853.6 # [Julian Date]

a_out = 10**(0.974) # [AU]
e_out = 0.19 #
i_out = 149.0 # [deg]
omega_out = 305.8 # [deg]
Omega_out = 282.0 # [deg]
T0_out = 2453855 # [Julian Date]
M_1 = 3.65 # [M_sun]
M_2 = 1.844 # [M_sun]
M_3 = 0.84 # [M_sun]
gamma = 26.29 # [km/s]

P_in = np.sqrt(4 * np.pi**2 / (C.G * (M_1 + M_2) * C.M_sun) * (a_in * C.AU)**3) / (24 * 3600) # [day]
P_out = np.sqrt(4 * np.pi**2 / (C.G * (M_1 + M_2 + M_3) * C.M_sun) * (a_out * C.AU)**3) / (24 * 3600) # [day]

# Pick a span of dates for one period
dates_in = np.linspace(T0_in, T0_in + P_in, num=600)
dates_out = np.linspace(T0_out, T0_out + P_out, num=600)

# Initialize the orbit
orb = orbit_astrometry.Triple(a_in, e_in, i_in, omega_in, Omega_in, T0_in, a_out, e_out, i_out, omega_out, Omega_out, T0_out, M_1, M_2, M_3, gamma, obs_dates=dates_out)


# Get the quantities to plot the outer orbit over one period first
full_dict = orb.get_full_orbit()

vAs, vBs, vCs, XYZ_ABs, XYZ_Cs, xy_ABs, xy_Cs = [full_dict[key] for key in ("vAs", "vBs", "vCs", "XYZ_ABs", "XYZ_Cs", "xy_ABs", "xy_Cs")]

# Convert to sky coordinates, using distance
alpha_dec_ABs = XYZ_ABs/dpc # [arcsec]
alpha_dec_Cs = XYZ_Cs/dpc # [arcsec]

# peri_A = orb.get_periastron_A()/dpc
# peri_B = orb.get_periastron_B()/dpc
# peri_BA = orb.get_periastron_BA()/dpc
#
# asc_A = orb.get_node_A()/dpc
# asc_B = orb.get_node_B()/dpc
# asc_BA = orb.get_node_BA()/dpc
#
# Since we are plotting vs one date, we need to plot the dots using a color scale so we can figure them out along the orbit.

# Set a colorscale for the lnprobs
cmap_primary = matplotlib.cm.get_cmap("Blues")
cmap_secondary = matplotlib.cm.get_cmap("Oranges")

norm = matplotlib.colors.Normalize(vmin=np.min(dates_out), vmax=np.max(dates_out))

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


# Make a series of astrometric plots from different angles.

# Now plot AB and C together, viewed from the Z axis
fig, ax = plt.subplots(nrows=1, figsize=(5,5))
ax.plot(0,0, "ok", ms=2)
plot_points(ax, dates_out, alpha_dec_ABs[:,0], alpha_dec_ABs[:,1], True)
plot_points(ax, dates_out, alpha_dec_Cs[:,0], alpha_dec_Cs[:,1], False)
# ax.plot(peri_A[0], peri_A[1], "ko", ms=3)
# ax.plot(peri_B[0], peri_B[1], "ko", ms=3)
# ax.plot(asc_A[0], asc_A[1], "^", color="C0", ms=3)
# ax.plot(asc_B[0], asc_B[1], "^", color="C1", ms=3)
ax.set_xlabel(r"$\Delta \delta$ mas")
ax.set_ylabel(r"$\Delta \alpha \cos \delta$ mas")
ax.set_aspect("equal", "datalim")

fig.subplots_adjust(left=0.15, right=0.85, bottom=0.15, top=0.85)
# Plot A and B together, viewed from the observer (along -Z axis).
fig.savefig(plot_dir + "orbit_AB_C_Z.png")


# Now plot A and B together, viewed from the X axis
# This means Y will form the "X" axis, or North
# And Z will form the Y axis, or towards observer
fig, ax = plt.subplots(nrows=1, figsize=(5,5))
ax.plot(0,0, "ok", ms=2)
plot_points(ax, dates_out, alpha_dec_ABs[:,1], alpha_dec_ABs[:,2], True)
plot_points(ax, dates_out, alpha_dec_Cs[:,1], alpha_dec_Cs[:,2], False)
# ax.plot(peri_A[1], peri_A[2], "ko", ms=3)
# ax.plot(peri_B[1], peri_B[2], "ko", ms=3)
# ax.plot(asc_A[1], asc_A[2], "^", color="C0", ms=3)
# ax.plot(asc_B[1], asc_B[2], "^", color="C1", ms=3)
ax.set_xlabel(r"$\Delta \alpha \cos \delta$ mas")
ax.set_ylabel(r"$\Delta Z$ mas (towards observer)")
ax.axhline(0, ls=":", color="k")
ax.set_aspect("equal", "datalim")
fig.savefig(plot_dir + "orbit_AB_C_X.png")

# Now plot A and B together, viewed from the Y axis
# This means Z will form the "X" axis, or towards the observer
# And X will form the Y axis, or East
fig, ax = plt.subplots(nrows=1, figsize=(5,5))
ax.plot(0,0, "ok", ms=2)
plot_points(ax, dates_out, alpha_dec_ABs[:,2], alpha_dec_ABs[:,0], True)
plot_points(ax, dates_out, alpha_dec_Cs[:,2], alpha_dec_Cs[:,0], False)
# ax.plot(peri_A[2], peri_A[0], "ko", ms=3)
# ax.plot(peri_B[2], peri_B[0], "ko", ms=3)
# ax.plot(asc_A[2], asc_A[0], "^", color="C0", ms=3)
# ax.plot(asc_B[2], asc_B[0], "^", color="C1", ms=3)
ax.axvline(0, ls=":", color="k")
ax.set_xlabel(r"$\Delta Z$ mas (towards observer)")
ax.set_ylabel(r"$\Delta \delta$ mas")
ax.set_aspect("equal", "datalim")
fig.savefig(plot_dir + "orbit_AB_C_Y.png")


# Plot velocities as function of time for one period
fig, ax = plt.subplots(nrows=3, sharex=True, figsize=(8,8))

ax[0].plot(dates_out, vAs)
ax[0].set_ylabel(r"$v_A$ km/s")

ax[1].plot(dates_out, vBs)
ax[1].set_ylabel(r"$v_B$ km/s")

ax[2].plot(dates_out, vCs)
ax[2].set_ylabel(r"$v_C$ km/s")

ax[-1].set_xlabel("date")
fig.savefig(plot_dir + "orbit_out_vel.png", dpi=400)


# Plot the outer orbits in the plane
fig, ax = plt.subplots(nrows=1, figsize=(5,5))
plot_points(ax, dates_out, xy_ABs[:,0], xy_ABs[:,1], True)
plot_points(ax, dates_out, xy_Cs[:,0], xy_Cs[:,1], False)
ax.plot(0,0, "ko", ms=10)
ax.set_xlabel(r"$X$ [AU]")
ax.set_ylabel(r"$Y$ [AU]")
ax.set_aspect("equal", "datalim")
fig.savefig(plot_dir + "orbit_AB_C_plane.png")

# Make the same plots for the inner orbit
full_dict = orb.get_full_orbit(dates_in)

XYZ_As, XYZ_Bs, xy_As, xy_Bs = [full_dict[key] for key in ("XYZ_A_locs", "XYZ_B_locs", "xy_A_locs", "xy_B_locs")]

# Convert to sky coordinates, using distance
alpha_dec_As = XYZ_As/dpc # [arcsec]
alpha_dec_Bs = XYZ_Bs/dpc # [arcsec]

norm = matplotlib.colors.Normalize(vmin=np.min(dates_in), vmax=np.max(dates_in))

# Now plot AB and C together, viewed from the Z axis
fig, ax = plt.subplots(nrows=1, figsize=(5,5))
ax.plot(0,0, "ok", ms=2)
plot_points(ax, dates_in, alpha_dec_As[:,0], alpha_dec_As[:,1], True)
plot_points(ax, dates_in, alpha_dec_Bs[:,0], alpha_dec_Bs[:,1], False)
# ax.plot(peri_A[0], peri_A[1], "ko", ms=3)
# ax.plot(peri_B[0], peri_B[1], "ko", ms=3)
# ax.plot(asc_A[0], asc_A[1], "^", color="C0", ms=3)
# ax.plot(asc_B[0], asc_B[1], "^", color="C1", ms=3)
ax.set_xlabel(r"$\Delta \delta$ mas")
ax.set_ylabel(r"$\Delta \alpha \cos \delta$ mas")
ax.set_aspect("equal", "datalim")

fig.subplots_adjust(left=0.2, right=0.8, bottom=0.2, top=0.8)
# Plot A and B together, viewed from the observer (along -Z axis).
fig.savefig(plot_dir + "orbit_A_B_Z.png")


# Now plot A and B together, viewed from the X axis
# This means Y will form the "X" axis, or North
# And Z will form the Y axis, or towards observer
fig, ax = plt.subplots(nrows=1, figsize=(5,5))
ax.plot(0,0, "ok", ms=2)
plot_points(ax, dates_in, alpha_dec_As[:,1], alpha_dec_As[:,2], True)
plot_points(ax, dates_in, alpha_dec_Bs[:,1], alpha_dec_Bs[:,2], False)
# ax.plot(peri_A[1], peri_A[2], "ko", ms=3)
# ax.plot(peri_B[1], peri_B[2], "ko", ms=3)
# ax.plot(asc_A[1], asc_A[2], "^", color="C0", ms=3)
# ax.plot(asc_B[1], asc_B[2], "^", color="C1", ms=3)
ax.set_xlabel(r"$\Delta \alpha \cos \delta$ mas")
ax.set_ylabel(r"$\Delta Z$ mas (towards observer)")
ax.axhline(0, ls=":", color="k")
ax.set_aspect("equal", "datalim")
fig.subplots_adjust(left=0.2, right=0.8, bottom=0.2, top=0.8)
fig.savefig(plot_dir + "orbit_A_B_X.png")

# Now plot A and B together, viewed from the Y axis
# This means Z will form the "X" axis, or towards the observer
# And X will form the Y axis, or East
fig, ax = plt.subplots(nrows=1, figsize=(5,5))
ax.plot(0,0, "ok", ms=2)
plot_points(ax, dates_in, alpha_dec_As[:,2], alpha_dec_As[:,0], True)
plot_points(ax, dates_in, alpha_dec_Bs[:,2], alpha_dec_Bs[:,0], False)
# ax.plot(peri_A[2], peri_A[0], "ko", ms=3)
# ax.plot(peri_B[2], peri_B[0], "ko", ms=3)
# ax.plot(asc_A[2], asc_A[0], "^", color="C0", ms=3)
# ax.plot(asc_B[2], asc_B[0], "^", color="C1", ms=3)
ax.axvline(0, ls=":", color="k")
ax.set_xlabel(r"$\Delta Z$ mas (towards observer)")
ax.set_ylabel(r"$\Delta \delta$ mas")
ax.set_aspect("equal", "datalim")
fig.subplots_adjust(left=0.2, right=0.8, bottom=0.2, top=0.8)
fig.savefig(plot_dir + "orbit_A_B_Y.png")


# Now plot the full 3D orbit over the long period.

norm = matplotlib.colors.Normalize(vmin=np.min(dates_out), vmax=np.max(dates_out))
full_dict = orb.get_full_orbit(dates_out)

XYZ_As, XYZ_Bs, XYZ_Cs = [full_dict[key] for key in ("XYZ_As", "XYZ_Bs", "XYZ_Cs")]

# Convert to sky coordinates, using distance
alpha_dec_As = XYZ_As/dpc # [arcsec]
alpha_dec_Bs = XYZ_Bs/dpc # [arcsec]
alpha_dec_Cs = XYZ_Cs/dpc # [arcsec]

fig, ax = plt.subplots(nrows=1, figsize=(5,5))
ax.plot(0,0, "ok", ms=2)
plot_points(ax, dates_out, alpha_dec_Bs[:,0], alpha_dec_Bs[:,1], False)
plot_points(ax, dates_out, alpha_dec_As[:,0], alpha_dec_As[:,1], True)
plot_points(ax, dates_out, alpha_dec_Cs[:,0], alpha_dec_Cs[:,1], False)
# ax.plot(peri_A[0], peri_A[1], "ko", ms=3)
# ax.plot(peri_B[0], peri_B[1], "ko", ms=3)
# ax.plot(asc_A[0], asc_A[1], "^", color="C0", ms=3)
# ax.plot(asc_B[0], asc_B[1], "^", color="C1", ms=3)
ax.set_xlabel(r"$\Delta \delta$ mas")
ax.set_ylabel(r"$\Delta \alpha \cos \delta$ mas")
ax.set_aspect("equal", "datalim")

fig.subplots_adjust(left=0.2, right=0.8, bottom=0.2, top=0.8)
fig.savefig(plot_dir + "orbit_A_B_C_Z.png")


# Now plot A and B together, viewed from the X axis
# This means Y will form the "X" axis, or North
# And Z will form the Y axis, or towards observer
fig, ax = plt.subplots(nrows=1, figsize=(5,5))
ax.plot(0,0, "ok", ms=2)
plot_points(ax, dates_out, alpha_dec_Bs[:,1], alpha_dec_Bs[:,2], False)
plot_points(ax, dates_out, alpha_dec_As[:,1], alpha_dec_As[:,2], True)
plot_points(ax, dates_out, alpha_dec_Cs[:,1], alpha_dec_Cs[:,2], False)
# ax.plot(peri_A[1], peri_A[2], "ko", ms=3)
# ax.plot(peri_B[1], peri_B[2], "ko", ms=3)
# ax.plot(asc_A[1], asc_A[2], "^", color="C0", ms=3)
# ax.plot(asc_B[1], asc_B[2], "^", color="C1", ms=3)
ax.set_xlabel(r"$\Delta \alpha \cos \delta$ mas")
ax.set_ylabel(r"$\Delta Z$ mas (towards observer)")
ax.axhline(0, ls=":", color="k")
ax.set_aspect("equal", "datalim")
fig.subplots_adjust(left=0.2, right=0.8, bottom=0.2, top=0.8)
fig.savefig(plot_dir + "orbit_A_B_C_X.png")

# Now plot A and B together, viewed from the Y axis
# This means Z will form the "X" axis, or towards the observer
# And X will form the Y axis, or East
fig, ax = plt.subplots(nrows=1, figsize=(5,5))
ax.plot(0,0, "ok", ms=2)
plot_points(ax, dates_out, alpha_dec_Bs[:,2], alpha_dec_Bs[:,0], False)
plot_points(ax, dates_out, alpha_dec_As[:,2], alpha_dec_As[:,0], True)
plot_points(ax, dates_out, alpha_dec_Cs[:,2], alpha_dec_Cs[:,0], False)
# ax.plot(peri_A[2], peri_A[0], "ko", ms=3)
# ax.plot(peri_B[2], peri_B[0], "ko", ms=3)
# ax.plot(asc_A[2], asc_A[0], "^", color="C0", ms=3)
# ax.plot(asc_B[2], asc_B[0], "^", color="C1", ms=3)
ax.axvline(0, ls=":", color="k")
ax.set_xlabel(r"$\Delta Z$ mas (towards observer)")
ax.set_ylabel(r"$\Delta \delta$ mas")
ax.set_aspect("equal", "datalim")
fig.subplots_adjust(left=0.2, right=0.8, bottom=0.2, top=0.8)
fig.savefig(plot_dir + "orbit_A_B_C_Y.png")
