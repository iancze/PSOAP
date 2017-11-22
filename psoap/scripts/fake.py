import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

from ..data import redshift, Chunk
from .. import matrix_functions
from .. import covariance
from .. import orbit

# Fake SB1
# Specify orbital parameters and make a sanity plot

K = 5.0 # km/s
e = 0.2 #
omega = 10.0 # deg
P = 10.0 # days
T0 = 0.0 # epoch
gamma = 5.0 # km/s

n_epochs = 7
obs_dates = np.linspace(5, 30, num=n_epochs)

sb1 = orbit.SB1(K, e, omega, P, T0, gamma, obs_dates)

vAs = sb1.get_component_velocities()

dates_fine = np.linspace(0, 35, num=200)
vA_fine = sb1.get_component_velocities(dates_fine)

fig, ax = plt.subplots()
ax.plot(dates_fine, vA_fine, "b")
ax.plot(sb1.obs_dates, vAs, "bo")

ax.axhline(gamma, ls="-.", color="0.5")
ax.set_xlabel(r"$t$ [days]")
ax.set_ylabel(r"$v_A$ [km $\mathrm{s}^{-1}$]")

fig.subplots_adjust(left=0.14, right=0.86, bottom=0.24)
fig.savefig("SB1/orbit.png")

# Load the fake primary spectra we prepared
wl, fl_f = np.load("primary_wl_fl.npy")

n_pix = len(wl)

# Create fake wavelengths with Doppler shifts by apply these to the master wl
wls_f = np.empty((n_epochs, n_pix))

for i in range(n_epochs):
    wls_f[i] = redshift(wl, vAs[i])


# Falling plot of all eight epochs of each spectrum, overlaid with the velocities for each
# Show spectra on each plot along with chosen amplitude scaling
fig, ax = plt.subplots(nrows=n_epochs, sharex=True)

for i in range(n_epochs):
    ax[i].plot(wls_f[i], fl_f)
    ax[i].set_ylabel("epoch {:}".format(i))

ax[-1].set_xlabel(r"$\lambda [\AA]$")
fig.savefig("SB1/dataset_noiseless_full.png", dpi=300)


# let alpha be the percentage of the primary as the total flux.
alpha = 1.0

# Truncate down to a smaller region to ensure overlap between all orders.
wl0 = 5255
wl1 = 5275

# Keep everything the same size. These are how many pixels we plan to keep in common between
# epochs
ind = (wls_f[0] > wl0) & (wls_f[0] < wl1)
n_pix_common = np.sum(ind)
print("n_pix_common = {}".format(n_pix_common))

# Now choose a narrower, common wl grid, which will be f.
# Now we should have a giant array of wavelengths that all share the same flux values, but shifted
wls_comb = np.zeros((n_epochs, n_pix_common))
fls_f = np.empty((n_epochs, n_pix_common))
fls_noise = np.zeros((n_epochs, n_pix_common))


# Assume a S/N = 40, so N = 1.0 / 40
S_N = 25
noise_amp = 1.0 / S_N
sigma_comb = noise_amp * np.ones((n_epochs, n_pix_common))


for i in range(n_epochs):
    # Select a subset of wl_f that has the appropriate number of pixels
    ind_0 = np.searchsorted(wls_f[i], wl0)
    print("Inserting at index {}, wavelength {:.2f}".format(ind_0, wls_f[i, ind_0]))

    wl_common = wls_f[i, ind_0:(ind_0 + n_pix_common)]

    # Interpolate the master spectrum onto this grid
    interp = interp1d(wls_f[i], fl_f)
    fl_common = interp(wl_common)

    # Add noise to it
    fl_common_noise = fl_common + np.random.normal(scale=noise_amp, size=n_pix_common)

    # Store into array
    wls_comb[i] = wl_common
    fls_f[i] = fl_common
    fls_noise[i] = fl_common_noise

    fig, ax = plt.subplots(nrows=2, sharex=True)
    ax[0].plot(wl_common, fl_common, "b")
    ax[0].set_ylabel(r"$f$")
    ax[1].plot(wl_common, fl_common_noise, "k")
    ax[1].set_ylabel(r"$f +$ noise")
    ax[-1].set_xlabel(r"$\lambda\;[\AA]$")
    fig.savefig("SB1/epoch_{}.png".format(i), dpi=300)


# Save the created spectra into a chunk
date_comb = obs_dates[:,np.newaxis] * np.ones_like(wls_comb)
chunkSpec = Chunk(wls_comb, fls_noise, sigma_comb, date_comb)
wl0 = np.min(wls_comb)
wl1 = np.max(wls_comb)

chunkSpec.save(0, wl0, wl1, prefix="SB1/")

# np.save("fake/fake_SB1_wls.npy", wls_comb)
# np.save("fake/fake_SB1_fls_noiseless.npy", fls_f)
# np.save("fake/fake_SB1_fls.npy", fls_noise)
# np.save("fake/fake_SB1_sigmas.npy", sigma_comb)



# Routines to make fake datasets.


# Fake SB2

# Specify orbital parameters and make a sanity plot

q = 0.2
K = 5.0 # km/s
e = 0.2 #
omega = 10.0 # deg
P = 10.0 # days
T0 = 0.0 # epoch
gamma = 5.0 # km/s

n_epochs = 10

obs_dates = np.array([2.1, 4.9, 8.0, 9.9, 12.2, 16.0, 16.9, 19.1, 22.3, 26.1])
# obs_dates = np.linspace(5, 150, num=n_epochs)


orb = orbit.SB2(q, K, e, omega, P, T0, gamma, obs_dates)

vAs, vBs = orb.get_component_velocities()

dates_fine = np.linspace(0, 30, num=200)
vA_fine, vB_fine = orb.get_component_velocities(dates_fine)

vAs_relative = vAs - vAs[0]
np.save("SB2/vAs_relative.npy", vAs_relative)

vBs_relative = vBs - vBs[0]
np.save("SB2/vBs_relative.npy", vBs_relative)

fig, ax = plt.subplots(nrows=3, figsize=(6,6))
ax[0].plot(dates_fine, vA_fine, "b")
ax[0].plot(orb.obs_dates, vAs, "bo")

ax[0].plot(dates_fine, vB_fine, "g")
ax[0].plot(orb.obs_dates, vBs, "go")

ax[0].axhline(gamma, ls="-.", color="0.5")
ax[-1].set_xlabel(r"$t$ [days]")
ax[0].set_ylabel(r"$v_A$ [km $\mathrm{s}^{-1}$]")

# For subsequent axes, plot velocities of stars relative to first observation.
ax[1].plot(orb.obs_dates, vAs_relative, "bo")
ax[1].set_ylabel(r"$v_A$ relative")

ax[2].plot(orb.obs_dates, vBs_relative, "go")
ax[2].set_ylabel(r"$v_B$ relative")

fig.subplots_adjust(left=0.14, right=0.86, bottom=0.24)
fig.savefig("SB2/orbit.png")


# Load the fake primary spectra we prepared
wl_f, fl_f = np.load("primary_wl_fl.npy")

# Load the fake secondary spectra we prepared
wl_g, fl_g = np.load("secondary_wl_fl.npy")


n_f = len(wl_f)
n_g = len(wl_g)


print("n_f:", n_f, "n_g:", n_g)

# Shorten these to be the same.
if n_f < n_g:
    n_pix = n_f
    print("Shortening g to f")
else:
    n_pix =n_g
    print("Shortening f to g")

wl = wl_f[0:n_pix]
fl_f = fl_f[0:n_pix]
fl_g = fl_g[0:n_pix]

# Just assume that wl_f will be wl_g as well.

# Create fake wavelengths with Doppler shifts by apply these to the master wl
wls_f = np.empty((n_epochs, n_pix))
wls_g = np.empty((n_epochs, n_pix))

for i in range(n_epochs):
    wls_f[i] = redshift(wl, vAs[i])
    wls_g[i] = redshift(wl, vBs[i])

# Falling plot of all eight epochs of each spectrum, overlaid with the velocities for each
# Show spectra on each plot along with chosen amplitude scaling
fig, ax = plt.subplots(nrows=n_epochs, sharex=True)

for i in range(n_epochs):
    ax[i].plot(wls_f[i], fl_f, "b")
    ax[i].plot(wls_g[i], fl_g, "g")
    ax[i].set_ylabel("epoch {:}".format(i))

ax[-1].set_xlabel(r"$\lambda [\AA]$")
fig.savefig("SB2/dataset_noiseless_full.png", dpi=300)


# Here is where we set up the number of chunks, and choose what region of overlaps we want.
# New chunks [start, stop]
# chunk_wls = [[5240, 5250], [5255, 5265], [5270, 5280]]

chunk_wls = [[5265, 5275]]

# Measure this as S/N per resolution element. That means that there is a sqrt(2.5) effect.

# let alpha be the percentage of the primary as the total flux.
ratio = 0.2
alpha = (1 / (ratio + 1))
print("Ratio: {}, alpha: {}".format(ratio, alpha))
# alpha = 0.90

# Assume a S/N = 40, so N = 1.0 / 40
S_N = 60 # per resolution element
noise_amp = 1.0 / (S_N/np.sqrt(2.5)) # per pixel

# Truncate down to a smaller region to ensure overlap between all orders.
for (wl0, wl1) in chunk_wls:
    print("Creating chunk {:.0f} to {:.0f}".format(wl0, wl1))

    # Keep everything the same size. These are how many pixels we plan to keep in common between
    # epochs
    ind = (wls_f[0] > wl0) & (wls_f[0] < wl1)
    n_pix_common = np.sum(ind)
    print("n_pix_common = {}".format(n_pix_common))

    # Now choose a narrower, common wl grid, which will just be f.
    # Now we should have a giant array of wavelengths that all share the same flux values, but shifted
    wls_comb = np.zeros((n_epochs, n_pix_common))
    fls_f = np.empty((n_epochs, n_pix_common))
    fls_g = np.empty((n_epochs, n_pix_common))
    fls_comb = np.empty((n_epochs, n_pix_common))
    fls_noise = np.zeros((n_epochs, n_pix_common))

    sigma_comb = noise_amp * np.ones((n_epochs, n_pix_common))


    for i in range(n_epochs):
        # Select a subset of wl_f that has the appropriate number of pixels
        ind_0 = np.searchsorted(wls_f[i], wl0)
        print("Inserting at index {}, wavelength {:.2f}".format(ind_0, wls_f[i, ind_0]))

        wl_common = wls_f[i, ind_0:(ind_0 + n_pix_common)]

        # Interpolate the master spectrum onto this grid
        interp = interp1d(wls_f[i], fl_f)
        fl_f_common = interp(wl_common)

        interp = interp1d(wls_g[i], fl_g)
        fl_g_common = interp(wl_common)

        fl_common = alpha * fl_f_common + (1 - alpha) * fl_g_common

        # Add noise to it
        fl_common_noise = fl_common + np.random.normal(scale=noise_amp, size=n_pix_common)

        # Store into array
        wls_comb[i] = wl_common
        fls_f[i] = fl_f_common
        fls_g[i] = fl_g_common
        fls_comb[i] = fl_common
        fls_noise[i] = fl_common_noise


        fig, ax = plt.subplots(nrows=4, sharex=True)
        ax[0].plot(wl_common, alpha * fl_f_common, "b")
        ax[0].set_ylabel(r"$f$")
        ax[1].plot(wl_common, (1 - alpha) * fl_g_common, "g")
        ax[1].set_ylabel(r"$g$")
        ax[2].plot(wl_common, fl_common, "k")
        ax[2].set_ylabel(r"$f + g$")
        ax[3].plot(wl_common, fl_common_noise, "k")
        ax[3].set_ylabel(r"$f + g +$ noise")
        ax[-1].set_xlabel(r"$\lambda\;[\AA]$")
        fig.savefig("SB2/epoch_{}.png".format(i), dpi=300)

    # Save the created spectra into a chunk
    date_comb = obs_dates[:,np.newaxis] * np.ones_like(wls_comb)
    chunkSpec = Chunk(wls_comb, fls_noise, sigma_comb, date_comb)
    wl0 = np.min(wls_comb)
    wl1 = np.max(wls_comb)

    chunkSpec.save(0, wl0, wl1, prefix="SB2/")

    # 2D arrays before we have summed them or added noise.
    print("STDEV primary", np.std(alpha * fls_f))
    print("STDEV secondary", np.std((1 - alpha) * fls_g))

    np.save("SB2/fls_f.npy", alpha * fls_f)
    np.save("SB2/fls_g.npy", (1 - alpha) * fls_g)
    np.save("SB2/fls_comb.npy", fls_comb)

# Fake ST3

# Specify orbital parameters and make a sanity plot

q_inner = 0.4
K_inner = 5.0 # km/s
e_inner = 0.2 #
omega_inner = 10.0 # deg
P_inner = 10.0 # days
T0_inner = 0.0 # epoch
q_outer = 0.2
K_outer = 4.0 # km/s
e_outer = 0.2 #
omega_outer = 80.0 # deg
P_outer = 100.0 # days
T0_outer = 3.0 # epoch
gamma = 5.0 # km/s

n_epochs = 20
obs_dates = np.linspace(5, 150, num=n_epochs)

orb = orbit.ST3(q_inner, K_inner, e_inner, omega_inner, P_inner, T0_inner, q_outer, K_outer, e_outer, omega_outer, P_outer, T0_outer, gamma, obs_dates)

vAs, vBs, vCs = orb.get_component_velocities()

dates_fine = np.linspace(0, 35, num=200)
vA_fine, vB_fine, vC_fine = orb.get_component_velocities(dates_fine)


fig, ax = plt.subplots(nrows=4, sharex=True)
# Plot all three orbits on top of each other (the actual orbit)
ax[0].plot(dates_fine, vA_fine, "b")
ax[0].plot(orb.obs_dates, vAs, "bo")

ax[0].plot(dates_fine, vB_fine, "g")
ax[0].plot(orb.obs_dates, vBs, "go")

ax[0].plot(dates_fine, vC_fine, "r")
ax[0].plot(orb.obs_dates, vCs, "ro")

ax[0].axhline(gamma, ls="-.", color="0.5")
ax[-1].set_xlabel(r"$t$ [days]")
ax[0].set_ylabel(r"$v_A$ [km $\mathrm{s}^{-1}$]")

vAs_relative = vAs - vAs[0]
np.save("ST3/vAs_relative.npy", vAs_relative)

vBs_relative = vBs - vBs[0]
np.save("ST3/vBs_relative.npy", vBs_relative)

vCs_relative = vCs - vCs[0]
np.save("ST3/vCs_relative.npy", vCs_relative)

# For subsequent axes, plot velocities of stars relative to first observation.
ax[1].plot(orb.obs_dates, vAs_relative, "bo")
ax[1].set_ylabel(r"$v_A$ relative")

ax[2].plot(orb.obs_dates, vBs_relative, "go")
ax[2].set_ylabel(r"$v_B$ relative")

ax[3].plot(orb.obs_dates, vCs_relative, "ro")
ax[3].set_ylabel(r"$v_C$ relative")

fig.subplots_adjust(left=0.14, right=0.86, bottom=0.24)
fig.savefig("ST3/orbit.png")


# Load the fake primary spectra we prepared
wl_f, fl_f = np.load("primary_wl_fl.npy")

# Load the fake secondary spectra we prepared
wl_g, fl_g = np.load("secondary_wl_fl.npy")

# Load the fake secondary spectra we prepared
wl_h, fl_h = np.load("secondary_wl_fl.npy")

n_f = len(wl_f)
n_g = len(wl_g)
n_h = len(wl_h)

print("n_f:", n_f, "n_g:", n_g, "n_h:", n_h)

# Shorten these to be the same.
if n_f < n_g:
    n_pix = n_f
    print("Shortening g to f")
else:
    n_pix = n_g
    print("Shortening f to g")

wl = wl_f[0:n_pix]
fl_f = fl_f[0:n_pix]
fl_g = fl_g[0:n_pix]
fl_h = fl_g[0:n_pix]

# Just assume that wl_f will be wl_g as well.

# Create fake wavelengths with Doppler shifts by apply these to the master wl
wls_f = np.empty((n_epochs, n_pix))
wls_g = np.empty((n_epochs, n_pix))
wls_h = np.empty((n_epochs, n_pix))

for i in range(n_epochs):
    wls_f[i] = redshift(wl, vAs[i])
    wls_g[i] = redshift(wl, vBs[i])
    wls_h[i] = redshift(wl, vCs[i])

# Falling plot of all eight epochs of each spectrum, overlaid with the velocities for each
# Show spectra on each plot along with chosen amplitude scaling
fig, ax = plt.subplots(nrows=n_epochs, sharex=True)

for i in range(n_epochs):
    ax[i].plot(wls_f[i], fl_f, "b")
    ax[i].plot(wls_g[i], fl_g, "g")
    ax[i].plot(wls_h[i], fl_h, "r")
    ax[i].set_ylabel("epoch {:}".format(i))

ax[-1].set_xlabel(r"$\lambda [\AA]$")
fig.savefig("ST3/dataset_noiseless_full.png", dpi=300)

# let alpha be the percentage of the primary as the total flux.
alpha = 0.5
beta = 0.3

# Here is where we set up the number of chunks, and choose what region of overlaps we want.
# New chunks [start, stop]
chunk_wls = [[5240, 5250], [5255, 5265], [5270, 5280]]


# wl0 = 5255
# wl1 = 5275

# Truncate down to a smaller region to ensure overlap between all orders.
for (wl0, wl1) in chunk_wls:
    print("Creating chunk {:.0f} to {:.0f}".format(wl0, wl1))

    # Keep everything the same size. These are how many pixels we plan to keep in common between
    # epochs
    ind = (wls_f[0] > wl0) & (wls_f[0] < wl1)
    n_pix_common = np.sum(ind)
    print("n_pix_common = {}".format(n_pix_common))

    # Now choose a narrower, common wl grid, which will just be f.
    # Now we should have a giant array of wavelengths that all share the same flux values, but shifted
    wls_comb = np.zeros((n_epochs, n_pix_common))
    fls_f = np.empty((n_epochs, n_pix_common))
    fls_g = np.empty((n_epochs, n_pix_common))
    fls_h = np.empty((n_epochs, n_pix_common))
    fls_comb = np.empty((n_epochs, n_pix_common))
    fls_noise = np.zeros((n_epochs, n_pix_common))


    # Assume a S/N = 40, so N = 1.0 / 40
    S_N = 40
    noise_amp = 1.0 / S_N
    sigma_comb = noise_amp * np.ones((n_epochs, n_pix_common))


    for i in range(n_epochs):
        # Select a subset of wl_f that has the appropriate number of pixels
        ind_0 = np.searchsorted(wls_f[i], wl0)
        print("Inserting at index {}, wavelength {:.2f}".format(ind_0, wls_f[i, ind_0]))

        wl_common = wls_f[i, ind_0:(ind_0 + n_pix_common)]

        # Interpolate the master spectrum onto this grid
        interp = interp1d(wls_f[i], fl_f)
        fl_f_common = interp(wl_common)

        interp = interp1d(wls_g[i], fl_g)
        fl_g_common = interp(wl_common)

        interp = interp1d(wls_h[i], fl_h)
        fl_h_common = interp(wl_common)

        fl_common = alpha * fl_f_common + beta * fl_g_common + (1 - (alpha + beta)) * fl_h_common

        # Add noise to it
        fl_common_noise = fl_common + np.random.normal(scale=noise_amp, size=n_pix_common)

        # Store into array
        wls_comb[i] = wl_common
        fls_f[i] = fl_f_common
        fls_g[i] = fl_g_common
        fls_h[i] = fl_h_common
        fls_comb[i] = fl_common
        fls_noise[i] = fl_common_noise

        fig, ax = plt.subplots(nrows=5, sharex=True)
        ax[0].plot(wl_common, alpha * fl_f_common, "b")
        ax[0].set_ylabel(r"$f$")
        ax[1].plot(wl_common, beta * fl_g_common, "g")
        ax[1].set_ylabel(r"$g$")
        ax[2].plot(wl_common, (1 - (alpha + beta)) * fl_h_common, "r")
        ax[2].set_ylabel(r"$h$")
        ax[3].plot(wl_common, fl_common, "k")
        ax[3].set_ylabel(r"$f + g + h$")
        ax[4].plot(wl_common, fl_common_noise, "k")
        ax[4].set_ylabel(r"$f + g + h +$ noise")
        ax[-1].set_xlabel(r"$\lambda\;[\AA]$")
        fig.savefig("ST3/{:.0f}_{:.0f}_epoch_{}.png".format(wl0, wl1, i), dpi=300)
        plt.close("all")

    # Save the created spectra into a chunk
    date_comb = obs_dates[:,np.newaxis] * np.ones_like(wls_comb)
    chunkSpec = Chunk(wls_comb, fls_noise, sigma_comb, date_comb)
    wl0 = np.min(wls_comb)
    wl1 = np.max(wls_comb)

    chunkSpec.save(0, wl0, wl1, prefix="ST3/")
