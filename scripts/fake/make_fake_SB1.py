import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

import psoap
from psoap.data import lkca14, redshift, Chunk
from psoap import matrix_functions
from psoap import covariance
from psoap import orbit

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
