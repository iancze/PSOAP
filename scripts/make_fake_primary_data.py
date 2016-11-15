import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import cho_factor, cho_solve

from psoap import constants as C
from psoap.data import lkca14, redshift
from psoap import covariance


# Optimized parameters for this chunk. Not that relevant though, since we are just using the GP
# as an interpolator.

lkca14.sort_by_SN()

amp_f = 0.2
l_f = 5.58

order = 23

# Select the relevant wavelengths
wl = lkca14.wl[0, order, :]

wl0 = 5235
wl1 = 5285

ind = (wl > wl0) & (wl < wl1)

n_epochs= 3

# Optimize the GP using the first 8 epochs
wl = lkca14.wl[0:n_epochs, order, ind]
fl = lkca14.fl[0:n_epochs, order, ind]
sigma = lkca14.sigma[0:n_epochs, order, ind]
date = lkca14.date[0:n_epochs]

n_epochs, n_pix = wl.shape

# Apply some calibration optimzation, although not strictly necessary
fl = covariance.cycle_calibration(wl, fl, sigma, amp_f, l_f, ncycles=3, limit_array=3)

print("finished cycling calibration")

# Optimize the GP parameters
amp_f, l_f = covariance.optimize_GP_f(wl.flatten(), fl.flatten(), sigma.flatten(), amp_f, l_f)
print("finished optimizing GP", amp_f, l_f)

# Use optimized fluxes, optimized GP parameters, and first epoch wavelength grid to predict a
# mean flux vector on to first epoch, which we take to be the master grid.
wl_predict = wl[0]
fl_predict, Sigma = covariance.predict_f(wl.flatten(), fl.flatten(), sigma.flatten(), wl_predict, amp_f, l_f)

# Plot all spectra up to see what it looks like
fig,ax = plt.subplots()
for i in range(n_epochs):
    ax.plot(wl[i], fl[i])
ax.plot(wl_predict, fl_predict, "k", lw=1.2)
fig.savefig("fake/primary_spectra.png")

np.save("fake/primary_wl_fl.npy", np.array([wl_predict, fl_predict]))
