import matplotlib.pyplot as plt
import numpy as np

from LkCa import constants as C
from LkCa import covariance
from LkCa.data import redshift, Spectrum

from orbit import get_vA, get_vB, get_vC

from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize
from astropy.io import ascii

gwori = Spectrum("../../data/GWOri.hdf5")

# Load chunk wavelengths from the file
chunks = ascii.read("chunks.dat")
print("Using the following chunks")
print(chunks)

# Load these into chunks that we can iterate over
wls = []
fls = []
sigmas = []
dates = []
n_pixs = []

n_epochs = 10

date1D = np.load("opt/date1D.npy")[0:n_epochs]

for chunk in chunks:
    order, wl0, wl1 = chunk
    wl = np.load("opt/wl_{:}_{:.0f}_{:.0f}.npy".format(order, wl0, wl1))[0:n_epochs]
    wls.append(wl)
    fls.append(np.load("opt/fl_{:}_{:.0f}_{:.0f}.npy".format(order, wl0, wl1))[0:n_epochs])
    sigmas.append(np.load("opt/sigma_{:}_{:.0f}_{:.0f}.npy".format(order, wl0, wl1))[0:n_epochs])
    dates.append(np.load("opt/date_{:}_{:.0f}_{:.0f}.npy".format(order, wl0, wl1))[0:n_epochs])
    n_pixs.append(wl.shape[1])


# Treat velocities for A as known
vA = get_vA(date1D)
# Shift wl to rest-frame of A component
wls_A = [redshift(wl, -vA[:,np.newaxis]) for wl in wls]

q_inner = 0.6
q_outer = 0.17

amp_f = 0.04
amp_g = 0.012
amp_h = 0.010


# amp_f = 0.047
# amp_g = 0.02
# amp_h = 0.012

# l_f = 31.5
# l_g = 28.0
# l_h = 22.0

l_f = 30.
l_g = 21.0
l_h = 20.0

for chunk, wl_A, wl, fl, sigma, n_pix in zip(chunks, wls_A, wls, fls, sigmas, n_pixs):
    order, wl0, wl1 = chunk

    # Predict the mean spectrum onto the input wavelengths

    vB = get_vB(date1D, q_inner)
    wl_B = redshift(wl, -vB[:,np.newaxis])

    vC = get_vC(date1D, q_outer)
    wl_C = redshift(wl, -vC[:,np.newaxis])

    # First predict the component spectra as mean 1 GPs
    mu, Sigma = covariance.predict_f_g_h(wl_A.flatten(), wl_B.flatten(), wl_C.flatten(), wl.flatten(), fl.flatten(), sigma.flatten(), mu_f=0.0, mu_g=0.0, mu_h=0.0, amp_f=amp_f, l_f=l_f, amp_g=amp_g, l_g=l_g, amp_h=amp_h, l_h=l_h)

    mu_sum, Sigma_sum = covariance.predict_f_g_h_sum(wl_A.flatten(), wl_B.flatten(), wl_C.flatten(), fl.flatten(), sigma.flatten(), wl_A.flatten(), wl_B.flatten(), wl_C.flatten(), mu_fgh=1.0, amp_f=amp_f, l_f=l_f, amp_g=amp_g, l_g=l_g, amp_h=amp_h, l_h=l_h)

    mu_f = mu[0:(n_pix * n_epochs)]
    mu_g = mu[(n_pix * n_epochs):2 * (n_pix * n_epochs)]
    mu_h = mu[2 * (n_pix * n_epochs):]

    # Reshape mu
    mu_f.shape = (n_epochs, -1)
    mu_g.shape = (n_epochs, -1)
    mu_h.shape = (n_epochs, -1)
    mu_sum.shape = (n_epochs, -1)


    for i in range(n_epochs):
        fig, ax = plt.subplots(nrows=4, sharex=True)

        ax[0].plot(wl[i], fl[i], ".", color="0.4")
        ax[0].plot(wl[i], mu_sum[i], "b")
        ax[0].plot(wl[i], mu_f[i] + mu_g[i] + mu_h[i] + 1.0, "m", ls="-.")

        ax[1].plot(wl[i], mu_f[i], "b")
        ax[2].plot(wl[i], mu_g[i], "g")
        ax[3].plot(wl[i], mu_h[i], "r")
        ax[-1].set_xlabel(r"$\lambda$")

        fig.savefig("plots/epoch_{}_{:.0f}_{:.0f}_{}.png".format(order, wl0, wl1, i), dpi=300)
        plt.close("all")
