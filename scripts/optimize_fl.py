import matplotlib.pyplot as plt
import numpy as np

from LkCa import constants as C
from LkCa import covariance
from LkCa.data import redshift, Spectrum

from orbit import get_vA, get_vB, get_vC

from scipy.linalg import cho_factor, cho_solve

from astropy.io import ascii

gwori = Spectrum("../../data/GWOri.hdf5")

# Optimize all, for the hell of it.
n_epochs = 80
amp_f = 0.15
l_f = 25.0


# Load chunk wavelengths from the file
chunks = ascii.read("chunks.dat")
print(chunks)

for chunk in chunks:
    order, wl0, wl1 = chunk

    wl = gwori.wl[0, order, :]

    ind = (wl > wl0) & (wl < wl1)

    wl = gwori.wl[:, order, ind]
    fl = gwori.fl[:, order, ind]
    sigma = gwori.sigma[:, order, ind]
    date = gwori.date[:]

    ind_sort = np.load("ind_sort_by_SN.npy")

    wl = wl[ind_sort][0:n_epochs]
    fl = fl[ind_sort][0:n_epochs]
    sigma = sigma[ind_sort][0:n_epochs]
    date = date[ind_sort][0:n_epochs]

    date1D = date[:, 0, 0]

    # Figure out if there is anything that is bad.
    ind = ~np.any(fl > 1.2, axis=1)
    wl = wl[ind]
    fl = fl[ind]
    sigma = sigma[ind]
    date = date[ind]
    date1D = date1D[ind]

    n_epochs, n_pix = wl.shape

    # Treat all spectra by just one GP and see how well optimization works.
    fl_cal = covariance.cycle_calibration(wl, fl, 1.5 * sigma, amp_f, l_f, ncycles=3, limit_array=3, mu_GP=1.0)

    np.save("opt/wl_{:}_{:.0f}_{:.0f}.npy".format(order, wl0, wl1), wl)
    np.save("opt/fl_{:}_{:.0f}_{:.0f}.npy".format(order, wl0, wl1), fl_cal)
    np.save("opt/sigma_{:}_{:.0f}_{:.0f}.npy".format(order, wl0, wl1), sigma)
    np.save("opt/date_{:}_{:.0f}_{:.0f}.npy".format(order, wl0, wl1), date)
    np.save("opt/date1D_{:}_{:.0f}_{:.0f}.npy".format(order, wl0, wl1), date1D)


    fig, ax = plt.subplots(nrows=2, sharex=True)

    for i in range(n_epochs):
        ax[0].plot(wl[i], fl[i])
        ax[1].plot(wl[i], fl_cal[i])

    fig.savefig("plots/fl_opt_{:}_{:.0f}_{:.0f}.png".format(order, wl0, wl1), dpi=300)
