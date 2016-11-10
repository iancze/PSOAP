
# coding: utf-8

# In[1]:

import numpy as np
import matplotlib

import matplotlib.pyplot as plt
import psoap
from psoap.data import lkca14
from psoap import matrix_functions
from psoap import covariance

from matplotlib.ticker import FormatStrFormatter as FSF
from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import MultipleLocator


# In[2]:

order = 22
wl0 = 5200
wl1 = 5270


# In[3]:

a = 0.26
l = 6.5


# In[4]:

n_epochs = 10
wl = lkca14.wl[0:n_epochs,order,:]
# Sort by ind
ind = (wl[0] > wl0) & (wl[0] < wl1)

wl = wl[:,ind]
fl = lkca14.fl[0:n_epochs,order,ind]
sigma = lkca14.sigma[0:n_epochs,order,ind]
date1D = lkca14.date1D[0:n_epochs]


# In[ ]:

# determine the calibration polynomials for each order

# wl_tweak = wl[1]
# fl_tweak = fl[1]
# sigma_tweak = sigma[1]
#
# # Temporary arrays without the epoch we just chose
# wl_remain = np.delete(wl, 1, axis=0)[0:3]
# fl_remain = np.delete(fl, 1, axis=0)[0:3]
# sigma_remain = np.delete(sigma, 1, axis=0)[0:3]
# #
# # Individual routines
# N_A = len(wl_cal)
# A = np.empty((N_A, N_A), dtype=np.float64)
#
# N_B = len(wl_fixed)
# B = np.empty((N_B, N_B), dtype=np.float64)
#
# C = np.empty((N_A, N_B), dtype=np.float64)
#
# matrix_functions.fill_V11_one(A, wl_cal, amp, l_f)
# matrix_functions.fill_V11_one(B, wl_fixed, amp, l_f)
# matrix_functions.fill_V12_one(C, wl_cal, wl_fixed, amp, l_f)

# Add in sigmas
# A[np.diag_indices_from(A)] += sigma_cal**2
# B[np.diag_indices_from(B)] += sigma_fixed**2

# optimize the calibration of "tweak" with respect to all other orders
# fl_cor, X = covariance.optimize_calibration(wl_tweak, fl_tweak, sigma_tweak, wl_remain.flatten(), fl_remain.flatten(), sigma_remain.flatten(), a, l)


fl_cal = covariance.cycle_calibration(wl, fl, sigma, a, l, ncycles=1, order=4, limit_array=3, soften=1.0)


# In[ ]:

fig, ax = plt.subplots(nrows=3, sharex=True, figsize=(3.5, 5.5))

for i in range(n_epochs):
    ax[0].plot(wl[i], fl[i], lw=0.8)



ax[-1].set_xlabel(r"$\lambda\;[\AA]$")

fig.savefig("calibration.pdf")
fig.savefig("calibration.png")


# In[ ]:
