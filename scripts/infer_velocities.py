import matplotlib.pyplot as plt
import numpy as np

from LkCa import constants as C
from LkCa import covariance
from LkCa.data import redshift, Spectrum
from LkCa import matrix_functions
from LkCa.matrix_functions import get_V11_three

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
V11s = []

n_epochs = 20

# date1D = np.load("opt/date1D.npy")[0:n_epochs]


for chunk in chunks:
    order, wl0, wl1 = chunk
    wl = np.load("opt/wl_{:}_{:.0f}_{:.0f}.npy".format(order, wl0, wl1))[0:n_epochs]
    wls.append(wl)
    fls.append(np.load("opt/fl_{:}_{:.0f}_{:.0f}.npy".format(order, wl0, wl1))[0:n_epochs])
    sigmas.append(np.load("opt/sigma_{:}_{:.0f}_{:.0f}.npy".format(order, wl0, wl1))[0:n_epochs])
    dates.append(np.load("opt/date1D_{:}_{:.0f}_{:.0f}.npy".format(order, wl0, wl1))[0:n_epochs])
    n_pixs.append(wl.shape[1])
    N = len(wl.flatten())
    # Define empty matrices that we will fill for each chunk
    V11s.append(np.empty((N, N), dtype=np.float64))


def lnp(p):

    # q_inner, q_outer, amp_f, l_f, amp_g, l_g, amp_h, l_h = p
    q_inner, q_outer, amp_f, amp_g, amp_h, l = p

    l_f = l
    l_g = l
    l_h = l

    if q_inner < 0.01 or q_inner > 1.0 or q_outer < 0.01 or q_outer > 1.0 or amp_f < 0.0 or amp_f > 1.0 or amp_g < 0.0 or amp_g > 1.0 or amp_h < 0.0 or l_f < 0.0 or l_h > 100.0 or l_g < 0.0 or l_g > 100.0 or l_h < 0.0 or l_h > 100:
        return -np.inf

    # Get the lnp for each chunk component
    lnps = []

    for V11, wl, fl, sigma, date in zip(V11s, wls, fls, sigmas, dates):

        vA = get_vA(date)
        # Shift wl to rest-frame of A component
        wl_A = redshift(wl, -vA[:,np.newaxis])

        # Get velocities for B component
        vB = get_vB(date, q_inner)

        # Shift wl to rest-frame of B component
        wl_B = redshift(wl, -vB[:,np.newaxis])

        # Get velocities for C component
        vC = get_vC(date, q_outer)

        # Shift wl to rest-frame of C component
        wl_C = redshift(wl, -vC[:,np.newaxis])

        # V11 = covariance.get_V11_three(wl_A.flatten(), wl_B.flatten(), wl_C.flatten(), 1.0 * sigma.flatten(), amp_f, l_f, amp_g, l_g, amp_h, l_h)

        # Acts on array passed into function.
        get_V11_three(V11, wl_A.flatten(), wl_B.flatten(), wl_C.flatten(), amp_f, l_f, amp_g, l_g, amp_h, l_h)

        # add observational uncertainty along diagonals
        V11[np.diag_indices_from(V11)] += sigma.flatten()**2

        try:
            factor, flag = cho_factor(V11)
        except np.linalg.linalg.LinAlgError:
            print("Returning -np.inf for", p)
            return -np.inf

        logdet = np.sum(2 * np.log((np.diag(factor))))

        lnprob = -0.5 * (np.dot((fl.flatten() - 1.0).T, cho_solve((factor, flag), (fl.flatten() - 1.0))) + logdet)

        lnps.append(lnprob)

    # Sum all the probabilities of the chunks together
    lnprob = np.sum(np.array(lnps))
    return lnprob

def objective(p):
    return -lnp(p)

# p0 = np.array([0.6, 0.15, 0.047, 31.0, 0.02, 28.0, 0.02, 22.0])
# p0 = np.array([0.6, 0.14, 0.03, 0.01, 0.01])
# print("returned lnprob is", lnp(p0))

# ans = minimize(objective, p0, method="Nelder-Mead")
# print(ans)
# #
# import sys
# sys.exit()

from emcee import EnsembleSampler

ndim = 6
nwalkers = 3 * ndim
# #
# p0 = np.array([np.random.uniform(0.6, 0.65, nwalkers),
#             np.random.uniform(0.3, 0.35, nwalkers),
#             np.random.uniform(0.04, 0.06, nwalkers),
#             # np.random.uniform(25.0, 29.00, nwalkers),
#             np.random.uniform(0.001, 0.01, nwalkers),
#             # np.random.uniform(25.0, 29.00, nwalkers),
#             np.random.uniform(0.001, 0.01, nwalkers),
#             np.random.uniform(25.0, 30.0, nwalkers)]).T

# Optionally start from the previous location
p0 = np.load("chain.npy")[:,-1,:]

# import sys
# sys.exit()

sampler = EnsembleSampler(nwalkers, ndim, lnp, threads=4)

# pos, prob, state = sampler.run_mcmc(p0, 100)

nsteps = 200
for i, result in enumerate(sampler.sample(p0, iterations=nsteps)):
    print("Iteration", i)
    # if (i+1) % 100 == 0:
        # print("{0:5.1%}".format(float(i) / nsteps))

# Save the actual chain of samples
np.save("lnprob.npy", sampler.lnprobability)
np.save("chain.npy", sampler.chain)
