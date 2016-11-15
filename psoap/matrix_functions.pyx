# encoding: utf-8
# cython: profile=True
# cython: linetrace=True
# filename: matrix_functions.pyx

import numpy as np
# from scipy.linalg import block_diag
cimport numpy as np
cimport cython
import psoap.constants as C
import math

# this is the fast expoential function
from libc.math cimport exp

cdef double c_kms = 2.99792458e5 #km s^-1

@cython.boundscheck(False)
@cython.wraparound(False)
def fill_V11_f(np.ndarray[np.double_t, ndim=2] mat, np.ndarray[np.double_t, ndim=1] wl_f, double amp_f, double l_f):

    cdef int N = len(mat)
    cdef int i = 0
    cdef int j = 0

    # Compute the squared values of these to save time within the for-loop
    cdef double amp2f = amp_f*amp_f
    cdef double p2f = -0.5/(l_f*l_f)

    # Temporary distance holders
    cdef double rf = 0.0

    # Temporary wavelength holder
    cdef double wl_f0 = 0.0
    cdef double wl_f1 = 0.0

    cdef double cov = 0.0

    #Loop over all the non-diagonal indices
    for i in range(N):

      wl_f0 = wl_f[i]

      for j in range(i):

        wl_f1 = wl_f[j]

        # Calculate the distance in km/s
        rf = c_kms/2.0 * (wl_f1 - wl_f0) / (wl_f1 + wl_f0)

        cov = amp2f * exp(p2f * rf * rf)

        # Enter this on both sides of the diagonal, since the matrix is symmetric
        mat[i,j] = cov
        mat[j,i] = cov

    #Loop over all the diagonals, since the distance here is 0.
    for i in range(N):
        mat[i,i] = amp2f

    # No return statetment

@cython.boundscheck(False)
@cython.wraparound(False)
def fill_V12_f(np.ndarray[np.double_t, ndim=2] mat, np.ndarray[np.double_t, ndim=1] wl_f, np.ndarray[np.double_t, ndim=1] wl_predict, double amp_f, double l_f):

    cdef int M = len(wl_f)
    cdef int N = len(wl_predict)
    cdef int i = 0
    cdef int j = 0

    # Compute the squared values of these to save time within the for-loop
    cdef double amp2f = amp_f*amp_f
    cdef double p2f = -0.5/(l_f*l_f)

    # Temporary distance holders
    cdef double rf = 0.0

    # Temporary wavelength holder
    cdef double wl_f0 = 0.0
    cdef double wl_f1 = 0.0

    cdef double cov = 0.0

    #Loop over all the non-diagonal indices
    for i in range(M):

      wl_f0 = wl_f[i]

      for j in range(N):

        wl_f1 = wl_predict[j]

        rf = c_kms/2.0 * (wl_f1 - wl_f0) / (wl_f1 + wl_f0)

        cov = amp2f * exp(p2f * rf * rf)

        # Enter this on just one side the diagonal, since the matrix is not symmetric
        mat[i,j] = cov

    # no return


@cython.boundscheck(False)
@cython.wraparound(False)
def fill_V11_f_g(np.ndarray[np.double_t, ndim=2] mat, np.ndarray[np.double_t, ndim=1] wl_f, np.ndarray[np.double_t, ndim=1] wl_g, double amp_f, double l_f, double amp_g, double l_g):

    cdef int N = len(mat)
    cdef int i = 0
    cdef int j = 0

    # Compute the squared values of these to save time within the for-loop
    cdef double amp2f = amp_f*amp_f
    cdef double p2f = -0.5/(l_f*l_f)

    cdef double amp2g = amp_g*amp_g
    cdef double p2g = -0.5/(l_g*l_g)

    # Temporary distance holders
    cdef double rf = 0.0
    cdef double rg = 0.0

    # Temporary wavelength holders
    cdef double wl_f0 = 0.0
    cdef double wl_f1 = 0.0
    cdef double wl_g0 = 0.0
    cdef double wl_g1 = 0.0

    cdef double cov = 0.0

    #Loop over all the non-diagonal indices
    for i in range(N):

      wl_f0 = wl_f[i]
      wl_g0 = wl_g[i]

      for j in range(i):

        # Just indexing each is very slow.
        wl_f1 = wl_f[j]
        wl_g1 = wl_g[j]

        # Calculate the distance in km/s
        rf = c_kms/2.0 * (wl_f1 - wl_f0) / (wl_f1 + wl_f0)
        rg = c_kms/2.0 * (wl_g1 - wl_g0) / (wl_g1 + wl_g0)

        cov = amp2f * exp(p2f * rf*rf) + amp2g * exp(p2g * rg*rg)

        # Enter this on both sides of the diagonal, since the matrix is symmetric
        mat[i,j] = cov
        mat[j,i] = cov

    #Loop over all the diagonals, since the distance here is 0.
    for i in range(N):
        mat[i,i] = amp2f + amp2g

    # No return statement


@cython.boundscheck(False)
@cython.wraparound(False)
def fill_V11_f_g_h(np.ndarray[np.double_t, ndim=2] mat, np.ndarray[np.double_t, ndim=1] wl_f, np.ndarray[np.double_t, ndim=1] wl_g, np.ndarray[np.double_t, ndim=1] wl_h, double amp_f, double l_f, double amp_g, double l_g, double amp_h, double l_h):

    cdef int N = len(mat)
    cdef int i = 0
    cdef int j = 0

    # Compute the squared values of these to save time within the for-loop
    cdef double amp2f = amp_f*amp_f
    cdef double p2f = -0.5/(l_f*l_f)

    cdef double amp2g = amp_g*amp_g
    cdef double p2g = -0.5/(l_g*l_g)

    cdef double amp2h = amp_h*amp_h
    cdef double p2h = -0.5/(l_h*l_h)

    # Temporary distance holders
    cdef double rf = 0.0
    cdef double rg = 0.0
    cdef double rh = 0.0

    # Temporary wavelength holders
    cdef double wl_f0 = 0.0
    cdef double wl_f1 = 0.0
    cdef double wl_g0 = 0.0
    cdef double wl_g1 = 0.0
    cdef double wl_h0 = 0.0
    cdef double wl_h1 = 0.0

    cdef double cov = 0.0

    #Loop over all the non-diagonal indices
    for i in range(N):

      wl_f0 = wl_f[i]
      wl_g0 = wl_g[i]
      wl_h0 = wl_h[i]

      for j in range(i):

        # Just indexing each is very slow.
        wl_f1 = wl_f[j]
        wl_g1 = wl_g[j]
        wl_h1 = wl_h[j]

        # Calculate the distance in km/s
        rf = c_kms/2.0 * (wl_f1 - wl_f0) / (wl_f1 + wl_f0)
        rg = c_kms/2.0 * (wl_g1 - wl_g0) / (wl_g1 + wl_g0)
        rh = c_kms/2.0 * (wl_h1 - wl_h0) / (wl_h1 + wl_h0)

        cov = amp2f * exp(p2f * rf*rf) + amp2g * exp(p2g * rg*rg) + amp2h * exp(p2h * rh*rh)

        # Enter this on both sides of the diagonal, since the matrix is symmetric
        mat[i,j] = cov
        mat[j,i] = cov

    #Loop over all the diagonals, since the distance here is 0.
    for i in range(N):
        mat[i,i] = amp2f + amp2g



@cython.boundscheck(False)
@cython.wraparound(False)
def get_V11_three(np.ndarray[np.double_t, ndim=2] mat, np.ndarray[np.double_t, ndim=1] wlA, np.ndarray[np.double_t, ndim=1] wlB, np.ndarray[np.double_t, ndim=1] wlC, double amp_f, double l_f, double amp_g, double l_g, double amp_h, double l_h):

    cdef int N = len(mat)
    cdef int i = 0
    cdef int j = 0

    cdef double amp2f = amp_f*amp_f
    cdef double p2f = -0.5/(l_f*l_f)
    cdef double amp2g = amp_g*amp_g
    cdef double p2g = -0.5/(l_g*l_g)
    cdef double amp2h = amp_h*amp_h
    cdef double p2h = -0.5/(l_h*l_h)
    cdef double rf = 0.0

    cdef double wlA0 = 0.0
    cdef double wlA1 = 0.0

    cdef double rg = 0.0

    cdef double wlB0 = 0.0
    cdef double wlB1 = 0.0

    cdef double rh = 0.0

    cdef double wlC0 = 0.0
    cdef double wlC1 = 0.0
    cdef double cov = 0.0

    #Loop over all the non-diagonal indices
    for i in range(N):

      wlA0 = wlA[i]
      wlB0 = wlB[i]
      wlC0 = wlC[i]

      for j in range(i):

        # Just indexing each is very slow.
        wlA1 = wlA[j]
        wlB1 = wlB[j]
        wlC1 = wlC[j]

        # Initilize [i,j] and [j,i]
        # Calculate the distance in km/s
        # df = (wlA1 - wlA0) # km/s
        # rf = c_kms * df / (wlA0 + 0.5 * df)
        rf = c_kms * (wlA1 - wlA0) / wlA0
        #
        # dg = (wlB1 - wlB0) # km/s
        # rf = c_kms * dg / (wlB0 + 0.5 * dg)
        rg = c_kms * (wlB1 - wlB0) / wlB0
        #
        # dh = (wlC1 - wlC0) # km/s
        # rh = c_kms * dh / (wlC0 + 0.5 * dh)
        rh = c_kms * (wlC1 - wlC0) / wlC0

        cov = amp2f * exp(p2f * rf*rf) + amp2g * exp(p2g * rg*rg) + amp2h * exp(p2h * rh*rh)

        # Enter this on both sides of the diagonal, since the matrix is symmetric
        mat[i,j] = cov
        mat[j,i] = cov

    #Loop over all the diagonals, since the distance here is 0.
    for i in range(N):
        mat[i,i] = amp2f + amp2g + amp2h

    return mat

# def V12(np.ndarray[np.double_t, ndim=1] params, np.ndarray[np.double_t, ndim=2] gparams, np.ndarray[np.double_t, ndim=2] h2params, int m):
#     '''
#     Calculate V12 for a single parameter value.
#
#     Assumes kernel params coming in squared as h2params
#     '''
#     cdef int M = len(gparams)
#
#     mat = np.zeros((m * M, m), dtype=np.float64)
#     for block in range(m):
#         for row in range(M):
#             mat[block * M + row, block] = k(gparams[row], params, h2params[block])
#     return mat
#
# def V12m(np.ndarray[np.double_t, ndim=2] params, np.ndarray[np.double_t, ndim=2] gparams, np.ndarray[np.double_t, ndim=2] h2params, int m):
#     '''
#     Calculate V12 for a multiple parameter values.
#
#     Assumes kernel params coming in squared as h2params
#     '''
#     cdef int M = len(gparams)
#     cdef int npar = len(params)
#
#     mat = np.zeros((m * M, m * npar), dtype=np.float64)
#
#     # Going down the rows in "blocks" corresponding to the eigenspectra
#     for block in range(m):
#         # Now go down the rows within that block
#         for row in range(M):
#             ii = block * M + row
#             # Now go across the columns within that row
#             for ip in range(npar):
#                 jj = block + ip * m
#                 mat[ii, jj] = k(gparams[row], params[ip], h2params[block])
#     return mat
#
# def V22(np.ndarray[np.double_t, ndim=1] params, np.ndarray[np.double_t, ndim=2] h2params, int m):
#     '''
#     Create V22.
#
#     Assumes kernel parameters are coming in squared as h2params
#     '''
#     cdef int i = 0
#
#     mat = np.zeros((m, m))
#     for i in range(m):
#             mat[i,i] = k(params, params, h2params[i])
#     return mat
#
#
# def V22m(np.ndarray[np.double_t, ndim=2] params, np.ndarray[np.double_t, ndim=2] h2params, int m):
#     '''
#     Create V22 for a set of many parameters.
#
#     Assumes kernel parameters are coming in squared as h2params
#     '''
#     cdef int i = 0
#     cdef int npar = len(params)
#     cdef double cov = 0.0
#
#     mat = np.zeros((m * npar, m * npar))
#     for ixp in range(npar):
#         for i in range(m):
#             for iyp in range(npar):
#                 ii = ixp * m + i
#                 jj = iyp * m + i
#                 cov = k(params[ixp], params[iyp], h2params[i])
#                 mat[ii, jj] = cov
#                 mat[jj, ii] = cov
#     return mat


# # Routines for data covariance matrix generation

# @cython.boundscheck(False)
# def get_dense_C_wl(np.ndarray[np.double_t, ndim=1] wl, k_func):
#     '''
#     Fill out the covariance matrix using just input wavelengths. No assumptions about sorting.
#
#
#
#     :param wl: numpy wavelength vector
#
#     :param k_func: partial function to fill in matrix
#
#     :param max_r: (km/s) max velocity to fill out to
#     '''
#
#     cdef int N = len(wl)
#     cdef int i = 0
#     cdef int j = 0
#     cdef double cov = 0.0
#
#     #Find all the indices that are less than the radius
#     rr = np.abs(wl[:, np.newaxis] - wl[np.newaxis, :]) * C.c_kms/wl #Velocity space
#     flag = (rr < max_r)
#     indices = np.argwhere(flag)
#
#     #The matrix that we want to fill
#     mat = np.zeros((N,N))
#
#     #Loop over all the indices
#     for index in indices:
#         i,j = index
#         if j > i:
#             continue
#         else:
#             #Initilize [i,j] and [j,i]
#             cov = k_func(wl[i], wl[j])
#             mat[i,j] = cov
#             mat[j,i] = cov
#
#     return mat
#
# @cython.boundscheck(False)
# def get_dense_C_wl_t(np.ndarray[np.double_t, ndim=1] wl, k_func):
#     pass
#
#
# def make_k_func(par):
#     cdef double amp = 10**par.logAmp
#     cdef double l = par.l #Given in Km/s
#     cdef double r0 = 6.0 * l #Km/s
#     cdef double taper
#     regions = par.regions #could be None or a 2D array
#
#     cdef double a, mu, sigma, rx0, rx1, r_tap, r0_r
#
#     if regions is None:
#         # make a k_func that excludes regions and is faster
#         def k_func(wl0, wl1):
#             cdef double cov = 0.0
#
#             #Initialize the global covariance
#             cdef double r = C.c_kms/wl0 * math.fabs(wl0 - wl1) # Km/s
#             if r < r0:
#                 taper = (0.5 + 0.5 * math.cos(np.pi * r/r0))
#                 cov = taper * amp*amp * (1 + math.sqrt(3) * r/l) * math.exp(-math.sqrt(3.) * r/l)
#
#             return cov
#     else:
#         # make a k_func which includes regions
#         def k_func(wl0, wl1):
#             cdef double cov = 0.0
#
#             #Initialize the global covariance
#             cdef double r = C.c_kms/wl0 * math.fabs(wl0 - wl1) # Km/s
#             if r < r0:
#                 taper = (0.5 + 0.5 * math.cos(np.pi * r/r0))
#                 cov = taper * amp*amp * (1 + math.sqrt(3) * r/l) * math.exp(-math.sqrt(3.) * r/l)
#
#             #If covered by a region, instantiate
#             for row in regions:
#                 a = 10**row[0]
#                 mu = row[1]
#                 sigma = row[2]
#
#                 rx0 = C.c_kms / mu * math.fabs(wl0 - mu)
#                 rx1 = C.c_kms / mu * math.fabs(wl1 - mu)
#                 r_tap = rx0 if rx0 > rx1 else rx1 # choose the larger distance
#                 r0_r = 4.0 * sigma # where the kernel goes to 0
#
#                 if r_tap < r0_r:
#                     taper = (0.5 + 0.5 * math.cos(np.pi * r_tap/r0_r))
#                     cov += taper * a*a * math.exp(-0.5 * (C.c_kms * C.c_kms) / (mu * mu) * ((wl0 - mu)*(wl0 - mu) + (wl1 - mu)*(wl1 - mu))/(sigma * sigma))
#             return cov
#
#     return k_func
