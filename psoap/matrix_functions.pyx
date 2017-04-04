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
