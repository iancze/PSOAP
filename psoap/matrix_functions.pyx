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
cdef double c_kms2 = (2.99792458e5)**2 #km s^-1

@cython.boundscheck(False)
@cython.wraparound(False)
def fill_V11_f(np.ndarray[np.double_t, ndim=2] mat, np.ndarray[np.double_t, ndim=1] lwl_f, double amp_f, double l_f):

    cdef int N = len(mat)
    cdef int i = 0
    cdef int j = 0

    # Compute the squared values of these to save time within the for-loop
    cdef double amp2f = amp_f*amp_f
    cdef double p2f = -0.5 * c_kms2/(l_f*l_f)

    # Temporary distance holders
    cdef double rf = 0.0

    # Temporary wavelength holder
    cdef double lwl_f0 = 0.0

    cdef double cov = 0.0

    #Loop over all the non-diagonal indices
    for i in range(N):

      lwl_f0 = lwl_f[i]

      for j in range(i):

        # Calculate the distance in km/s
        rf = lwl_f[j] - lwl_f0

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
def fill_V12_f(np.ndarray[np.double_t, ndim=2] mat, np.ndarray[np.double_t, ndim=1] lwl_f, np.ndarray[np.double_t, ndim=1] lwl_predict, double amp_f, double l_f):

    cdef int M = len(lwl_f)
    cdef int N = len(lwl_predict)
    cdef int i = 0
    cdef int j = 0

    # Compute the squared values of these to save time within the for-loop
    cdef double amp2f = amp_f*amp_f
    cdef double p2f = -0.5 * c_kms2/(l_f*l_f)

    # Temporary distance holders
    cdef double rf = 0.0

    # Temporary wavelength holder
    cdef double lwl_f0 = 0.0

    cdef double cov = 0.0

    #Loop over all the non-diagonal indices
    for i in range(M):

      lwl_f0 = lwl_f[i]

      for j in range(N):

        rf = lwl_predict[j] - lwl_f0

        cov = amp2f * exp(p2f * rf * rf)

        # Enter this on just one side the diagonal, since the matrix is not symmetric
        mat[i,j] = cov

    # no return


@cython.boundscheck(False)
@cython.wraparound(False)
def fill_V11_f_g(np.ndarray[np.double_t, ndim=2] mat, np.ndarray[np.double_t, ndim=1] lwl_f, np.ndarray[np.double_t, ndim=1] lwl_g, double amp_f, double l_f, double amp_g, double l_g):

    cdef int N = len(mat)
    cdef int i = 0
    cdef int j = 0

    # Compute the squared values of these to save time within the for-loop
    cdef double amp2f = amp_f*amp_f
    cdef double p2f = -0.5*c_kms2/(l_f*l_f)

    cdef double amp2g = amp_g*amp_g
    cdef double p2g = -0.5*c_kms2/(l_g*l_g)

    # Temporary distance holders
    cdef double rf = 0.0
    cdef double rg = 0.0

    # Temporary wavelength holders
    cdef double lwl_f0 = 0.0
    cdef double lwl_g0 = 0.0

    cdef double cov = 0.0

    #Loop over all the non-diagonal indices
    for i in range(N):

      lwl_f0 = lwl_f[i]
      lwl_g0 = lwl_g[i]

      for j in range(i):

        # Calculate the distance in km/s
        rf = lwl_f[j] - lwl_f0
        rg = lwl_g[j] - lwl_g0

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
def fill_V11_f_g_h(np.ndarray[np.double_t, ndim=2] mat, np.ndarray[np.double_t, ndim=1] lwl_f, np.ndarray[np.double_t, ndim=1] lwl_g, np.ndarray[np.double_t, ndim=1] lwl_h, double amp_f, double l_f, double amp_g, double l_g, double amp_h, double l_h):

    cdef int N = len(mat)
    cdef int i = 0
    cdef int j = 0

    # Compute the squared values of these to save time within the for-loop
    cdef double amp2f = amp_f*amp_f
    cdef double p2f = -0.5*c_kms2/(l_f*l_f)

    cdef double amp2g = amp_g*amp_g
    cdef double p2g = -0.5*c_kms2/(l_g*l_g)

    cdef double amp2h = amp_h*amp_h
    cdef double p2h = -0.5*c_kms2/(l_h*l_h)

    # Temporary distance holders
    cdef double rf = 0.0
    cdef double rg = 0.0
    cdef double rh = 0.0

    # Temporary wavelength holders
    cdef double lwl_f0 = 0.0
    cdef double lwl_g0 = 0.0
    cdef double lwl_h0 = 0.0

    cdef double cov = 0.0

    #Loop over all the non-diagonal indices
    for i in range(N):

      lwl_f0 = lwl_f[i]
      lwl_g0 = lwl_g[i]
      lwl_h0 = lwl_h[i]

      for j in range(i):

        # Calculate the distance in km/s
        rf = lwl_f[j] - lwl_f0
        rg = lwl_g[j] - lwl_g0
        rh = lwl_h[j] - lwl_h0

        cov = amp2f * exp(p2f * rf*rf) + amp2g * exp(p2g * rg*rg) + amp2h * exp(p2h * rh*rh)

        # Enter this on both sides of the diagonal, since the matrix is symmetric
        mat[i,j] = cov
        mat[j,i] = cov

    #Loop over all the diagonals, since the distance here is 0.
    for i in range(N):
        mat[i,i] = amp2f + amp2g + amp2h
