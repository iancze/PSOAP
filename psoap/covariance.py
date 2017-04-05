import numpy as np
from numpy.polynomial import Chebyshev as Ch

from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize

import psoap
from psoap import constants as C
from psoap import matrix_functions
from psoap.data import redshift

# Covariance Kernel Functions
# To be later fed into np.from_function routines
# These functions have been superseeded by those in matrix_functions.pyx, however they are left
# here to serve as checks in the tests module.
def sqexp_func(x0i, x1i, x0v=None, x1v=None, amp2=None, l2=None):
    '''amp is the amplitude
    lv is the scale length in km/s.

    This funny signature with x0i, x1i, x0v, etc... is so that this function can be later used
    in the np.from_function routine to create a matrix.'''

    x0 = x0v[x0i]
    x1 = x1v[x1i]

    # Calculate the distance in km/s
    r = 0.5 * C.c_kms * (x0v[x0i] - x1v[x1i])/(x0v[x0i] + x1v[x1i]) # Km/s
    # r = C.c_kms/x0v[x0i] * (x0v[x0i] - x1v[x1i]) # Km/s

    return amp2 * np.exp(-0.5 * r**2 / l2)


def sqexp_func_three(x0i, x1i, f0v=None, f1v=None, g0v=None, g1v=None, h0v=None, h1v=None, amp2f=None, l2f=None, amp2g=None, l2g=None, amp2h=None, l2h=None):
    '''
    f0v is the vector of (shifted) wavelengths for the f spectrum
    f1v is the othre vector

    g0v is the vector for g spectrum,

    h0v, etc.
    '''

    # Calculate the distance in km/s
    rf = C.c_kms/f0v[x0i] * (f0v[x0i] - f1v[x1i]) # Km/s
    rg = C.c_kms/g0v[x0i] * (g0v[x0i] - g1v[x1i]) # Km/s
    rh = C.c_kms/h0v[x0i] * (h0v[x0i] - h1v[x1i]) # Km/s

    return amp2f * np.exp(-0.5 * rf**2/l2f) + amp2g * np.exp(-0.5 * rg**2/l2g) + amp2h * np.exp(-0.5 * rh**2/l2h)

# These functions deliver a filled covariance matrix (part of V11, V12, V22, etc) from the squared exponential kernels
def get_C(wl0, wl1, amp, l):
    '''Returns a covariance matrix that is (len(wl0), len(wl1))'''

    mat = np.fromfunction(sqexp_func, (len(wl0),len(wl1)), x0v=wl0, x1v=wl1, amp2=amp**2, l2=l**2, dtype=np.int)

    return mat

def get_C_three(wlA0, wlA1, wlB0, wlB1, wlC0, wlC1, amp_f, l_f, amp_g, l_g, amp_h, l_h):
    '''
    Returns a covariance matrix as the sum of three Gaussian processes.
    '''

    mat = np.fromfunction(sqexp_func_three, (len(wlA0), len(wlA1)), f0v=wlA0, f1v=wlA1, g0v=wlB0, g1v=wlB1, h0v=wlC0, h1v=wlC1, amp2f=amp_f**2, l2f=l_f**2, amp2g=amp_g**2, l2g=l_g**2, amp2h=amp_h**2, l2h=l_h**2, dtype=np.int)

    return mat

# These kernel functions are only useful for visualizing the time dependence
def rt_func(x0i, x1i, d0v=None, d1v=None, tau=None):
    d0 = d0v[x0i]
    d1 = d1v[x1i]

    rt = np.abs(d1 - d0) #days
    return rt

# These kernel functions are only useful for visualizing the time dependence
def rt_exp_func(x0i, x1i, d0v=None, d1v=None, tau=None):
    d0 = d0v[x0i]
    d1 = d1v[x1i]

    rt = np.abs(d1 - d0) #days

    return np.exp(-0.5 * (rt**2/tau**2))

# These functions return a filled covariance matrix for the date distance kernels
def get_C_rt(d0, d1, tau):
    mat = np.fromfunction(rt_func, (len(d0), len(d1)), d0v=d0, d1v=d1, tau=tau, dtype=np.int)
    return mat

def get_C_exp_rt(d0, d1, tau):
    mat = np.fromfunction(rt_exp_func, (len(d0), len(d1)), d0v=d0, d1v=d1, tau=tau, dtype=np.int)
    return mat


def get_V11(wl_known, sigma_known, amp_f, l_f):
    '''
    wl_known is a 1D array.
    sigma_known is a 1D array.
    amp and lv are scalar values.
    '''

    V11 = get_C(wl_known, wl_known, amp_f, l_f) + sigma_known**2 * np.eye(len(wl_known))

    return V11


def get_V12(wl_known, wl_predict, amp_f, l_f):

    V12 = get_C(wl_known, wl_predict, amp_f, l_f)

    return V12

def get_V22(wl_predict, amp_f, l_f):

    # Might need a small offset for numerical stability
    V22 = get_C(wl_predict, wl_predict, amp_f, l_f) + 1e-5 * np.eye(len(wl_predict)) # Add a small nugget

    return V22


def predict_f(wl_known, fl_known, sigma_known, wl_predict, amp_f, l_f, mu_GP=1.0):

    '''wl_known are known wavelengths.
    wl_predict are the prediction wavelengths.
    Assumes all inputs are 1D arrays.'''

    # determine V11, V12, V21, and V22
    M = len(wl_known)
    V11 = np.empty((M, M), dtype=np.float64)
    matrix_functions.fill_V11_f(V11, wl_known, amp_f, l_f)
    # V11[np.diag_indices_from(V11)] += sigma_known**2
    V11 = V11 + sigma_known**2 * np.eye(M)

    N = len(wl_predict)
    V12 = np.empty((M, N), dtype=np.float64)
    matrix_functions.fill_V12_f(V12, wl_known, wl_predict, amp_f, l_f)

    V22 = np.empty((N, N), dtype=np.float64)
    # V22 is the covariance between the prediction wavelengths
    # The routine to fill V11 is the same as V22
    matrix_functions.fill_V11_f(V22, wl_predict, amp_f, l_f)

    # Find V11^{-1}
    factor, flag = cho_factor(V11)

    mu = mu_GP + np.dot(V12.T, cho_solve((factor, flag), (fl_known - mu_GP)))

    Sigma = V22 - np.dot(V12.T, cho_solve((factor, flag), V12))

    return (mu, Sigma)


def predict_python(wl_known, fl_known, sigma_known, wl_predict, amp_f, l_f, mu_GP=1.0):

    '''wl_known are known wavelengths.
    wl_predict are the prediction wavelengths.'''

    # determine V11, V12, V21, and V22
    V11 = get_V11(wl_known, sigma_known, amp_f, l_f)

    # V12 is covariance between data wavelengths and prediction wavelengths
    V12 = get_V12(wl_known, wl_predict, amp_f, l_f)

    # V22 is the covariance between the prediction wavelengths
    V22 = get_V22(wl_predict, amp_f, l_f)

    # Find V11^{-1}
    factor, flag = cho_factor(V11)

    mu = mu_GP + np.dot(V12.T, cho_solve((factor, flag), (fl_known - mu_GP)))

    Sigma = V22 - np.dot(V12.T, cho_solve((factor, flag), V12))

    return (mu, Sigma)


def predict_f_g(wl_f, wl_g, fl_fg, sigma_fg, wl_f_predict, wl_g_predict, mu_f, amp_f, l_f, mu_g, amp_g, l_g):
    '''
    Given that f + g is the flux that we're modeling, jointly predict the components.
    '''
    # Assert that wl_f and wl_g are the same length
    assert len(wl_f) == len(wl_g), "Input wavelengths must be the same length."
    n_pix = len(wl_f)

    assert len(wl_f_predict) == len(wl_g_predict), "Prediction wavelengths must be the same length."
    n_pix_predict = len(wl_f_predict)

    # Convert mu constants into vectors
    mu_f = mu_f * np.ones(n_pix_predict)
    mu_g = mu_g * np.ones(n_pix_predict)

    # Cat these into a single vector
    mu_cat = np.hstack((mu_f, mu_g))


    # Create the matrices for the input data
    # print("allocating V11_f, V11_g", n_pix, n_pix)
    V11_f = np.empty((n_pix, n_pix), dtype=np.float)
    V11_g = np.empty((n_pix, n_pix), dtype=np.float)

    # print("filling V11_f, V11_g", n_pix, n_pix)
    matrix_functions.fill_V11_f(V11_f, wl_f, amp_f, l_f)
    matrix_functions.fill_V11_f(V11_g, wl_g, amp_g, l_g)

    B = V11_f + V11_g
    B[np.diag_indices_from(B)] += sigma_fg**2

    # print("factoring sum")
    factor, flag = cho_factor(B)

    # print("Allocating prediction matrices")
    # Now create separate matrices for the prediction
    V11_f_predict = np.empty((n_pix_predict, n_pix_predict), dtype=np.float)
    V11_g_predict = np.empty((n_pix_predict, n_pix_predict), dtype=np.float)

    # print("Filling prediction matrices")
    matrix_functions.fill_V11_f(V11_f_predict, wl_f_predict, amp_f, l_f)
    matrix_functions.fill_V11_f(V11_g_predict, wl_g_predict, amp_g, l_g)

    # print("Allocating A")
    zeros = np.zeros((n_pix_predict, n_pix_predict))
    A = np.vstack((np.hstack([V11_f_predict, zeros]), np.hstack([zeros, V11_g_predict])))
    # A[np.diag_indices_from(A)] += 1e-4 # Add a small nugget term

    # print("Allocating cross-matrices")
    # C is now the cross-matrices between the predicted wavelengths and the data wavelengths
    V12_f = np.empty((n_pix_predict, n_pix), dtype=np.float)
    V12_g = np.empty((n_pix_predict, n_pix), dtype=np.float)

    # print("Filling cross-matrices")
    matrix_functions.fill_V12_f(V12_f, wl_f_predict, wl_f, amp_f, l_f)
    matrix_functions.fill_V12_f(V12_g, wl_g_predict, wl_g, amp_g, l_g)

    C = np.vstack((V12_f, V12_g))

    # print("Shapes.\n fl: {} \n A: {} \n B: {} \n C: {}".format(fl_fg.shape, A.shape, B.shape, C.shape))

    # print("Sloving for mu, sigma")
    # the 1.0 signifies that mu_f + mu_g = mu_fg = 1
    mu = mu_cat + np.dot(C, cho_solve((factor, flag), fl_fg - 1.0))
    Sigma = A - np.dot(C, cho_solve((factor, flag), C.T))

    return mu, Sigma


def predict_f_g_sum(wl_f, wl_g, fl_fg, sigma_fg, wl_f_predict, wl_g_predict, mu_fg, amp_f, l_f, amp_g, l_g):

    # Assert that wl_f and wl_g are the same length
    assert len(wl_f) == len(wl_g), "Input wavelengths must be the same length."

    M = len(wl_f_predict)
    N = len(wl_f)

    V11_f = np.empty((M, M), dtype=np.float)
    V11_g = np.empty((M, M), dtype=np.float)

    matrix_functions.fill_V11_f(V11_f, wl_f_predict, amp_f, l_f)
    matrix_functions.fill_V11_f(V11_g, wl_g_predict, amp_g, l_g)
    V11 = V11_f + V11_g
    V11[np.diag_indices_from(V11)] += 1e-8

    V12_f = np.empty((M, N), dtype=np.float64)
    V12_g = np.empty((M, N), dtype=np.float64)
    matrix_functions.fill_V12_f(V12_f, wl_f_predict, wl_f, amp_f, l_f)
    matrix_functions.fill_V12_f(V12_g, wl_g_predict, wl_g, amp_g, l_g)
    V12 = V12_f + V12_g

    V22_f = np.empty((N,N), dtype=np.float)
    V22_g = np.empty((N,N), dtype=np.float)

    # It's a square matrix, so we can just reuse fill_V11_f
    matrix_functions.fill_V11_f(V22_f, wl_f, amp_f, l_f)
    matrix_functions.fill_V11_f(V22_g, wl_g, amp_g, l_g)
    V22 = V22_f + V22_g
    V22[np.diag_indices_from(V22)] += sigma_fg**2

    factor, flag = cho_factor(V22)

    # print("Shapes.\n fl: {} \n V11: {} \n V22: {} \n V12: {}".format(fl_fg.shape, V11.shape, V22.shape, V12.shape))


    mu = mu_fg + np.dot(V12, cho_solve((factor, flag), (fl_fg - 1.0)))
    Sigma = V11 - np.dot(V12, cho_solve((factor, flag), V12.T))

    return mu, Sigma


def predict_f_g_h(wl_f, wl_g, wl_h, fl_fgh, sigma_fgh, wl_f_predict, wl_g_predict, wl_h_predict, mu_f, mu_g, mu_h, amp_f, l_f, amp_g, l_g, amp_h, l_h):
    '''
    Given that f + g + h is the flux that we're modeling, jointly predict the components.
    '''
    # Assert that wl_f and wl_g are the same length
    assert len(wl_f) == len(wl_g), "Input wavelengths must be the same length."
    assert len(wl_f) == len(wl_h), "Input wavelengths must be the same length."
    n_pix = len(wl_f)

    assert len(wl_f_predict) == len(wl_g_predict), "Prediction wavelengths must be the same length."
    assert len(wl_f_predict) == len(wl_h_predict), "Prediction wavelengths must be the same length."
    n_pix_predict = len(wl_f_predict)

    # Convert mu constants into vectors
    mu_f = mu_f * np.ones(n_pix_predict)
    mu_g = mu_g * np.ones(n_pix_predict)
    mu_h = mu_h * np.ones(n_pix_predict)

    # Cat these into a single vector
    mu_cat = np.hstack((mu_f, mu_g, mu_h))

    V11_f = np.empty((n_pix, n_pix), dtype=np.float)
    V11_g = np.empty((n_pix, n_pix), dtype=np.float)
    V11_h = np.empty((n_pix, n_pix), dtype=np.float)

    matrix_functions.fill_V11_f(V11_f, wl_f, amp_f, l_f)
    matrix_functions.fill_V11_f(V11_g, wl_g, amp_g, l_g)
    matrix_functions.fill_V11_f(V11_h, wl_h, amp_h, l_h)

    B = V11_f + V11_g + V11_h
    B[np.diag_indices_from(B)] += sigma_fgh**2

    factor, flag = cho_factor(B)

    # Now create separate matrices for the prediction
    V11_f_predict = np.empty((n_pix_predict, n_pix_predict), dtype=np.float)
    V11_g_predict = np.empty((n_pix_predict, n_pix_predict), dtype=np.float)
    V11_h_predict = np.empty((n_pix_predict, n_pix_predict), dtype=np.float)

    # Fill the prediction matrices
    matrix_functions.fill_V11_f(V11_f_predict, wl_f_predict, amp_f, l_f)
    matrix_functions.fill_V11_f(V11_g_predict, wl_g_predict, amp_g, l_g)
    matrix_functions.fill_V11_f(V11_h_predict, wl_h_predict, amp_h, l_h)

    zeros = np.zeros((n_pix_predict, n_pix_predict))

    A = np.vstack((np.hstack([V11_f_predict, zeros, zeros]), np.hstack([zeros, V11_g_predict, zeros]), np.hstack([zeros, zeros, V11_h_predict])))

    V12_f = np.empty((n_pix_predict, n_pix), dtype=np.float)
    V12_g = np.empty((n_pix_predict, n_pix), dtype=np.float)
    V12_h = np.empty((n_pix_predict, n_pix), dtype=np.float)

    matrix_functions.fill_V12_f(V12_f, wl_f_predict, wl_f, amp_f, l_f)
    matrix_functions.fill_V12_f(V12_g, wl_g_predict, wl_g, amp_g, l_g)
    matrix_functions.fill_V12_f(V12_h, wl_h_predict, wl_h, amp_h, l_h)

    C = np.vstack((V12_f, V12_g, V12_h))

    mu = mu_cat + np.dot(C, cho_solve((factor, flag), fl_fgh - 1.0))
    Sigma = A - np.dot(C, cho_solve((factor, flag), C.T))

    return mu, Sigma

def predict_f_g_h_sum(wl_f, wl_g, wl_h, fl_fgh, sigma_fgh, wl_f_predict, wl_g_predict, wl_h_predict, mu_fgh, amp_f, l_f, amp_g, l_g, amp_h, l_h):
    '''
    Given that f + g + h is the flux that we're modeling, predict the joint sum.
    '''
    # Assert that wl_f and wl_g are the same length
    assert len(wl_f) == len(wl_g), "Input wavelengths must be the same length."

    M = len(wl_f_predict)
    N = len(wl_f)

    V11_f = np.empty((M, M), dtype=np.float)
    V11_g = np.empty((M, M), dtype=np.float)
    V11_h = np.empty((M, M), dtype=np.float)

    matrix_functions.fill_V11_f(V11_f, wl_f_predict, amp_f, l_f)
    matrix_functions.fill_V11_f(V11_g, wl_g_predict, amp_g, l_g)
    matrix_functions.fill_V11_f(V11_h, wl_h_predict, amp_h, l_h)
    V11 = V11_f + V11_g + V11_h
    # V11[np.diag_indices_from(V11)] += 1e-5 # small nugget term

    V12_f = np.empty((M, N), dtype=np.float64)
    V12_g = np.empty((M, N), dtype=np.float64)
    V12_h = np.empty((M, N), dtype=np.float64)
    matrix_functions.fill_V12_f(V12_f, wl_f_predict, wl_f, amp_f, l_f)
    matrix_functions.fill_V12_f(V12_g, wl_g_predict, wl_g, amp_g, l_g)
    matrix_functions.fill_V12_f(V12_h, wl_h_predict, wl_h, amp_h, l_h)
    V12 = V12_f + V12_g + V12_h

    V22_f = np.empty((N,N), dtype=np.float)
    V22_g = np.empty((N,N), dtype=np.float)
    V22_h = np.empty((N,N), dtype=np.float)

    # It's a square matrix, so we can just reuse fil_V11_f
    matrix_functions.fill_V11_f(V22_f, wl_f, amp_f, l_f)
    matrix_functions.fill_V11_f(V22_g, wl_g, amp_g, l_g)
    matrix_functions.fill_V11_f(V22_h, wl_h, amp_h, l_h)
    V22 = V22_f + V22_g + V22_h
    V22[np.diag_indices_from(V22)] += sigma_fgh**2

    factor, flag = cho_factor(V22)

    mu = mu_fgh + np.dot(V12.T, cho_solve((factor, flag), (fl_fgh - mu_fgh)))
    Sigma = V11 - np.dot(V12, cho_solve((factor, flag), V12.T))

    return mu, Sigma

def lnlike_f(V11, wl_f, fl, sigma, amp_f, l_f, mu_GP=1.):
    '''
    V11 is a matrix to be allocated.
    wl_known, fl_known, and sigma_known are flattened 1D arrays.

    '''
    if  amp_f < 0.0 or l_f < 0.0:
        return -np.inf

    # Fill the matrix using fast cython routine.
    matrix_functions.fill_V11_f(V11, wl_f, amp_f, l_f)
    V11[np.diag_indices_from(V11)] += sigma**2

    try:
        factor, flag = cho_factor(V11)
    except np.linalg.linalg.LinAlgError:
        return -np.inf

    logdet = np.sum(2 * np.log((np.diag(factor))))

    return -0.5 * (np.dot((fl - mu_GP).T, cho_solve((factor, flag), (fl - mu_GP))) + logdet)

def lnlike_f_g(V11, wl_f, wl_g, fl, sigma, amp_f, l_f, amp_g, l_g, mu_GP=1.):
    '''
    V11 is a matrix to be allocated.
    wl_known, fl_known, and sigma_known are flattened 1D arrays.

    '''
    if  amp_f < 0.0 or l_f < 0.0 or amp_g < 0.0 or l_g < 0.0:
        return -np.inf

    # Fill the matrix using fast cython routine.
    matrix_functions.fill_V11_f_g(V11, wl_f, wl_g, amp_f, l_f, amp_g, l_g)
    V11[np.diag_indices_from(V11)] += sigma**2

    try:
        factor, flag = cho_factor(V11)
    except np.linalg.linalg.LinAlgError:
        return -np.inf

    logdet = np.sum(2 * np.log((np.diag(factor))))

    return -0.5 * (np.dot((fl - mu_GP).T, cho_solve((factor, flag), (fl - mu_GP))) + logdet)

def lnlike_f_g_h(V11, wl_f, wl_g, wl_h, fl, sigma, amp_f, l_f, amp_g, l_g, amp_h, l_h, mu_GP=1.):
    '''
    V11 is a matrix to be allocated.
    wl_known, fl_known, and sigma_known are flattened 1D arrays.

    '''
    if  amp_f < 0.0 or l_f < 0.0 or amp_g < 0.0 or l_g < 0.0 or amp_h < 0.0 or l_h < 0.0:
        return -np.inf

    # Fill the matrix using fast cython routine.
    matrix_functions.fill_V11_f_g_h(V11, wl_f, wl_g, wl_h, amp_f, l_f, amp_g, l_g, amp_h, l_h)
    V11[np.diag_indices_from(V11)] += sigma**2

    try:
        factor, flag = cho_factor(V11)
    except np.linalg.linalg.LinAlgError:
        return -np.inf

    logdet = np.sum(2 * np.log((np.diag(factor))))

    return -0.5 * (np.dot((fl - mu_GP).T, cho_solve((factor, flag), (fl - mu_GP))) + logdet)

def optimize_GP_f(wl_known, fl_known, sigma_known, amp_f, l_f, mu_GP=1.0):
    '''
    Optimize the GP hyperparameters for the given slice of data. Amp and lv are starting guesses.
    '''
    N = len(wl_known)
    V11 = np.empty((N,N), dtype=np.float64)

    def func(x):
        try:
            a, l = x
            return -lnlike_f(V11, wl_known, fl_known, sigma_known, a, l, mu_GP)
        except np.linalg.linalg.LinAlgError:
            return np.inf

    ans = minimize(func, np.array([amp_f, l_f]), method="Nelder-Mead")

    return ans["x"]

def optimize_epoch_velocity(wl_epoch, fl_epoch, sigma_epoch, wl_fixed, fl_fixed, sigma_fixed, amp_f, l_f, mu_GP=1.0):
    '''
    Optimize the wavelengths of the chosen epoch relative to the fixed wavelengths. Identify the velocity required to redshift the chosen epoch.
    '''

    fl = np.concatenate((fl_epoch, fl_fixed)).flatten()
    sigma = np.concatenate((sigma_epoch, sigma_fixed)).flatten()

    def func(v):
        try:
            # Doppler shift the input wl_epoch
            wl_shift = redshift(wl_epoch, v)

            # Reconcatenate spectra into 1D array
            wl = np.concatenate((wl_shift, wl_fixed)).flatten()
            return -lnlike_GP(wl, fl, sigma, amp_f, l_f, mu_GP)
        except np.linalg.linalg.LinAlgError:
            return np.inf

    ans = minimize(func, np.array([0.0]), method="Nelder-Mead")

    # The velocity returned is the amount that was required to redshift wl_epoch to line up with wl_fixed.

    return ans["x"][0]

# New (as of 4/4/17) routine to incorporate more complex covariance matrices in the optimization routine, which can incorporate orbital motion.
def optimize_calibration(wl0, wl1, wl_cal, fl_cal, fl_fixed, A, B, C, order=1, mu_GP=1.0):
    '''
    Determine the calibration parameters for this epoch of observations.

    wl0, wl1: set the points for the Chebyshev.

    This is a more general method than optimize_calibration_static, since it allows arbitrary covariance matrices, which should be used when there is orbital motion.

    wl_cal: the wavelengths corresponding to the epoch we want to calibrate
    fl_cal: the fluxes corresponding to the epoch we want to calibrate

    fl_fixed: the remaining epochs of data to calibrate in reference to.

    A : matrix_functions.fill_V11_f(A, wl_cal, amp, l_f) with sigma_cal already added to the diagonal
    B : matrix_functions.fill_V11_f(B, wl_fixed, amp, l_f) with sigma_fixed already added to the diagonal
    C : matrix_functions.fill_V12_f(C, wl_cal, wl_fixed, amp, l_f) cross matrix (with no sigma added, since these are independent measurements).

    order: the degree polynomial to use. order = 1 is a line, order = 2 is a line + parabola

    Assumes that covariance matrices are appropriately filled out.
    '''

    # basically, assume that A, B, and C are already filled out.
    # the only thing this routine needs to do is fill out the Q matrix

    # Get a clean set of the Chebyshev polynomials evaluated on the input wavelengths
    T = []
    for i in range(0, order + 1):
        coeff = [0 for j in range(i)] + [1]
        Chtemp = Ch(coeff, domain=[wl0, wl1])
        Ttemp = Chtemp(wl_cal)
        T += [Ttemp]

    T = np.array(T)

    D = fl_cal[:,np.newaxis] * T.T


    # Solve for the calibration coefficients c0, c1, ...

    # Find B^{-1}, fl_prime, and C_prime
    try:
        B_cho = cho_factor(B)
    except np.linalg.linalg.LinAlgError:
        print("Failed to solve matrix inverse. Calibration not valid.")
        raise

    fl_prime = mu_GP + np.dot(C, cho_solve(B_cho, (fl_fixed.flatten() - mu_GP)))
    C_prime = A - np.dot(C, cho_solve(B_cho, C.T))

    # Find {C^\prime}^{-1}
    CP_cho = cho_factor(C_prime)

    # Invert the least squares problem
    left = np.dot(D.T, cho_solve(CP_cho, D))
    right = np.dot(D.T, cho_solve(CP_cho, fl_prime))

    left_cho = cho_factor(left)

    # the coefficents, X = [c0, c1]
    X = cho_solve(left_cho, right)

    # Apply the correction
    fl_cor = np.dot(D, X)

    # Return both the corrected flux and the coefficients, in case we want to log them,
    # or apply the correction later.
    return fl_cor, X


# Previous optimize calibration routine that assumed that relative velocities between all epochs were zero.
def optimize_calibration_static(wl0, wl1, wl_cal, fl_cal, sigma_cal, wl_fixed, fl_fixed, sigma_fixed, amp, l_f, order=1, mu_GP=1.0):
    '''
    Determine the calibration parameters for this epoch of observations.
    Assumes all wl, fl arrays are 1D.

    order is the Chebyshev order to use.

    returns a corrected fl_cal, as well as c0 and c1
    '''

    N_A = len(wl_cal)
    A = np.empty((N_A, N_A), dtype=np.float64)

    N_B = len(wl_fixed)
    B = np.empty((N_B, N_B), dtype=np.float64)

    C = np.empty((N_A, N_B), dtype=np.float64)

    matrix_functions.fill_V11_f(A, wl_cal, amp, l_f)
    matrix_functions.fill_V11_f(B, wl_fixed, amp, l_f)
    matrix_functions.fill_V12_f(C, wl_cal, wl_fixed, amp, l_f)

    # Add in sigmas
    A[np.diag_indices_from(A)] += sigma_cal**2
    B[np.diag_indices_from(B)] += sigma_fixed**2


    # Get a clean set of the Chebyshev polynomials evaluated on the input wavelengths
    T = []
    for i in range(0, order + 1):
        coeff = [0 for j in range(i)] + [1]
        Chtemp = Ch(coeff, domain=[wl0, wl1])
        Ttemp = Chtemp(wl_cal)
        T += [Ttemp]

    T = np.array(T)

    D = fl_cal[:,np.newaxis] * T.T


    # Solve for the calibration coefficients c0, c1

    # Find B^{-1}, fl_prime, and C_prime
    try:
        B_cho = cho_factor(B)
    except np.linalg.linalg.LinAlgError:
        print("Failed to solve matrix inverse. Calibration not valid.")
        raise

    fl_prime = mu_GP + np.dot(C, cho_solve(B_cho, (fl_fixed.flatten() - mu_GP)))
    C_prime = A - np.dot(C, cho_solve(B_cho, C.T))

    # Find {C^\prime}^{-1}
    CP_cho = cho_factor(C_prime)

    # Invert the least squares problem
    left = np.dot(D.T, cho_solve(CP_cho, D))
    right = np.dot(D.T, cho_solve(CP_cho, fl_prime))

    left_cho = cho_factor(left)

    # the coefficents, X = [c0, c1]
    X = cho_solve(left_cho, right)

    # Apply the correction
    fl_cor = np.dot(D, X)

    return fl_cor, X


def cycle_calibration(wl, fl, sigma, amp_f, l_f, ncycles, order=1, limit_array=3, mu_GP=1.0, soften=1.0):
    '''
    Given a chunk of spectra, cycle n_cycles amongst all spectra and return the spectra with inferred calibration adjustments.

    order : what order of Chebyshev polynomials to use. 1st order = line.

    Only use `limit_array` number of spectra to save memory.
    '''
    wl0 = np.min(wl)
    wl1 = np.max(wl)

    fl_out = np.copy(fl)

    # Soften the sigmas a little bit
    sigma = soften * sigma

    n_epochs = len(wl)

    for cycle in range(ncycles):
        for i in range(n_epochs):
            wl_tweak = wl[i]
            fl_tweak = fl_out[i]
            sigma_tweak = sigma[i]

            # Temporary arrays without the epoch we just chose
            wl_remain = np.delete(wl, i, axis=0)[0:limit_array]
            fl_remain = np.delete(fl_out, i, axis=0)[0:limit_array]
            sigma_remain = np.delete(sigma, i, axis=0)[0:limit_array]

            # optimize the calibration of "tweak" with respect to all other orders
            fl_cor, X = optimize_calibration(wl0, wl1, wl_tweak, fl_tweak, sigma_tweak, wl_remain.flatten(), fl_remain.flatten(), sigma_remain.flatten(), amp_f, l_f, order=order, mu_GP=mu_GP)

            # replace this epoch with the corrected fluxes
            fl_out[i] = fl_cor

    return fl_out

def cycle_calibration_chunk(chunk, amp_f, l_f, n_cycles, order=1, limit_array=3, mu_GP=1.0, soften=1.0):
    '''
    Do the calibration on a chunk at at time, incorporating the masks.
    '''

    # Figure out the min and max wavelengths to set the domain of the Chebyshevs
    wl0 = np.min(chunk.wl)
    wl1 = np.max(chunk.wl)

    # Temporary copy, so that we can do multiple cycle corrections.
    fl_out = np.copy(chunk.fl)

    # Soften the sigmas a little bit to prevent inversion errors.
    sigma = soften * chunk.sigma

    for cycle in range(n_cycles):
        for i in range(chunk.n_epochs):
            wl_tweak = chunk.wl[i]
            fl_tweak = fl_out[i]
            sigma_tweak = chunk.sigma[i]
            mask_tweak = chunk.mask[i]

            # Temporary arrays without the epoch we just chose
            wl_remain = np.delete(chunk.wl, i, axis=0)[0:limit_array]
            fl_remain = np.delete(fl_out, i, axis=0)[0:limit_array]
            sigma_remain = np.delete(chunk.sigma, i, axis=0)[0:limit_array]
            mask_remain = np.delete(chunk.mask, i, axis=0)[0:limit_array]

            # optimize the calibration of "tweak" with respect to all other orders
            fl_cor, X = optimize_calibration(wl0, wl1, wl_tweak[mask_tweak], fl_tweak[mask_tweak], sigma_tweak[mask_tweak], wl_remain[mask_remain], fl_remain[mask_remain], sigma_remain[mask_remain], amp_f, l_f, order=order, mu_GP=mu_GP)

            # since fl_cor may have actually have fewer pixels than originally, we can't just
            # stuff the corrected fluxes directly back into the array.
            # Instead, we need to re-evaluate the line on all the wavelengths
            # using the chebyshev coefficients, and apply this.
            T = []
            for k in range(0, order + 1):
                pass
                coeff = [0 for j in range(k)] + [1]
                Chtemp = Ch(coeff, domain=[wl0, wl1])
                Ttemp = Chtemp(wl_tweak)
                T += [Ttemp]

            T = np.array(T)

            D = fl_tweak[:,np.newaxis] * T.T
            # Apply the correction
            fl_cor = np.dot(D, X)

            # replace this epoch with the corrected fluxes
            fl_out[i] = fl_cor

    # Stuff the full set of corrected fluxes (masked points included) back into the chunk.
    chunk.fl[:] = fl_out
