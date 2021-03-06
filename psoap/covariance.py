import numpy as np
from numpy.polynomial import Chebyshev as Ch

from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize

import psoap
from psoap import constants as C
from psoap import matrix_functions
from psoap.data import lredshift

try:
    import celerite
    from celerite import terms
except ImportError:
    print("If you want to use the fast 1D (SB1 or ST1 models), please install celerite")

try:
    import george
    from george import kernels
except ImportError:
    print("If you want to use the fast GP solver (SB2, ST2, or ST3 models) please install george")


def predict_f(lwl_known, fl_known, sigma_known, lwl_predict, amp_f, l_f, mu_GP=1.0):

    '''wl_known are known wavelengths.
    wl_predict are the prediction wavelengths.
    Assumes all inputs are 1D arrays.'''

    # determine V11, V12, V21, and V22
    M = len(lwl_known)
    V11 = np.empty((M, M), dtype=np.float64)
    matrix_functions.fill_V11_f(V11, lwl_known, amp_f, l_f)
    # V11[np.diag_indices_from(V11)] += sigma_known**2
    V11 = V11 + sigma_known**2 * np.eye(M)

    N = len(wl_predict)
    V12 = np.empty((M, N), dtype=np.float64)
    matrix_functions.fill_V12_f(V12, lwl_known, lwl_predict, amp_f, l_f)

    V22 = np.empty((N, N), dtype=np.float64)
    # V22 is the covariance between the prediction wavelengths
    # The routine to fill V11 is the same as V22
    matrix_functions.fill_V11_f(V22, lwl_predict, amp_f, l_f)

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


def predict_f_g(lwl_f, lwl_g, fl_fg, sigma_fg, lwl_f_predict, lwl_g_predict, mu_f, amp_f, l_f, mu_g, amp_g, l_g, get_Sigma=True):
    '''
    Given that f + g is the flux that we're modeling, jointly predict the components.
    '''
    # Assert that wl_f and wl_g are the same length
    assert len(lwl_f) == len(lwl_g), "Input wavelengths must be the same length."
    n_pix = len(lwl_f)

    assert len(lwl_f_predict) == len(lwl_g_predict), "Prediction wavelengths must be the same length."
    n_pix_predict = len(lwl_f_predict)

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
    matrix_functions.fill_V11_f(V11_f, lwl_f, amp_f, l_f)
    matrix_functions.fill_V11_f(V11_g, lwl_g, amp_g, l_g)

    B = V11_f + V11_g
    B[np.diag_indices_from(B)] += sigma_fg**2

    # print("factoring sum")
    factor, flag = cho_factor(B)

    # print("Allocating prediction matrices")
    # Now create separate matrices for the prediction
    V11_f_predict = np.empty((n_pix_predict, n_pix_predict), dtype=np.float)
    V11_g_predict = np.empty((n_pix_predict, n_pix_predict), dtype=np.float)

    # print("Filling prediction matrices")
    matrix_functions.fill_V11_f(V11_f_predict, lwl_f_predict, amp_f, l_f)
    matrix_functions.fill_V11_f(V11_g_predict, lwl_g_predict, amp_g, l_g)

    zeros = np.zeros((n_pix_predict, n_pix_predict))
    A = np.vstack((np.hstack([V11_f_predict, zeros]), np.hstack([zeros, V11_g_predict])))
    # A[np.diag_indices_from(A)] += 1e-4 # Add a small nugget term

    # C is now the cross-matrices between the predicted wavelengths and the data wavelengths
    V12_f = np.empty((n_pix_predict, n_pix), dtype=np.float)
    V12_g = np.empty((n_pix_predict, n_pix), dtype=np.float)

    # print("Filling cross-matrices")
    matrix_functions.fill_V12_f(V12_f, lwl_f_predict, lwl_f, amp_f, l_f)
    matrix_functions.fill_V12_f(V12_g, lwl_g_predict, lwl_g, amp_g, l_g)

    C = np.vstack((V12_f, V12_g))

    # print("Sloving for mu, sigma")
    # the 1.0 signifies that mu_f + mu_g = mu_fg = 1
    mu = mu_cat + np.dot(C, cho_solve((factor, flag), fl_fg - 1.0))

    if get_Sigma:
        Sigma = A - np.dot(C, cho_solve((factor, flag), C.T))

        return mu, Sigma

    else:
        return mu


def predict_f_g_sum(lwl_f, lwl_g, fl_fg, sigma_fg, lwl_f_predict, lwl_g_predict, mu_fg, amp_f, l_f, amp_g, l_g):

    # Assert that wl_f and wl_g are the same length
    assert len(lwl_f) == len(lwl_g), "Input wavelengths must be the same length."

    M = len(lwl_f_predict)
    N = len(lwl_f)

    V11_f = np.empty((M, M), dtype=np.float)
    V11_g = np.empty((M, M), dtype=np.float)

    matrix_functions.fill_V11_f(V11_f, lwl_f_predict, amp_f, l_f)
    matrix_functions.fill_V11_f(V11_g, lwl_g_predict, amp_g, l_g)
    V11 = V11_f + V11_g
    V11[np.diag_indices_from(V11)] += 1e-8

    V12_f = np.empty((M, N), dtype=np.float64)
    V12_g = np.empty((M, N), dtype=np.float64)
    matrix_functions.fill_V12_f(V12_f, lwl_f_predict, lwl_f, amp_f, l_f)
    matrix_functions.fill_V12_f(V12_g, lwl_g_predict, lwl_g, amp_g, l_g)
    V12 = V12_f + V12_g

    V22_f = np.empty((N,N), dtype=np.float)
    V22_g = np.empty((N,N), dtype=np.float)

    # It's a square matrix, so we can just reuse fill_V11_f
    matrix_functions.fill_V11_f(V22_f, lwl_f, amp_f, l_f)
    matrix_functions.fill_V11_f(V22_g, lwl_g, amp_g, l_g)
    V22 = V22_f + V22_g
    V22[np.diag_indices_from(V22)] += sigma_fg**2

    factor, flag = cho_factor(V22)

    mu = mu_fg + np.dot(V12, cho_solve((factor, flag), (fl_fg - 1.0)))
    Sigma = V11 - np.dot(V12, cho_solve((factor, flag), V12.T))

    return mu, Sigma


def predict_f_g_h(lwl_f, lwl_g, lwl_h, fl_fgh, sigma_fgh, lwl_f_predict, lwl_g_predict, lwl_h_predict, mu_f, mu_g, mu_h, amp_f, l_f, amp_g, l_g, amp_h, l_h):
    '''
    Given that f + g + h is the flux that we're modeling, jointly predict the components.
    '''
    # Assert that wl_f and wl_g are the same length
    assert len(lwl_f) == len(lwl_g), "Input wavelengths must be the same length."
    assert len(lwl_f) == len(lwl_h), "Input wavelengths must be the same length."
    n_pix = len(lwl_f)

    assert len(lwl_f_predict) == len(lwl_g_predict), "Prediction wavelengths must be the same length."
    assert len(lwl_f_predict) == len(lwl_h_predict), "Prediction wavelengths must be the same length."
    n_pix_predict = len(lwl_f_predict)

    # Convert mu constants into vectors
    mu_f = mu_f * np.ones(n_pix_predict)
    mu_g = mu_g * np.ones(n_pix_predict)
    mu_h = mu_h * np.ones(n_pix_predict)

    # Cat these into a single vector
    mu_cat = np.hstack((mu_f, mu_g, mu_h))

    V11_f = np.empty((n_pix, n_pix), dtype=np.float)
    V11_g = np.empty((n_pix, n_pix), dtype=np.float)
    V11_h = np.empty((n_pix, n_pix), dtype=np.float)

    matrix_functions.fill_V11_f(V11_f, lwl_f, amp_f, l_f)
    matrix_functions.fill_V11_f(V11_g, lwl_g, amp_g, l_g)
    matrix_functions.fill_V11_f(V11_h, lwl_h, amp_h, l_h)

    B = V11_f + V11_g + V11_h
    B[np.diag_indices_from(B)] += sigma_fgh**2

    factor, flag = cho_factor(B)

    # Now create separate matrices for the prediction
    V11_f_predict = np.empty((n_pix_predict, n_pix_predict), dtype=np.float)
    V11_g_predict = np.empty((n_pix_predict, n_pix_predict), dtype=np.float)
    V11_h_predict = np.empty((n_pix_predict, n_pix_predict), dtype=np.float)

    # Fill the prediction matrices
    matrix_functions.fill_V11_f(V11_f_predict, lwl_f_predict, amp_f, l_f)
    matrix_functions.fill_V11_f(V11_g_predict, lwl_g_predict, amp_g, l_g)
    matrix_functions.fill_V11_f(V11_h_predict, lwl_h_predict, amp_h, l_h)

    zeros = np.zeros((n_pix_predict, n_pix_predict))

    A = np.vstack((np.hstack([V11_f_predict, zeros, zeros]), np.hstack([zeros, V11_g_predict, zeros]), np.hstack([zeros, zeros, V11_h_predict])))

    V12_f = np.empty((n_pix_predict, n_pix), dtype=np.float)
    V12_g = np.empty((n_pix_predict, n_pix), dtype=np.float)
    V12_h = np.empty((n_pix_predict, n_pix), dtype=np.float)

    matrix_functions.fill_V12_f(V12_f, lwl_f_predict, lwl_f, amp_f, l_f)
    matrix_functions.fill_V12_f(V12_g, lwl_g_predict, lwl_g, amp_g, l_g)
    matrix_functions.fill_V12_f(V12_h, lwl_h_predict, lwl_h, amp_h, l_h)

    C = np.vstack((V12_f, V12_g, V12_h))

    mu = mu_cat + np.dot(C, cho_solve((factor, flag), fl_fgh - 1.0))
    Sigma = A - np.dot(C, cho_solve((factor, flag), C.T))

    return mu, Sigma

def predict_f_g_h_sum(lwl_f, lwl_g, lwl_h, fl_fgh, sigma_fgh, lwl_f_predict, lwl_g_predict, lwl_h_predict, mu_fgh, amp_f, l_f, amp_g, l_g, amp_h, l_h):
    '''
    Given that f + g + h is the flux that we're modeling, predict the joint sum.
    '''
    # Assert that wl_f and wl_g are the same length
    assert len(lwl_f) == len(lwl_g), "Input wavelengths must be the same length."

    M = len(lwl_f_predict)
    N = len(lwl_f)

    V11_f = np.empty((M, M), dtype=np.float)
    V11_g = np.empty((M, M), dtype=np.float)
    V11_h = np.empty((M, M), dtype=np.float)

    matrix_functions.fill_V11_f(V11_f, lwl_f_predict, amp_f, l_f)
    matrix_functions.fill_V11_f(V11_g, lwl_g_predict, amp_g, l_g)
    matrix_functions.fill_V11_f(V11_h, lwl_h_predict, amp_h, l_h)
    V11 = V11_f + V11_g + V11_h
    # V11[np.diag_indices_from(V11)] += 1e-5 # small nugget term

    V12_f = np.empty((M, N), dtype=np.float64)
    V12_g = np.empty((M, N), dtype=np.float64)
    V12_h = np.empty((M, N), dtype=np.float64)
    matrix_functions.fill_V12_f(V12_f, lwl_f_predict, lwl_f, amp_f, l_f)
    matrix_functions.fill_V12_f(V12_g, lwl_g_predict, lwl_g, amp_g, l_g)
    matrix_functions.fill_V12_f(V12_h, lwl_h_predict, lwl_h, amp_h, l_h)
    V12 = V12_f + V12_g + V12_h

    V22_f = np.empty((N,N), dtype=np.float)
    V22_g = np.empty((N,N), dtype=np.float)
    V22_h = np.empty((N,N), dtype=np.float)

    # It's a square matrix, so we can just reuse fil_V11_f
    matrix_functions.fill_V11_f(V22_f, lwl_f, amp_f, l_f)
    matrix_functions.fill_V11_f(V22_g, lwl_g, amp_g, l_g)
    matrix_functions.fill_V11_f(V22_h, lwl_h, amp_h, l_h)
    V22 = V22_f + V22_g + V22_h
    V22[np.diag_indices_from(V22)] += sigma_fgh**2

    factor, flag = cho_factor(V22)

    mu = mu_fgh + np.dot(V12.T, cho_solve((factor, flag), (fl_fgh - mu_fgh)))
    Sigma = V11 - np.dot(V12, cho_solve((factor, flag), V12.T))

    return mu, Sigma

def lnlike_f(V11, wl_f, fl, sigma, amp_f, l_f, mu_GP=1.):
    """Calculate the log-likelihood for a single-lined spectrum.

    This function takes a pre-allocated array and fills out the covariance matrices and evaluates the likelihood function for a single-lined spectrum, assuming a squared-exponential kernel (does not ``celerite``).

    Args:
        V11 (numpy 2D array): Description of arg1
        wl_f (numpy 1D array): Description of arg2
        fl (numpy 1D array): ae
        amp_f (float) : amplitude of GP
        l_f (float) : length scale of GP
        mu_GP (float) : mean of GP

    Returns:
        float: The log-likelihood value

    """

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
        # factor, flag = cho_factor(V11)
        factor, flag = cho_factor(V11, overwrite_a=True, lower=False, check_finite=False)
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

# Assemble lnlikelihood functions for the different models
lnlike = {"SB1": lnlike_f, "SB2": lnlike_f_g, "ST1": lnlike_f, "ST2": lnlike_f_g, "ST3": lnlike_f_g_h}

# Alternatively, use george to do the likelihood calculations
def lnlike_f_g_george(lwl_f, lwl_g, fl, sigma, amp_f, l_f, amp_g, l_g, mu_GP=1.):
    '''
    Evaluate the joint likelihood for *f* and *g* using George.
    '''

    # assumes that the log wavelengths, fluxes, and errors are already flattened
    # lwl_f = chunk.lwl.flatten()
    # lwl_g = chunk.lwl.flatten()

    # does it help to sort?
    # ind = np.argsort(lwl_f)

    x = np.vstack((lwl_f, lwl_g)).T

    # might also want to "block" the kernel to limit it to some velocity range
    kernel = amp_f * kernels.ExpSquaredKernel(l_f, ndim=2, axes=0) # primary
    kernel += amp_g * kernels.ExpSquaredKernel(l_g, ndim=2, axes=1) # secondary

    # instantiate the GP and evaluate the kernel for the prior
    gp = george.GP(kernel)
    gp.compute(x, sigma)

    # evaluate the likelihood for the data
    return gp.log_likelihood(fl)


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

def optimize_epoch_velocity_f(lwl_epoch, fl_epoch, sigma_epoch, lwl_fixed, fl_fixed, sigma_fixed, gp):
    '''
    Optimize the wavelengths of the chosen epoch relative to the fixed wavelengths. Identify the velocity required to redshift the chosen epoch.
    '''


    fl = np.concatenate((fl_epoch, fl_fixed)).flatten()
    sigma = np.concatenate((sigma_epoch, sigma_fixed)).flatten()

    def func(p):
        try:
            v, log_sigma, log_rho = p

            if v < -200 or v > 200 or log_sigma < -3 or log_sigma > -2 or log_rho < -9 or log_rho > -8:
                return -np.inf

            # Doppler shift the input wl_epoch
            lwl_shift = lredshift(lwl_epoch, v)

            # Reconcatenate spectra into 1D array and sort
            lwl = np.concatenate((lwl_shift, lwl_fixed)).flatten()

            indsort = np.argsort(lwl)

            # Set the par vectors
            gp.set_parameter_vector(p[1:])

            # compute GP on new wl grid
            gp.compute(lwl[indsort], yerr=sigma[indsort])
            return -gp.log_likelihood(fl[indsort])

        except np.linalg.linalg.LinAlgError:
            return np.inf

    # bound as -200 to 200 km/s
    p0 = np.concatenate((np.array([0.0]), gp.get_parameter_vector()))
    # bounds = [(-200, 200.)] + gp.get_parameter_bounds()
    # print(bounds)

    # ans = minimize(func, p0, method="L-BFGS-B", bounds=bounds)
    ans = minimize(func, p0, method="Nelder-Mead")
    # The velocity returned is the amount that was required to redshift wl_epoch to line up with wl_fixed.

    if ans["success"]:
        print("Success found with", ans["x"])
        return ans["x"][0]
    else:
        print(ans)
        raise C.ChunkError("Unable to optimize velocity for epoch.")

def determine_all_velocities(chunk, log_sigma, log_rho, mu_GP=1.0):
    kernel = terms.Matern32Term(log_sigma=log_sigma, log_rho=log_rho)
    gp = celerite.GP(kernel, mean=1.0, fit_mean=False)

    lwl_fixed = chunk.lwl[0]
    fl_fixed = chunk.fl[0]
    sigma_fixed = chunk.sigma[0]

    velocities = np.empty(chunk.n_epochs, dtype=np.float64)
    velocities[0] = 0.0

    for i in range(1, chunk.n_epochs):
        try:
            velocities[i] = optimize_epoch_velocity_f(chunk.lwl[i], chunk.fl[i], chunk.sigma[i], lwl_fixed, fl_fixed, sigma_fixed, gp)
        except C.ChunkError as e:
            print("Unable to optimize velocity for epoch {:}".format(chunk.date1D[i]))
            velocities[i] = 0.0

    return velocities

# uses smart inverse from Celerite
def optimize_calibration_ST1(lwl0, lwl1, lwl_cal, fl_cal, fl_fixed, gp, A, C, mu_GP=1.0, order=1):
    '''
    Determine the calibration parameters for this epoch of observations.

    lwl0, lwl1: set the points for the Chebyshev.

    This is a more general method than optimize_calibration_static, since it allows arbitrary covariance matrices, which should be used when there is orbital motion.

    lwl_cal: the wavelengths corresponding to the epoch we want to calibrate
    fl_cal: the fluxes corresponding to the epoch we want to calibrate

    fl_fixed: the remaining epochs of data to calibrate in reference to.

    gp: the celerite GP

    order: the degree polynomial to use. order = 1 is a line, order = 2 is a line + parabola

    Assumes that covariance matrices are appropriately filled out.
    '''

    # Get a clean set of the Chebyshev polynomials evaluated on the input wavelengths
    T = []
    for i in range(0, order + 1):
        coeff = [0 for j in range(i)] + [1]
        Chtemp = Ch(coeff, domain=[lwl0, lwl1])
        Ttemp = Chtemp(lwl_cal)
        T += [Ttemp]

    T = np.array(T)

    D = fl_cal[:,np.newaxis] * T.T


    # Solve for the calibration coefficients c0, c1, ...

    # Find B^{-1}, fl_prime, and C_prime
    # B^{-1} corresponds to the gp.apply_inverse

    fl_prime = mu_GP + np.dot(C, gp.apply_inverse(fl_fixed.flatten() - mu_GP))

    C_prime = A - np.dot(C, gp.apply_inverse(C.T))

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



def optimize_calibration(lwl0, lwl1, lwl_cal, fl_cal, fl_fixed, A, B, C, order=1, mu_GP=1.0):
    '''
    Determine the calibration parameters for this epoch of observations. This is a more general method than :py:meth:`psoap.covariance.optimize_calibration_static`, since it allows arbitrary covariance matrices, which should be used when there is orbital motion. Assumes that covariance matrices are appropriately filled out.

    Args:
        lwl0 (float) : left side evaluation point for Chebyshev
        lwl1 (float) : right side evaluation point for Chebyshev
        lwl_cal (np.array): the wavelengths corresponding to the epoch we want to calibrate
        fl_cal (np.array): the fluxes corresponding to the epoch we want to calibrate
        fl_fixed (np.array): the remaining epochs of data to calibrate in reference to.
        A (2D np.array) : matrix_functions.fill_V11_f(A, lwl_cal, amp, l_f) with sigma_cal already added to the diagonal
        B (2D np.array) : matrix_functions.fill_V11_f(B, lwl_fixed, amp, l_f) with sigma_fixed already added to the diagonal
        C (2D np.array): matrix_functions.fill_V12_f(C, lwl_cal, lwl_fixed, amp, l_f) cross matrix (with no sigma added, since these are independent measurements).
        order (int): the degree polynomial to use. order = 1 is a line, order = 2 is a line + parabola

    Returns:
        (np.array, np.array): a tuple of two data products. The first is the ``fl_cal`` vector, now calibrated. The second is the array of the Chebyshev coefficients, in case one wants to re-evaluate the calibration polynomials.
    '''

    # basically, assume that A, B, and C are already filled out.
    # the only thing this routine needs to do is fill out the Q matrix

    # Get a clean set of the Chebyshev polynomials evaluated on the input wavelengths
    T = []
    for i in range(0, order + 1):
        coeff = [0 for j in range(i)] + [1]
        Chtemp = Ch(coeff, domain=[lwl0, lwl1])
        Ttemp = Chtemp(lwl_cal)
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



def optimize_calibration_static(wl0, wl1, wl_cal, fl_cal, sigma_cal, wl_fixed, fl_fixed, sigma_fixed, amp, l_f, order=1, mu_GP=1.0):
    '''
    Determine the calibration parameters for this epoch of observations. Assumes all wl, fl arrays are 1D, and that the relative velocities between all epochs are zero.

    Args:
        wl0 (float) : left wl point to evaluate the Chebyshev
        wl1 (float) : right wl point to evaluate the Chebyshev
        wl_cal (np.array) : the wavelengths of the epoch to calibrate
        fl_cal (np.array) : the fluxes of the epoch to calibrate
        sigma_cal (np.array): the sigmas of the epoch to calibrate
        wl_fixed (np.array) : the 1D (flattened) array of the reference wavelengths
        fl_fixed (np.array) : the 1D (flattened) array of the reference fluxes
        sigma_fixed (np.array) : the 1D (flattened) array of the reference sigmas
        amp (float): the GP amplitude
        l_f (float): the GP length
        order (int): the Chebyshev order to use
        mu_GP (optional): the mean of the GP to assume.

    Returns:
        (np.array, np.array): a tuple of two data products. The first is the ``fl_cal`` vector, now calibrated. The second is the array of the Chebyshev coefficients, in case one wants to re-evaluate the calibration polynomials.
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
