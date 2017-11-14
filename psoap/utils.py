r'''
The ``utils`` module contains values and functions for interacting with the registered spectroscopic orbits (:ref:`models`), converting these values from dictionaries to numpy vectors, filling frozen parameters, and assessing MCMC convergence through the Gelman-Rubin statistic.
'''

import numpy as np

# A dictionary of parameter lists for conversion.
registered_params = {"SB1": ["K", "e", "omega", "P", "T0", "gamma", "amp_f", "l_f"],
"SB2": ["q", "K", "e", "omega", "P", "T0", "gamma", "amp_f", "l_f", "amp_g", "l_g"],
"ST1": ["K_in", "e_in", "omega_in", "P_in", "T0_in", "K_out", "e_out", "omega_out", "P_out", "T0_out", "gamma", "amp_f", "l_f"],
"ST2": ["q_in", "K_in", "e_in", "omega_in", "P_in", "T0_in", "K_out", "e_out", "omega_out", "P_out", "T0_out", "gamma", "amp_f", "l_f"],
"ST3": ["q_in", "K_in", "e_in", "omega_in", "P_in", "T0_in", "q_out", "K_out", "e_out", "omega_out", "P_out", "T0_out", "gamma", "amp_f", "l_f", "amp_g", "l_g", "amp_h", "l_h"]}

registered_models = registered_params.keys()

# Calculate how many orbital parameters there are to each model.
# For now, we'll just calculate this based upon the position of the "gamma" parameter
n_params_orb = {model: (registered_params[model].index("gamma") + 1) for model in registered_params.keys()}

# labels used for plotting
registered_labels = {"SB1": [r"$K$", r"$e$", r"$\omega$", r"$P$", r"$T_0$", r"$\gamma$", r"$a_f$", r"$l_f$"],
"SB2": [r"$q$", r"$K$", r"$e$", r"$\omega$", r"$P$", r"$T_0$", r"$\gamma$", r"$a_f$", r"$l_f$", r"$a_g$", r"$l_g$"],
"ST3": [r"$q_\mathrm{in}$", r"$K_\mathrm{in}$", r"$e_\mathrm{in}$", r"$\omega_\mathrm{in}$", r"$P_\mathrm{in}$", r"$T_{0,\mathrm{in}}$", r"$q_\mathrm{out}$", r"$K_\mathrm{out}$", r"$e_\mathrm{out}$", r"$\omega_\mathrm{out}$", r"$P_\mathrm{out}$", r"$T_{0,\mathrm{out}}$", r"$\gamma$", r"$a_f$", r"$l_f$", r"$a_g$", r"$l_g$", r"$a_h$", r"$l_h$"]}

# For example, if the config.yaml file specifies model as "SB1", then it must list all of these parameters under "SB1".

# It must also specify fix_params, which will always include gamma.

# Then, within the sampling routines, we will create a partial function that takes in model type, list of fixed parameters, and dictionary of default  parameter values and upon invocation will convert a vector of only a subset of values into a full vector.

def convert_vector(p, model, fix_params, **kwargs):
    '''Unroll a vector of parameter values into a parameter type, using knowledge about which model we are fitting, the parameters we are fixing, and the default values of those parameters.

    Args:
        p (np.float) : 1D input array of only a subset of parameter values.
        model (str): ``"SB1"``, ``"SB2"``, etc..
        fix_params (list of str): names of parameters that will be fixed
        **kwargs: input for ``{param_name: default_value}`` pairs

    Returns:
        (np.float, np.float) : a 2-tuple of the  full vectors for the orbital parameters, and the GP parameters, augmented with the previously missing values.

    '''

    # Select the registered parameters corresponding to this model
    reg_params = registered_params[model]
    nparams = len(reg_params)

    # fit parameters are the ones in reg_params that are *not* in fix_params
    fit_params = [param for param in reg_params if param not in fix_params]
    fit_ind = [i for (i,param) in enumerate(reg_params) if param not in fix_params]

    # go through each element in fix_params, and find out where it would fit in reg_params
    fix_ind = [reg_params.index(param) for param in fix_params]

    # make an empty parameter vector of total parameter length
    par_vec = np.empty(nparams, dtype=np.float64)

    # Stuff it with all the values that we are fitting
    par_vec[fit_ind] = p

    # Fill in the holes with parameters that we are fixing
    # First, convert all fixed parameters to a similar type vector
    p_fixed = np.array([kwargs[par_name] for par_name in fix_params])
    par_vec[fix_ind] = p_fixed

    # Now split it to the orbital parameters and the GP parameters
    ind_split = n_params_orb[model]

    par_orb = par_vec[:ind_split]
    par_GP = par_vec[ind_split:]

    return (par_orb, par_GP)

# function convert_dict(p::Dict, model::AbstractString)
def convert_dict(model, fix_params, **kwargs):
    r'''Used to turn a dictionary of parameter values directly into a numpy array. Generally used for initializing starting position for MCMC routines from ``config.yaml``.

    Args:
        model (str): ``"SB1"``, ``"SB2"``, etc..
        fix_params (list): list of parameters that are kept fixed, whose values are to be read from ``kwargs``.
        kwargs (dict): dictionary of parameters with default values.

    Returns:
        np.array : length ``nparam`` vector constructed from ``kwargs`` values, excluding the fixed parameters.
    '''

    # Select the registered parameters corresponding to this model
    reg_params = registered_params[model]
    nparams = len(reg_params) - len(fix_params)

    par_vec = np.empty(nparams, dtype=np.float64)

    fit_params = [param for param in reg_params if param not in fix_params]

    p_fit = np.array([kwargs[par_name] for par_name in fit_params])

    return p_fit

def get_labels(model, fix_params):
    '''
    Collect the labels for each model to be used in plotting MCMC samples.

    Args:
        model (str):  ``"SB1"``, ``"SB2"``, etc..
        fix_params (list): list of parameters that are kept fixed, whose values are to be read from ``kwargs``.

    Returns:
        list : a list of strings corresponding to the name of each parameter that was fit for.
    '''
    reg_params = registered_params[model]
    reg_labels = registered_labels[model]
    fit_index = [i for (i,param) in enumerate(reg_params) if param not in fix_params]

    labels = [reg_labels[i] for i in fit_index]

    return labels

def gelman_rubin(samplelist):
    r'''
    Given a list of flatchains from separate runs (that already have burn in cut
    and have been trimmed, if desired), compute the Gelman-Rubin statistic for each parameter (more information in
    Bayesian Data Analysis 3, pg 284 Gelman et al.). If you want to compute this for fewer
    parameters, then slice the list before feeding it in.

    Args:
        samplelist (list): list of flatchains

    Returns:
        float: the Gelman-Rubin statistic :math:`\hat{R}`
    '''

    full_iterations = len(samplelist[0])
    assert full_iterations % 2 == 0, "Number of iterations must be even. Try cutting off a different number of burn in samples."
    shape = samplelist[0].shape
    #make sure all the chains have the same number of iterations
    for flatchain in samplelist:
        assert len(flatchain) == full_iterations, "Not all chains have the same number of iterations!"
        assert flatchain.shape == shape, "Not all flatchains have the same shape!"

    #make sure all chains have the same number of parameters.

    #Following Gelman,
    # n = length of split chains
    # i = index of iteration in chain
    # m = number of split chains
    # j = index of which chain
    n = full_iterations//2
    m = 2 * len(samplelist)
    nparams = samplelist[0].shape[-1] #the trailing dimension of a flatchain

    #Block the chains up into a 3D array
    chains = np.empty((n, m, nparams))
    for k, flatchain in enumerate(samplelist):
        chains[:,2*k,:] = flatchain[:n]  #first half of chain
        chains[:,2*k + 1,:] = flatchain[n:] #second half of chain

    #Now compute statistics
    #average value of each chain
    avg_phi_j = np.mean(chains, axis=0, dtype="f8") #average over iterations, now a (m, nparams) array
    #average value of all chains
    avg_phi = np.mean(chains, axis=(0,1), dtype="f8") #average over iterations and chains, now a (nparams,) array

    B = n/(m - 1.0) * np.sum((avg_phi_j - avg_phi)**2, axis=0, dtype="f8") #now a (nparams,) array

    s2j = 1./(n - 1.) * np.sum((chains - avg_phi_j)**2, axis=0, dtype="f8") #now a (m, nparams) array

    W = 1./m * np.sum(s2j, axis=0, dtype="f8") #now a (nparams,) arary

    var_hat = (n - 1.)/n * W + B/n #still a (nparams,) array
    std_hat = np.sqrt(var_hat)

    R_hat = np.sqrt(var_hat/W) #still a (nparams,) array


    data = Table({   "Value": avg_phi,
                     "Uncertainty": std_hat},
                 names=["Value", "Uncertainty"])

    print(data)

    ascii.write(data, sys.stdout, Writer = ascii.Latex, formats={"Value":"%0.3f", "Uncertainty":"%0.3f"}) #

    #print("Average parameter value: {}".format(avg_phi))
    #print("std_hat: {}".format(np.sqrt(var_hat)))
    print("R_hat: {}".format(R_hat))

    if np.any(R_hat >= 1.1):
        print("You might consider running the chain for longer. Not all R_hats are less than 1.1.")




def estimate_covariance(flatchain, ndim=0):
    '''
    Estimate the optimal Metropolis-Hastings jumps using the advice in Bayesian Data Analysis.

    Args:
        flatchain (np.array): array of samples
        d (float): estimate the covariance using a restricted number of dimensions.

    Returns:
        np.array: 2D covariance matrix for "optimal" M-H jumps.
    '''

    if ndim == 0:
        d = flatchain.shape[1]
    else:
        d = ndim

    import matplotlib.pyplot as plt

    #print("Parameters {}".format(flatchain.param_tuple))
    #samples = flatchain.samples
    cov = np.cov(flatchain, rowvar=0)

    #Now try correlation coefficient
    cor = np.corrcoef(flatchain, rowvar=0)

    # Make a plot of correlation coefficient.

    fig, ax = plt.subplots(figsize=(0.5 * d, 0.5 * d), nrows=1, ncols=1)
    ext = (0.5, d + 0.5, 0.5, d + 0.5)
    ax.imshow(cor, origin="upper", vmin=-1, vmax=1, cmap="bwr", interpolation="none", extent=ext)
    fig.savefig("cor_coefficient.png")

    print("'Optimal' jumps with covariance (units squared)")

    opt_jump = 2.38**2/d * cov
    # opt_jump = 1.7**2/d * cov # gives about ??

    std_dev = np.sqrt(np.diag(cov))

    print("'Optimal' jumps")
    print(2.38/np.sqrt(d) * std_dev)

    return opt_jump


def main():
    # "K", "e", "omega", "P", "T0", "gamma", "amp_f", "l_g"
    p = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    p_return = convert_vector(p, "SB1", ["gamma", "e"], gamma=1.0, e=-0.4)
    print(p_return)


if __name__=="__main__":
    main()
