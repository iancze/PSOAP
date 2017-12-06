.. _sampling:

Sampling and Reconstructing Spectra
===================================

The main function of `PSOAP` as a package is to sample the distribution of intrinsic stellar spectra and the orbital parameters, and then use these parameters to understand the orbit and stellar spectra. Since the sampling can take a long time (in fact, much longer than your astronomy career if you've chosen a poor starting point and/or poor proposal steps), and you may already have a decent guess at a stellar orbit (especially if you already have 5 or more spectra and RVs from the same telescope), it might be worth trying out the reconstruction step to see if your guess is close before committing the computational resources to do a full exploration of the posterior.

Reconstructing Spectra
----------------------

First, edit the ``parameters`` section of your ``config.yaml`` to reflect your best guess orbital parameters. In general, the amplitude hyperparameters of the GP will be reflective of the amplitude of the spectral line variations (typically in the range 5% - 60% of the continuum level, depending on the type of star). The length scale (in km/s) is roughly proportional to the broadening kernel of the spectra. For regular stars acquired with high resolution spectrographs, ``l`` will probably be in the range of 2 - 40 km/s.

The reconstruction of spectra boils down to Equation 27 of `Czekala et al. 2017 <http://adsabs.harvard.edu/abs/2017ApJ...840...49C>`_, which is implemented for single, double, and triple spectroscopic components in :ref:`covariance`. These routines are wrapped in the script ``psoap-reconstruct``. For more information, see the documentation of this routine in :ref:`scripts`.


Sampling the posterior
----------------------

The main purpose of the sampling routines are to propose and evaluate the posterior probability of the orbital and GP hyperparameters, partially provided by the likelihood routines in :ref:`covariance`. Because we have chunked the spectrum for faster likelihood evaluation, the main routine you will likely use is ``psoap-sample``, whose command-line arguments are provided in :ref:`scripts`. This routine runs fastest when your cluster provides a number of CPUs equal to or exceeding the number of chunks in your spectrum. As long as you have more CPUs available, you should be able to continue adding more chunks of spectrum to the inference process with no additional time per each posterior evaluation. For testing and tutorial purposes, a serial version of this script is provided in ``psoap-sample-serial``.

While `PSOAP` uses the `emcee` `package <http://emcee.readthedocs.io/en/stable/>`_, it doesn't actually use the marquee ensemble sampler, rather we simply use  the regular Metropolis-Hastings proposal since it is more straightforward to use with our parallel framework instead of the ensemble sampler. Since the `emcee` M-H module is well-tested and uses the same sampling interface that many users are likely familiar with, we thought it best to stick to what works.

User-generated prior
--------------------

By default, the main prior functions are uniform over the range of acceptable orbit parameters and GP hyperparameters. For example, this enforcing that the period, velocity semi-amplitudes, and GP amplitude and length scale are positive, that the eccentricity of the orbit is between [0,1).

A situation may arise where the user would like to modify the prior distribution to incorporate additional knowledge. For example, perhaps the period is extremely well known because the system is an eclipsing binary. In this case, the user can write a new ``prior`` function in a ``prior.py`` module in their current working directory. When ``psoap-sample`` is run, this module will be loaded and the ``prior`` function will completely replace the default prior. This means that in addition to your new constraints, it's important to replicate the behavior of the default prior that you want to keep in your new function. An example template for a user-defined prior is located in ``psoap/data/prior.py`` in the repository.


Plotting the MCMC chains
------------------------

Output from the sampling routines is saved in a ``output/run00`` directory, where by default the index increments based upon any previous runs, so as not to overwrite them. Users can also specify the directory to be written into via a command-line argument to ``psoap-sample``.

User can make diagnostic plots using the ``psoap-plot-samples`` script, which by default plots the chain, but can also make triangle plots using the `corner.py` `package <https://corner.readthedocs.io/en/latest/>`_. If you've run multiple chains, you can use the ``psoap-gelman-rubin`` script to assess convergence. As always, see :ref:`scripts` for more information.
