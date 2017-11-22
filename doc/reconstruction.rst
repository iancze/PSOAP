Reconstructing Spectra
======================

The main function of `PSOAP` as a package is to sampling the distribution of intrinsic stellar spectra and the orbital parameters. The aim of most analyses is to then sample the posterior distribution of the parameters and use these parameters to recreate the orbit and stellar spectra. Since the sampling can take a long time (in fact, much longer than your astronomy career if you've chosen a poor starting point and/or poor proposal steps), and you may have a decent guess at a stellar orbit to begin with (especially if you already have 5 or more spectra from the same telescope), it might be worth trying out the reconstruction step before committing the computational resources to do a full exploration of the posterior.

First, edit the ``parameters`` section of your ``config.yaml`` to reflect your best guess orbital parameters. In general, the amplitude hyperparameters of the GP will be reflective of the amplitude of the spectral line variations (typically in the range 5% - 60% of the continuum level, depending on the type of star). The length scale (in km/s) is roughly proportional to the broadening kernel of the spectra. For regular stars acquired with high resolution spectrographs, ``l`` will probably be in the range of 2 - 40 km/s.

The reconstruction of spectra boils down to Equation 27 of `Czekala et al. 2017 <http://adsabs.harvard.edu/abs/2017ApJ...840...49C>`_, which is implemented for single, double, and triple spectroscopic components in :ref:`covariance`. These routines are wrapped in the script ``psoap-reconstruct``. For more information, see the documentation of this routine in :ref:`scripts`.

.. automodule:: psoap.scripts.reconstruct
    :members:
