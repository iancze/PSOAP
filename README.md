# PSOAP
Pronounced "soap."

[![Documentation Status](https://readthedocs.org/projects/psoap/badge/?version=latest)](http://psoap.readthedocs.io/en/latest/?badge=latest) [![Build Status](https://travis-ci.org/iancze/PSOAP.svg?branch=master)](https://travis-ci.org/iancze/PSOAP)

**Precision Spectroscopic Orbits A-Parametrically**

PSOAP is a package for simultaneously inferring stellar (and/or exoplanet) orbits and stellar spectra using Gaussian processes. Some uses include:

* Fitting for radial velocities in a template-free manner
* Inferring orbits of single-lined spectroscopic binaries (e.g., exoplanets/their host stars)
* Generation of high-fidelity stellar templates (for use with traditional RV cross-correlation measurements, variability searches)
* Inferring orbits and spectra of double-lined spectroscopic binaries (see gif below)

Documentation and installation instructi
ons are available at [http://psoap.readthedocs.io](http://psoap.readthedocs.io).

![disentangling loop](output.gif "disentangling loop")

If you use our paper, code, or a derivative of it in your research, we would really appreciate a citation to [Czekala et al. 2017](http://adsabs.harvard.edu/abs/2017ApJ...840...49C):

    @ARTICLE{2017ApJ...840...49C,
        author = {{Czekala}, I. and {Mandel}, K.~S. and {Andrews}, S.~M. and {Dittmann}, J.~A. and
        {Ghosh}, S.~K. and {Montet}, B.~T. and {Newton}, E.~R.},
        title = "{Disentangling Time-series Spectra with Gaussian Processes: Applications to Radial Velocity Analysis}",
        journal = {\apj},
        archivePrefix = "arXiv",
        eprint = {1702.05652},
        primaryClass = "astro-ph.SR",
        keywords = {binaries: spectroscopic, celestial mechanics, stars: fundamental parameters, stars: individual: LP661-13, techniques: radial velocities, techniques: spectroscopic},
         year = 2017,
        month = may,
        volume = 840,
          eid = {49},
        pages = {49},
          doi = {10.3847/1538-4357/aa6aab},
        adsurl = {http://adsabs.harvard.edu/abs/2017ApJ...840...49C},
        adsnote = {Provided by the SAO/NASA Astrophysics Data System}
    }

Copyright Ian Czekala and collaborators 2016-17.

## Papers using PSOAP

* *Disentangling Time-series Spectra with Gaussian Processes: Applications to Radial Velocity Analysis*, [Czekala et al. 2017](http://adsabs.harvard.edu/abs/2017ApJ...840...49C)
* *The Architecture of the GW Ori Young Triple Star System and Its Disk: Dynamical Masses, Mutual Inclinations, and Recurrent Eclipses*, [Czekala et al. 2017](http://adsabs.harvard.edu/abs/2017arXiv171003153C)
