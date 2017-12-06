=========
Changelog
=========

------
v0.2.0
------

docs
----

The main feature of this release is significant progress in documenting most of the functionality of the PSOAP package, and creating a tutorial based upon LP661-13.

scripts
-------

Scripts have been moved from the ``scripts/`` directory to the ``psoap/scripts`` directory, and entry points have been added via ``setup.py``. Now, the user shouldn't need to manually add these scripts to their PATH. Several of the disparate scripts have also been consolidated into single code modules within this new directory. This has resulted in a lot of code reorganization, but functionality has remained mostly the same.

testing
-------

Several routines have been added to test the orbital routines, and especially the joint rv-astrometry fits. More functional tests still need to be written for the rest of the package.

------
v0.1.1
------

standardized different model selections
---------------------------------------

Now models can be specified as combinations of the number of gravitationally significant bodies, and the number of spectroscopically significant bodies. See `models.md` for more information. This has an impact for the way `orbit.py` is used.


calibration optimization
------------------------

For now, only implemented for the `ST3` model.

log-lambda input coordinates
----------------------------

Using log-lambda input coordinates instead of lambda enables a small but not insignificant speedup in the calculation of the kernel spacing, since the calculation of velocity spacing becomes a simple subtraction.

The following modules and scripts have been converted

* psoap_sample_parallel.py
* covariance.py
* matrix_functions.pyx
* psoap_retrieve_SB2.py
* psoap_retrieve_ST3.py

v0.1.0
------

Beta Release corresponding to code used for paper on the arXiv.
