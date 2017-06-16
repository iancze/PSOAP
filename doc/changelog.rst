=========
Changelog
=========

------
v0.1.1
------

standardized different model selections
---------------------------------------

Now models can be specified as combinations of the number of gravitationally significant bodies, and the number of spectroscopically significant bodies. See `models.md` for more information. This has an impact for the way `orbit.py` is used.


Calibration optimization
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
