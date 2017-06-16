Orbit Routines
==============

The ``psoap.orbit`` module is the backbone for the main PSOAP tasks of computing radial velocities to shift the wavelength vectors. For a description of the various radial velocity orbital models, see :ref:`models`. There is also an additional module included within the PSOAP packaged, ``psoap.orbit_astrometry``, which is not used for the main Gaussian process routines, but includes functions for jointly modeling radial velocity measurements and relative astrometric measurements.

psoap.orbit
-----------

.. automodule:: psoap.orbit
    :members:
    :inherited-members:


psoap.orbit_astrometry
----------------------

This is an alternate parameterization suited for joint astrometric and radial velocity orbital modeling.

.. automodule:: psoap.orbit_astrometry
    :members:
