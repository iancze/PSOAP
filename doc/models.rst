.. _models:

======
Models
======

`PSOAP` currently supports a number of possible orbital models to a number of different astrophysical systems.

Gravitational bodies
--------------------
First, we use the following terminology to describe the *number of gravitationally significant bodies in a system*, meaning those bodies that must be factored into an orbital model. A single star would be a "spectroscopic single," or ``SS``, a binary star or a single star with an exoplanet would both be called a "spectroscopic binary," or ``SB`` (hecklers---please excuse the terminology or suggest an improvement), a hierarchical triple star would be called a "spectroscopic triple," or ``ST``, and a hierarchical quadruple star (or a double binary) would be called a "spectroscopic quadruple," or ``SQ``. More complicated orbital hierarchies can of course be implemented, but we will need to think more critically about a satisfying naming convention.

Spectroscopic bodies
--------------------
Second, we use a digit to describe the *number of spectroscopically significant bodies in a system*. In this case, a single star is ``1``, and a star with an exoplanet is also ``1``. A double-lined spectroscopic binary would be ``2``, and a triple-lined spectroscopic triple would be ``3``.

The final model specification is the concatenation of both of these specifications. So a double lined spectroscopic binary would be ``SB2``, while a single-lined spectroscopic triple is ``ST1``. Fortunately, the ``1`` models permit the usage of the extremely fast ``celerite`` `framework <http://adsabs.harvard.edu/abs/2017arXiv170309710F>`_. However, if the system is a multiple star and there is in fact significant flux contribution from the other components, using a single-lined model can deliver biased results.

More complicated models
***********************

Specifications become more complicated when dealing with triple stars. A single-lined spectroscopic triple is ``ST1``, and a double-lined spectroscopic triple is ``ST2``. Right now, the triple model assumes an (A-B)-C orbital architecture, and that ``1`` corresponds to light from A only, ``2`` corresponds to light from A and B only, and ``3`` corresponds to light from A, B, and C. If you have a different triple architecture, please consider raising an issue or submitting a pull request.


Specifying orbital models
-------------------------
Although one would think that only the number of gravitational bodies is needed to specify an orbit (and this is true), we also want to know how many spectroscopic bodies there are, since different orbital parameters are constrained based upon how many spectroscopic bodies we see. Therefore, `PSOAP` includes a number of orbital models for a wide range of situations (e.g., ``SS1``, ``SB1``, ``SB2``, ``ST1``, ``ST2``, ``ST3``, etc...).


If/when `PSOAP` grows to include telluric models these will be denoted with a ``+``. For example, a single star with a telluric model would be denoted by ``SS1+T``.
