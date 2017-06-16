.. _models:

======
Models
======

As PSOAP has grown, there a number of different possible models available to fit a number of different astrophysical systems.

Gravitational bodies
--------------------
First, we use the following terminology to describe the *number of gravitationally significant bodies in a system*, meaning those bodies that must be factored into an orbital model. A single star would be a "spectroscopic single," or `SS`, a binary star or a single star with an exoplanet would both be called a "spectroscopic binary," or `SB` (hecklers---please excuse the terminology or suggest an improvement), a hierarchical triple star would be called a "spectroscopic triple," or `ST`, and a hierarchical quadruple star (or a double binary) would be called a "spectroscopic quadruple," or `SQ`. More complicated orbital hierarchies can be implemented, but we will need to think more critically about how to name these.

Spectroscopic bodies
--------------------
Second, we use a number to describe the *number of spectroscopically significant bodies in a system*. In this case, a single star is `1`, and a star with an exoplanet is also `1`. A double-lined spectroscopic binary would be `2`, and a triple-lined spectroscopic triple would be `3`. However, a single-lined spectroscopic triple would be `1`.

The final model specification is the concatenation of both of these. So a double lined spectroscopic binary is `SB2`, while a single-lined spectroscopic triple is `ST1`. Fortunately, the `1` models permit the usage of the extremely fast `celerite` framework by [Foreman-Mackey et al. 2017](http://adsabs.harvard.edu/abs/2017arXiv170309710F). However, if the system is a multiple star and there is in fact significant flux contribution from the other components, using a single-lined model can deliver biased results.

Specifying orbital models
-------------------------
Although one would think that we would only need the number of gravitational bodies to specify an orbit (and this is true), we do usually want to know how many spectroscopic bodies there are as well, since there are a different number of orbital parameters that we can constrain based upon how many spectroscopic bodies we see. Therefore `orbit.py` has orbital models for a wide range of models (e.g., `ST1`, `SB2`, etc...).

When this framework grows to include telluric models, or veiling models, these will be denoted with a `+`. For example, a single star with a telluric model would be denoted by `SS1+T`. Unfortunately these additional models are not able to use the fast `celerite` framework.
