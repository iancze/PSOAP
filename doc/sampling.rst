.. _sampling:

Sampling the Parameters
=======================

Options to use SB2 and SB1. Show an example using the fast ``celerite`` GPs for single lined systems, including options to determine radial velocity on a per-epoch basis. Also strong warnings about contamination from secondary light.

While this package requires the ``emcee`` package, it doesn't actually use the ensemble sampler, just the simple Metropolis-Hastings proposal. Since this version is well-tested and uses the same interface most users might be familiar with, why not use that?

User-generated prior
--------------------

Could incorporate astrometric constraints.
