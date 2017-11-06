Covariance Routines
===================

In the original paper, the kernel was specified using the velocity distance between two wavelengths (:math:`\lambda_i`, :math:`\lambda_j`). In subsequent versions, we choose to simply use the absolute magnitude of the distance between the *natural log* of the wavelengths, which is functionally equivalent to the former definition, but makes the computation easier. We let

.. math::

    \zeta_i = \ln \lambda_i

.. math::

    r_{ij} = |\zeta_i - \zeta_j|


Of course, when we are fitting for two spectral components, *f* and *g*, then there will be two sets of input wavelength vectors and thus two distances

.. math::

    r_{f,ij} = |\zeta_{f,i} - \zeta_{f,j}| \\
    r_{g,ij} = |\zeta_{g,i} - \zeta_{g,j}|

With two stars, to evaluate the kernel for a specific element in the matrix actually requires four inputs

.. math::

    k(\zeta_{f,i}, \zeta_{f,j}, \zeta_{g,i}, \zeta_{g,j} | a_f, l_f, a_g, l_g) = a_f^2 \exp \left ( - \frac{|\zeta_{f,i} - \zeta_{f,j}|^2}{2 l_f^2} \right) + a_g^2 \exp \left ( - \frac{|\zeta_{g,i} - \zeta_{g,j}|^2}{2 l_g^2} \right)




.. automodule:: psoap.covariance
    :members:

This is the covariance module.
