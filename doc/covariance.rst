Covariance Routines
===================

In the original paper, the kernel was specified using the velocity distance between two wavelengths (:math:`\lambda_i`, :math:`\lambda_j`). In subsequent versions, we choose to simply use the absolute magnitude of the distance between the *natural log* of the wavelengths, which is functionally equivalent to the former definition, but makes the computation easier.

The non-relativistic Doppler shift is

.. math::

    \mathrm{d}v = c \frac{\mathrm{d}\lambda}{\lambda}

If we use the property of the natural log

.. math::

    \frac{\mathrm{d} \ln \lambda}{\lambda} = \frac{\mathrm{d}\lambda}{\lambda}

Then we have

.. math::

    \Delta \ln \lambda \approx \frac{\Delta \lambda}{\lambda}

And

.. math::

    \Delta v = c\; \Delta \ln \lambda

Where :math:`c` is the speed of light in km/s. If we define

.. math::

    \zeta_i = \ln \lambda_i

Then we have as a distance metric

.. math::

    r_{ij} = c\,|\zeta_i - \zeta_j|

where :math:`r_{ij}` has units of km/s. Of course, when we are fitting for two spectral components, *f* and *g*, then there will be two sets of input wavelength vectors and thus two distances

.. math::

    r_{f,ij} = c\,|\zeta_{f,i} - \zeta_{f,j}| \\
    r_{g,ij} = c\,|\zeta_{g,i} - \zeta_{g,j}|

With two stars, to evaluate the kernel for a specific element in the matrix actually requires four inputs

.. math::

    k(\zeta_{f,i}, \zeta_{f,j}, \zeta_{g,i}, \zeta_{g,j} | a_f, l_f, a_g, l_g) = a_f^2 \exp \left ( - \frac{c^2\,|\zeta_{f,i} - \zeta_{f,j}|^2}{2 l_f^2} \right) + a_g^2 \exp \left ( - \frac{c^2\,|\zeta_{g,i} - \zeta_{g,j}|^2}{2 l_g^2} \right)




.. automodule:: psoap.covariance
    :members:

This is the covariance module.
