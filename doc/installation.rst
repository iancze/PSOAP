.. _installation:

Installation
============


PSOAP requires a few standard packages. It is written for Python 3.3+. It is not tested on the Python 2.x series and I do not anticipate it to work on it either.

* ``numpy``
* ``scipy``
* ``astropy``
* ``h5py``
* ``matplotlib``
* ``celerite`` (optional)

All of these packages can be installed via an Anaconda python installation.

PSOAP has preliminary support for using the `celerite package <http://celerite.readthedocs.io/>`_, which implements fast, one dimensional Gaussian processes which are used when fitting a single-lined spectroscopic binary or triple. You can optionally install this package following the link above.

Then, clone the package from the `github repository <https://github.com/iancze/PSOAP>`_
and change to the top level ``PSOAP`` directory. Because the ``PSOAP`` package is still under occasional development, it may be wise to install the package in 'development mode' via the following commands::

    $ pip install -e .
    $ python setup.py build_ext --inplace

Then, you will need to add the ``psoap/scripts`` directory to your ``PATH`` and source your shell configuration file (e.g., ``source ~/.bashrc``). To check that you've got everything installed properly, try running from your shell ::

    $ psoap_initialize.py --check
    Using Python Version 3.5.2 |Anaconda custom (64-bit)| (default, Jul  2 2016, 17:53:06)
    [GCC 4.4.7 20120313 (Red Hat 4.4.7-1)]
    PSOAP successfully installed and linked.


If you installed the package via development mode, then it in the case that the package is upgraded, it is easy to upgrade your local copy by simply pulling down the latest changes and rerunning the build script::

    $ git pull
    $ python setup.py build_ext --inplace

Once we reach version 1.0, then the install command will be::

    $ python setup.py install

or::

    $ pip install psoap
