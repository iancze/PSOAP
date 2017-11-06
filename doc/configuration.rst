.. _configuration:

Configuration
=============

To start a new project, first decide what type of model you might like to use and navigate to a fresh directory. For example, let's choose an ``SB2``

Then, within your new directory ::

    $ psoap_initialize.py --model SB2

This will copy the appropriate scripts to your directory. You are encouraged to use your favorite text editor and inspect the contents of the ``config.yaml`` file, which will be used frequently by this package.

One of the first things you should notice is the field ``data_file: data.hdf5``.

.. _hdf5:

Processing your spectra to an HDF5 file
=======================================

PSOAP uses HDF5 files to storing a set of echelle spectra in a commonly used binary format.
An example of spectra in this format is provided via the LP661-13 dataset, which you can download `here <https://figshare.com/articles/LP661-13_TRES_Spectra/5572714>`_. Since PSOAP is primarily designed to be used with high resolution spectra, which are commonly acquired with echelle spectrographs, it presumes an echelle-like format. If you have a regular spectrum, then you would just treat your dataset as an echelle with only one order.

Assumptions of this format (and of PSOAP):

* all spectra are taken with the same instrument
* all spectra have been corrected to the barycentric frame
* all orders of the echelle have the same number of pixels

Though the limitation on the same number of pixels per order is not necessarily a strict constraint. Let ``n_pix`` be the number of pixels in each echelle order, ``n_orders`` be the number of echelle orders of the spectrograph, and ``n_epochs`` be the number of epochs of data you have. If you have telescope data that does not have an equal number of ``n_pix`` for each order, you can choose the largest order as defining ``n_pix`` and then masking (see below) the blank regions in the shorter orders.

The HDF5 file has the following top-level entries as datasets:

* ``BCV``: a length ``n_epochs`` array that provides the barycentric correction that was applied for each epoch. This value is only stored here for telluric/debugging purposes.
* ``JD``: a length ``n_epochs`` array that provides the barycentric Julian Date for each epoch.
* ```wl``: a size ``(n_epochs, n_orders, n_pix)`` array that contains the pixel wavelengths of the spectrum, *already corrected to the barycentric frame*.
* ``fl``: a size ``(n_epochs, n_orders, n_pix)`` array that contains the pixel fluxes of the spectrum.
* ``sigma``: a size ``(n_epochs, n_orders, n_pix)`` array that contains the pixel flux uncertainties of the spectrum.

If you write scripts for converting spectra from your telescope, please consider adding them to this package via a pull request (into a ``scripts/your-telescope-name/`` directory) so that they may be useful for other users with similar data.
