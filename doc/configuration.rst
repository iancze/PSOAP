.. _configuration:

Configuration
=============

.. _hdf5:

PSOAP Spectra File Format
-------------------------

`PSOAP` uses HDF5 files to store the dataset of echelle spectra in a commonly used binary format.
An example of spectra in this format is provided via the LP661-13 dataset, which you can download `here <https://figshare.com/articles/LP661-13_TRES_Spectra/5572714>`_. Since `PSOAP` is primarily designed to be used with high resolution spectra, which are commonly acquired with echelle spectrographs, it presumes an echelle-like format. If you have a regular spectrum, then you would just treat your dataset as an echelle with only one order.

Assumptions of this format (and of `PSOAP`):

* all spectra are taken with the same instrument
* all spectra have been corrected to the barycentric frame
* all orders of the echelle have the same number of pixels

The limitation on the same number of pixels per order is not necessarily a strict constraint. Let ``n_pix`` be the number of pixels in each echelle order, ``n_orders`` be the number of echelle orders of the spectrograph, and ``n_epochs`` be the number of epochs of data you have. If you have telescope data that does not have an equal number of ``n_pix`` for each order, you can choose the largest order as defining ``n_pix`` and then mask (see below) the blank regions in the shorter orders. Future versions of `PSOAP` may have the ability to incorporate data from multiple instruments, but for the foreseeable future it should be assumed that all data must be acquired from the same spectrograph (with the same spectral resolution/line spread function).

The HDF5 file has the following top-level entries as datasets:

* ``BCV``: a length ``n_epochs`` array that provides the barycentric correction that was applied for each epoch. This value is only stored here for telluric/debugging purposes.
* ``JD``: a length ``n_epochs`` array that provides the barycentric Julian Date for each epoch.
* ```wl``: a size ``(n_epochs, n_orders, n_pix)`` array that contains the pixel wavelengths of the spectrum, *already corrected to the barycentric frame*.
* ``fl``: a size ``(n_epochs, n_orders, n_pix)`` array that contains the pixel fluxes of the spectrum.
* ``sigma``: a size ``(n_epochs, n_orders, n_pix)`` array that contains the pixel flux uncertainties of the spectrum.

If you write scripts for converting spectra from your telescope, please consider adding them to this package via a pull request (into a ``scripts/your-telescope-name/`` directory) so that they may be useful for other users with similar data. For example, there are already some scripts for converting data taken with the *TRES* spectrograph into the HDF5 format `here <https://github.com/iancze/PSOAP/tree/master/scripts/TRES>`_.

A note on flux calibration
**************************

It is important that all of your spectra be "flux-calibrated" in the same manner, i.e., that successive epochs of the spectrum have been calibrated to be the same level in some arbitrary unit (we usually use units where the continuum = 1). This doesn't necessarily mean that you need simultaneous observations of spectrophotometric standard stars. If your spectrograph has stable throughput, you may achieve the best success by simply blaze-correcting and then rectifying your spectra (i.e., dividing by the average value) rather than following a continuum-normalization process, since the continuum might be somewhat undefined for later spectral types. If the bandpass calibration of your spectrograph is somewhat variable, it may be worth exploring the iterative self-calibration procedures in the ``psoap-process-calibration`` script.

config.yaml
-----------

`PSOAP` reads parameters specific to your project from a ``config.yaml`` file located in your current working directory. You can initialize your directory using the ``psoap-initialize`` script with the correct choice of model (see :ref:`models` and :ref:`scripts` for more information). This file is written in the `YAML <http://www.yaml.org/start.html>`_ format. For the most part, the default parameters should be sufficient to get most projects started. The main thing that you will eventually want to change will be the orbital and GP hyperparameters in the ``parameters`` and ``jumps`` sections. For more information on this, see :ref:`sampling`.

Chunking the dataset
--------------------

Due to the steep :math:`{\cal O}(N^3)` scaling of the matrix multiplications for Gaussian processes and the relative sparseness of the covariance kernels (only nearby pixels are significantly correlated), it is fastest to do the likelihood calculations on chunks of the spectrum, rather than the full spectral range in one go. This also enables the likelihood evaluations to be parallelized on multi-core machines or compute clusters.

The fitting process revolves around the ``chunks.dat`` file, which is an ascii file with the following space-separated form ::

  order wl0 wl1
  22 5160.0 5165.2
  22 5165.0 5170.2
  22 5170.0 5175.2
  22 5175.0 5180.2
  22 5180.0 5185.2
  22 5185.0 5190.2
  22 5190.0 5195.2
  22 5195.0 5200.2

Each row lists a new "chunk" to be made from the HDF5 file with your data. These chunk files are also stored in HDF5 format with the name ``chunk_<order>_<wl0>_<wl1>.hdf5``. You can autogenerate a ``chunks.dat`` file using the ``psoap-generate-chunks`` script, and then process the data file into chunks using ``psoap-process-chunks``. For more information, use the ``--help`` flag or see :ref:`scripts`.


Masking
-------

Occasionally, your spectra may suffer from local non-Gaussian defects such as cosmic ray hits, poor night-sky line subtraction, or wide swaths of contamination by telluric absorption lines. While ``PSOAP`` may eventually support disentangling telluric lines, the easiest thing to currently do is just mask these artifacts from the likelihood calculation. This is done through the ``masks.dat`` file, which has the following format ::

    wl0 wl1 t0 t1
    3000.0 9999.9 2456966.0 2457015.5
    3000.0 9999.9 2457297.5 2457298.5
    5160.8 5161.8 2455957.6 2455958.0
    5164.5 5165.5 2455856.9 2455857.1
    5185.5 5186.3 2455826.9 2455827.1
    6104.6 6106.3 2456559.7 2456560.1

Each row lists a starting wavelength and ending wavelength (in AA), along with a starting and ending date (in JD). For example, you can mask out entire epochs of spectra using something like the first two rows, or just mask out a single ~1 AA chunk in a specific range of epochs using the remaining rows. You can use the ``psoap-generate-masks`` script to autogenerate this file, though you may need to go back and tweak some regions by hand later. Once you are satisified with the choices, you can use the ``psoap-process-masks`` script to mask out these regions in all of the spectrum chunks.
