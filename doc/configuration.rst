.. _configuration:

Configuration
=============

To start a new project, first decide what type of model you might like to use and navigate to a fresh directory. For example, let's choose an ``SB2``

Then, within your new directory ::

    $ psoap_initialize.py --model SB2

This will copy the appropriate scripts to your directory. You are encouraged to use your favorite text editor and inspect the contents of the ``config.yaml`` file, which will be used frequently by this package.

One of the first things you should notice is the field ``data_file: data.hdf5``.

Processing your spectra to an HDF5 file
=======================================

Complicated.
