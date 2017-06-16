
.. module:: psoap

.. note:: This tutorial was generated from an IPython notebook that can be
          downloaded `here <../../_static/notebooks/tutorial.ipynb>`_.

.. _tutorial:

Tutorial
========

This tutorial works through the example of fitting the LP661-13 dataset
which appeared in Czekala et al. 2017. The spectra were originally
acquired by Dittmann et al. 2017, and can be downloaded already in HDF5
format HERE.

If you are looking to use data from a different telescope, you will need
to process these spectra into a format like this HDF5 file. Some
additional notes on how to do this are HERE.

This tutorial assumes that you have already followed the installation
instructions, HERE.

Visualizing the dataset
=======================

Before we do any analysis with PSOAP, it's a good idea to plot up all of
your data. That way, we can see if there are any regions of the spectrum
we may want to pay special attention to

.. code:: python

    !psoap_hdf5_exploder.py --help


.. parsed-literal::

    usage: psoap_hdf5_exploder.py [-h] [--orders [ORDERS [ORDERS ...]]] [--SNR]
                                  [--topo]
    
    Make summary plots for a full HDF5 dataset.
    
    optional arguments:
      -h, --help            show this help message and exit
      --orders [ORDERS [ORDERS ...]]
                            Which orders to plot. By default, all orders are
                            plotted. Can add more than one order in a spaced list,
                            e.g., --orders 22 23 24 but not --orders=22,23,24
      --SNR                 Plot spectra in order of highest SNR first, instead of
                            by date. Default is by date.
      --topo                Plot spectra in topocentric frame instead of
                            barycentric frame. Default is barycentric frame.


This will produce a bunch of plots in a newly-created ``plots``
directory.

Creating a configuration file
=============================

PSOAP generally relies upon a configuration text file for many of the
project-specific settings. To create one from scratch, use the
``psoap_initialize.py`` command

.. code:: python

    !psoap_initialize.py --help


.. parsed-literal::

    Using Python Version 3.6.1 |Anaconda 4.4.0 (64-bit)| (default, May 11 2017, 13:09:58) 
    [GCC 4.4.7 20120313 (Red Hat 4.4.7-1)]
    usage: psoap_initialize.py [-h] [--check] [--model {SB1,SB2,ST3}]
    
    Initialize a new directory to do inference.
    
    optional arguments:
      -h, --help            show this help message and exit
      --check               To help folks check whether the package was installed
                            properly.
      --model {SB1,SB2,ST3}
                            Which type of model to use, SB1, SB2, ST1, or SB3.


For this project, we'll do

.. code:: python

    !psoap_initialize.py --model SB2

Open up the new ``config.yaml`` file in your directory with your
favorite text editor, and familiarize yourself with the settings. For
more information, check out :ref:``configuration``.

Setting up the chunks file
==========================

Because Gaussian processes are generally very computationally intensive,
we'll need to split the spectrum up into chunks so that it can be
processed in parallel. The easiest way to get started is with
``psoap_generate_chunks.py``

.. code:: python

    !psoap_generate_chunks.py --help


.. parsed-literal::

    usage: psoap_generate_chunks.py [-h] [--pixels PIXELS] [--overlap OVERLAP]
                                    [--start START] [--end END]
    
    Auto-generate comprehensive chunks.dat file, which can be later edited by
    hand.
    
    optional arguments:
      -h, --help         show this help message and exit
      --pixels PIXELS    Roughly how many pixels should we keep in each chunk?
      --overlap OVERLAP  How many pixels of overlap to aim for.
      --start START      Starting wavelength.
      --end END          Ending wavelength.


Try running this command with the default values, and then open up the
``chunks.dat`` file that now exists in your local directory. You can try
playing around with the specific values, but if you want to regenerate
the file, you'll need to delete the existing ``chunks.dat`` file from
the directory first. To make things go quickly for this tutorial, we're
only going to use a limited section of the spectrum. Therefore, we're
going to open up ``chunks.dat`` and delete the chunks blueward of XX AA
and redward of AA, leaving only 3 actual chunks. If you were doing this
for real, you could choose your chunks more wisely. The inference
procedure is set up so that it's one chunk per CPU core, so generally
feel free to use as many chunks as you have CPU cores, since there is no
additional time penalty.

