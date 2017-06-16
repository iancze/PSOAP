====
Data
====

Data formats
------------

PSOAP relies upon chunks of data. When working with real data, there are a few things to keep in mind.

First, it may so happen that certain pixels may need to be masked, for example due to cosmic ray hits. This means that actual data chunks will probably have an un-equal number of pixels per epoch. This is OK.

So, all chunks will be generated with their full complement of data, but when executing any inference routines, the masks will be applied to the data.

All data and chunks are stored in an HDF5 format.

Data module
-----------

.. automodule:: psoap.data
    :members:


Utils module
------------

.. automodule:: psoap.utils
    :members:
