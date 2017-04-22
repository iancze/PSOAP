'''
Load the data files.
'''

import inspect
import os
import numpy as np
import h5py

import psoap
from psoap import constants as C


def redshift(wl, v):
    '''
    Redshift a collection of wavelengths. This means that a positive velocity corresponds to a lengthening (increase) of the wavelengths in the array.
    '''

    wl_red = wl * np.sqrt((C.c_kms + v)/(C.c_kms - v))
    return wl_red

def lredshift(lwl, v):
    '''
    Redshift spectra that are already in log-lambda. A positive velocity corresponds to a lengthing (increase) of the wavelengths of the array.
    '''

    lwl_red = lwl + v/C.c_kms
    return lwl_red


class Spectrum:
    def __init__(self, fname):
        '''
        Data structure for the raw spectra, stored in an HDF5 file.

        This is the main datastructure used to interact with your dataset. The key is getting your spectra into an HDF5 format first.

        Args:
            fname (string): location of the HDF5 file.

        Returns:
            Spectrum: the Spectrum object.
        '''

        data = h5py.File(fname,'r')

        self.wl = data["wl"][:]
        self.fl = data["fl"][:]
        self.sigma = data["sigma"][:]
        # Broadcast a 1D array of dates into a 2D array.
        dates = data["JD"][:]
        self.date1D = dates
        self.date = (np.ones_like(self.wl).T * dates[np.newaxis, np.newaxis, :]).T

        BCV = data["BCV"][:]
        self.BCV = (np.ones_like(self.wl).T * BCV[np.newaxis, np.newaxis, :]).T

        self.n_epochs, self.n_orders, self.n_pix = self.wl.shape

        data.close()

    def sort_by_SN(self, order=22):
        '''
        Sort the dataset in order of decreasing signal to noise. This is designed to make it easy to limit the analysis to the highest SNR epochs, if you wish to speed things up.

        Args:
            order (int): the order to calculate the signal-to-noise. By default, the TRES Mg b order is chosen, which is generally a good order for TRES data. If you are using data from a different telescope, you will likely need to adjust this value.

        '''

        # Since all spectra have been pre-normalized to 1.0, this is equivalent to simply finding
        # the spectrum that has the lowest average noise.
        noise_per_epoch = np.mean(self.sigma[:,order,:], axis=1)
        ind_sort = np.argsort(noise_per_epoch)

        self.wl = self.wl[ind_sort]
        self.fl = self.fl[ind_sort]
        self.sigma = self.sigma[ind_sort]
        self.date = self.date[ind_sort]
        self.date1D = self.date1D[ind_sort]
        self.BCV = self.BCV[ind_sort]


class Chunk:
    '''
    Hold a chunk of data. Each chunk is shape (n_epochs, n_pix) and has components wl, fl, sigma, date, and mask (all the same length).
    '''
    def __init__(self, wl, fl, sigma, date, mask=None):
        self.wl = wl
        self.lwl = np.log(wl) # Natural log!
        self.fl = fl
        self.sigma = sigma
        self.date = date
        self.date1D = date[:,0]

        if mask is None:
            self.mask = np.ones_like(self.wl, dtype="bool")
        else:
            self.mask = mask
        self.n_epochs, self.n_pix = self.wl.shape

    def apply_mask(self):
        '''
        Apply the mask to all of the attributes, so now we return 1D arrays.
        '''
        self.wl = self.wl[self.mask]
        self.lwl = self.lwl[self.mask]
        self.fl = self.fl[self.mask]
        self.sigma = self.sigma[self.mask]
        self.date = self.date[self.mask]
        self.N = len(self.wl)

    @classmethod
    def open(cls, order, wl0, wl1, limit=100, prefix=""):
        '''
        Load a spectrum from a directory link pointing to HDF5 output.
        :param fname: HDF5 file containing files on disk.
        '''

        #Open the HDF5 file, try to load each of these values.
        fname = prefix + C.chunk_fmt.format(order, wl0, wl1) + ".hdf5"
        import h5py
        with h5py.File(fname, "r") as hdf5:
            wl = hdf5["wl"]
            n_epochs = len(wl)
            if limit > n_epochs:
                limit = n_epochs
            wl = hdf5["wl"][:limit]
            fl = hdf5["fl"][:limit]
            sigma = hdf5["sigma"][:limit]
            date = hdf5["date"][:limit]
            mask = np.array(hdf5["mask"][:limit], dtype="bool") # Make sure it is explicitly boolean mask

            print("Limiting to {} epochs".format(len(wl)))
        #Although the actual fluxes and errors may be reasonably stored as float32, we need to do
        # all of the calculations in float64, and so we convert here.
        #The wl must be stored as float64, because of precise velocity issues.
        return cls(wl.astype(np.float64), fl.astype(np.float64), sigma.astype(np.float64), date.astype(np.float64), mask)

    def save(self, order, wl0, wl1, prefix=""):
        fname = prefix + C.chunk_fmt.format(order, wl0, wl1) + ".hdf5"
        shape = self.wl.shape
        import h5py
        with h5py.File(fname, "w") as hdf5:

            wl = hdf5.create_dataset("wl", shape, dtype="f8")
            wl[:] = self.wl

            fl = hdf5.create_dataset("fl", shape, dtype="f8")
            fl[:] = self.fl

            sigma = hdf5.create_dataset("sigma", shape, dtype="f8")
            sigma[:] = self.sigma

            date = hdf5.create_dataset("date", shape, dtype="f8")
            date[:] = self.date

            mask = hdf5.create_dataset("mask", shape, dtype="bool")
            mask[:] = self.mask

            hdf5.close()

# Load the HDF5 files into global scope
basedir = os.path.dirname(inspect.getfile(psoap))
lkca14 = Spectrum(basedir + "/../data/LkCa14.hdf5")
