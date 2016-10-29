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

# Data structure for the LkCa14 and LkCa15 spectra
class Spectrum:
    def __init__(self, fname):
        '''
        Load a spectrum from HDF5 file.
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
        Call this function to reorder all spectra from higest to lowest S/N.
        Optionally provide the order that should be used to determine the S/N.
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
        

# Load the HDF5 files into global scope
basedir = os.path.dirname(inspect.getfile(psoap))
lkca14 = Spectrum(basedir + "/../data/LkCa14.hdf5")
gwori = Spectrum(basedir + "/../data/GWOri.hdf5")
