import inspect
import os
import numpy as np
import h5py

import psoap
from psoap import constants as C


def redshift(wl, v):
    '''
    Redshift a vector of wavelengths. A positive velocity corresponds to a lengthening (increase) of the wavelengths in the array.

    Args:
        wl (np.array, arbitrary shape): the input wavelengths
        velocity (float): the velocity by which to redshift the wavelengths

    Returns:
        np.array: A redshifted version of the wavelength vector
    '''

    wl_red = wl * np.sqrt((C.c_kms + v)/(C.c_kms - v))
    return wl_red

def lredshift(lwl, v):
    '''
    Redshift a vector of wavelengths that are already in log-lamba (natural log). A positive velocity corresponds to a lengthening (increase) of the wavelengths in the array.

    Args:
        wl (np.array, arbitrary shape): the input ln(wavelengths).
        velocity (float): the velocity by which to redshift the wavelengths

    Returns:
        np.array: A redshifted version of the wavelength vector
    '''

    lwl_red = lwl + v/C.c_kms
    return lwl_red

def replicate_wls(lwls, velocities, mask):
    '''
    Using the set of velocities calculated from an orbit, copy and *blue*-shift the input ln(wavelengths), so that they correspond to the rest-frame wavelengths of the individual components. This routine is primarily for producing  replicated ln-wavelength vectors ready to feed to the GP routines.

    Args:
        lwls (1D np.array with length ``(n_epochs * n_good_pixels)``): this dataproduct is the 1D representation of the natural log of the (masked) input wavelength vectors. The masking process naturally makes it 1D.
        velocities (2D np.array with shape ``(n_components, n_epochs)``) : a set of velocities determined from an orbital model.
        mask : the np.bool mask used to select the good datapoints. It is necessary for properly replicating the velocities to the right epoch.

    Returns:
        np.array: A 2D ``(n_components, n_epochs * n_good_pixels)`` shape array of the wavelength vectors *blue*-shifted according to the velocities. This means that for each component, the arrays are flattened into 1D vectors.
    '''

    # n_epochs_lwl = len(lwls)
    n_components, n_epochs = velocities.shape
    # assert n_epochs_lwl == n_epochs, "There is a mismatch between the number of epochs implied by the log-wavelength vector ({:}) and the number of epochs implied by the orbital velocities {:}".format(n_epochs_lwl, n_epochs)

    n_good_pix = np.sum(mask)

    lwls_out = np.empty((n_components, n_good_pix), dtype=np.float64)
    for i in range(n_components):
        lwls_out[i] = lredshift(lwls, (-velocities[i][:,np.newaxis] * np.ones_like(mask))[mask])

    return lwls_out

class Spectrum:
    '''
    Data structure for the raw spectra, stored in an HDF5 file.

    This is the main datastructure used to interact with your dataset. The key is getting your spectra into an HDF5 format first.

    Args:
        fname (string): location of the HDF5 file.

    Returns:
        Spectrum: the instantiated Spectrum object.
    '''

    def __init__(self, fname):


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
        self.wl = wl #: wavelength vector
        self.lwl = np.log(wl) #: natural log of the wavelength vector
        self.fl = fl #: flux vector
        self.sigma = sigma #: measurement uncertainty vector
        self.date = date #: date vector
        self.date1D = date[:,0] #: data vector of length `n_epochs`

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
