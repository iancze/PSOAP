import argparse

parser = argparse.ArgumentParser(description="Process all the data")
parser.add_argument("target", help="The name of the object.")
args = parser.parse_args()

import numpy as np
import h5py
import EchelleJSON as ej
from astropy.io import ascii, fits
from scipy.interpolate import interp1d


# Load/Create an HDF5 file for LkCa14
f = h5py.File(args.target + '.hdf5','a')

# Load in the dates
dates = np.loadtxt("HJD.txt", ndmin=1)

f["JD"] = dates

rdata = ascii.read("rectification.txt")

# Now insert these into a dictionary by order
rect_regions = {}
for row in rdata:
    rect_regions[row["order"]] = [row["wl0"], row["wl1"]]

def load_sens(filename):
    sfunc, hdr = fits.getdata(filename, header=True)
    wl = hdr['CDELT1'] * np.arange(len(sfunc)) + hdr['CRVAL1']
    return [wl, sfunc]

n_epochs = len(dates)
n_orders = 51
n_pix = 2304 - 6

files14 = open("files.txt").readlines()
fnames14 = [file[:-5] + "json" for file in files14]

# Create empty dataset to store spectra
wl14 = np.empty((n_epochs, n_orders, n_pix))
fl14 = np.empty((n_epochs, n_orders, n_pix))
sigma14 = np.empty((n_epochs, n_orders, n_pix))

# Create an array to store the BCVs
bcv14 = np.empty(n_epochs)

# Final dataset will be wls, it will have shape (n_epochs, n_orders, n_pix)
# Load each spectrum using EchelleJSON,
# Assemble into a 51 * 2298 sized dataset of wavelengths and fluxes
# The 2298 is because we trim first 6 pixels due to reduction error
for i,name in enumerate(fnames14):
    edict = ej.read("jsons_BCV/" + name)

    bcv = edict["BCV"]
    bcv14[i] = bcv

    for j in range(51):
        odict = edict["order_{}".format(j)]
        wl = odict["wl"][6:]
        fl = odict["fl"][6:]
        sigma = odict["sigma"][6:]

        # Load the sensfunc for this order
        # wl_s, s_func = load_sens('../flux/sens_BD+28.{:0>4d}.fits'.format(j + 1))
        # f_func = 10**(0.4 * s_func)

        # interp = interp1d(wl_s, f_func, kind="linear", fill_value="extrapolate")

        # flux calibrate the spectrum and the noise
        # fl /= interp(wl)
        # sigma /= interp(wl)

        # Normalize each order by the median flux in that order (both fl and sigma)
        wl0, wl1 = rect_regions[j]
        ind = (wl > wl0) & (wl < wl1)

        med = np.median(fl[ind])
        fl /= med
        sigma /= med

        wl14[i,j,:] = wl
        fl14[i,j,:] = fl
        sigma14[i,j,:] = sigma

# Save these datasets to the HDF5
f["wl"] = wl14
f["fl"] = fl14
f["sigma"] = sigma14
f["BCV"] = bcv14

f["PI"] = "PI Ian Czekala and Jason Dittmann. Contact jason.dittmann@gmail.com or iancze@gmail.com for usage permission."

# Now save this
f.close()
