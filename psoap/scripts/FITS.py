import numpy
from scipy.interpolate import UnivariateSpline

from astropy.io import fits

from psoap import constants as C


import yaml

try:
    f = open("config.yaml")
    config = yaml.load(f)
    f.close()
except FileNotFoundError as e:
    print("You need to copy a config.yaml file to this directory, and then edit the values to your particular case.")
    raise

# read in the chunks.dat file
chunks = ascii.read(config["chunk_file"])
print("Optimizing the calibration for the following chunks of data")
print(chunks)


# Go through each chunk and optimize the calibration.
for chunk_index,chunk in enumerate(chunks):

    order, wl0, wl1 = chunk
    fname = "plots_" + C.chunk_fmt.format(order, wl0, wl1) + "/f.npy"



    fname = "plots_" + C.chunk_fmt.format(order, wl0, wl1) + "/g.npy"


# Assemble a master spectrum from all of the chunks.


# In overlap regions, we should average things together.


# Interpolate everything to a common wavelength grid


# Then, create an actual FITS file

def calculate_dv(wl):
    '''
    Given a wavelength array, calculate the minimum ``dv`` of the array.
    :param wl: wavelength array
    :type wl: np.array
    :returns: (float) delta-v in units of km/s
    '''
    return C.c_kms * np.min(np.diff(wl)/wl[:-1])


def create_log_lam_grid(dv, wl_start=3000., wl_end=13000.):
    '''
    Create a log lambda spaced grid with ``N_points`` equal to a power of 2 for
    ease of FFT.
    :param wl_start: starting wavelength (inclusive)
    :type wl_start: float, AA
    :param wl_end: ending wavelength (inclusive)
    :type wl_end: float, AA
    :param dv: upper bound on the size of the velocity spacing (in km/s)
    :type dv: float
    :returns: a wavelength dictionary containing the specified properties. Note
        that the returned dv will be <= specified dv.
    :rtype: wl_dict
    '''
    assert wl_start < wl_end, "wl_start must be smaller than wl_end"

    CDELT_temp = np.log10(dv/C.c_kms + 1.)
    CRVAL1 = np.log10(wl_start)
    CRVALN = np.log10(wl_end)
    N = (CRVALN - CRVAL1) / CDELT_temp
    NAXIS1 = 2
    while NAXIS1 < N: #Make NAXIS1 an integer power of 2 for FFT purposes
        NAXIS1 *= 2

    CDELT1 = (CRVALN - CRVAL1) / (NAXIS1 - 1)

    p = np.arange(NAXIS1)
    wl = 10 ** (CRVAL1 + CDELT1 * p)
    return {"wl": wl, "CRVAL1": CRVAL1, "CDELT1": CDELT1, "NAXIS1": NAXIS1}


# Take the reconstruced spectra and put them into a FITS file
def create_fits(filename, fl, CRVAL1, CDELT1, dict=None):
    '''Assumes that wl is already log lambda spaced'''

    hdu = fits.PrimaryHDU(fl)
    head = hdu.header
    head["DISPTYPE"] = 'log lambda'
    head["DISPUNIT"] = 'log angstroms'
    head["CRPIX1"] = 1.

    head["CRVAL1"] = CRVAL1
    head["CDELT1"] = CDELT1
    head["DC-FLAG"] = 1

    if dict is not None:
        for key, value in dict.items():
            head[key] = value

    hdu.writeto(filename)
