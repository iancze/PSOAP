#!/usr/bin/env python

import numpy as np
from astropy.io import fits
from specutils.io import read_fits
import EchelleJSON as ej


c_ang = 2.99792458e18 #A s^-1
c_kms = 2.99792458e5 #km s^-1

#n @ 3000: 1.0002915686329712
#n @ 6000: 1.0002769832562917
#n @ 8000: 1.0002750477973053

n_air = 1.000277
c_ang_air = c_ang/n_air
c_kms_air = c_kms/n_air

def convert_spectrum(fraw, fblaze, fout, BCV=False):
    '''
    param fraw: the raw counts
    param fblaze: the blaze-corrected spectrum
    param fout: the .json file to save to

    '''

    raw_list = read_fits.read_fits_spectrum1d(fraw)
    blaze_list = read_fits.read_fits_spectrum1d(fblaze)
    head = fits.getheader(fraw)
    try:
        BCV = head["BCV"]
    except KeyError:
        print("No BCV correction for", fraw)
        BCV = 0.0

    BCV_cor = np.sqrt((c_kms_air + BCV) / (c_kms_air - BCV))

    echelle_dict = {}
    npix = len(blaze_list[0].wavelength)

    # Do this for each order in the spectrum
    for i,(raw,blaze) in enumerate(zip(raw_list, blaze_list)):

        # Correct for the barycentric shift
        wl = blaze.wavelength.value

        # Scale the sigma values by the same blaze function that the raw fluxes were scaled by
        raw_flux = raw.flux.value
        raw_flux[raw_flux==0] = 1.0
        # Where the ratio values are 0, just set it to 1, since the noise will be 0 here too.
        ratio = blaze.flux.value/raw_flux
        sigma = ratio * np.sqrt(raw.flux.value)

        if BCV:
            wl = wl * BCV_cor

        order_dict = {  "wl":wl,
                        "fl":blaze.flux.value,
                        "sigma":sigma,
                        "mask": np.ones((npix,), dtype="bool")}

        echelle_dict["order_{}".format(i)] = order_dict

    UT = head["DATE-OBS"]
    echelle_dict["UT"] = UT
    try:
        echelle_dict["HJD"] = head["HJDN"]
    except KeyError:
        print("Spectrum does not have HJDN keyword", UT)
        print("Setting to 0.0")
        echelle_dict["HJD"] = 0.0

    try:
        echelle_dict["BCV"] = BCV
    except KeyError:
        print("Spectrum does not have BCV keyword", UT)
        print("Setting to 0.0")
        echelle_dict["BCV"] = 0.0

    ej.write(fout, echelle_dict)


if __name__=="__main__":

    import argparse

    parser = argparse.ArgumentParser(description="Process TRES echelle spectra into an EchelleJSON file.")
    parser.add_argument("rawfile", help="The un-blaze-corrected, un-flux-calibrated FITS file.")
    parser.add_argument("blazefile", help="The blaze-corrected, flux-calibrated FITS file.")
    parser.add_argument("outfile", help="Output Filename to contain the processed file. Should have no extension, *.hdf5 or *.npy added automatically.")

    parser.add_argument("-t", "--trim", type=int, default=6, help="How many pixels to trim from the front of the file. Default is 6")
    parser.add_argument("--BCV", action="store_true", help="If provided, do the barycentric correction.")

    parser.add_argument("--clobber", action="store_true", help="Overwrite existing outfile?")

    args = parser.parse_args()

    convert_spectrum(args.rawfile, args.blazefile, args.outfile, args.BCV)
