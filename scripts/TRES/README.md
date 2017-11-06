# TRES conversion scripts

To use these scripts, you'll want to download two versions of your spectra from the TRES archive, the 'raw' spectra, and those that have been corrected for the blaze function (those in the `b/` directory). We'll use both of these together to get an estimate of the noise in each pixel.

You'll also need to install the EchelleJSON package from here: https://github.com/iancze/EchelleJSON (I know, yet another format...)

Download the raw spectra into a sub-directory called `spectra_raw/`, and the blaze-corrected spectra into a directory called `spectra/`.

Then, execute the script and you'll have an HDF5 file at the end

    $ ./extract_files_BCV.sh

If you're setting up extraction scripts for a different telescope, you may consider looking inside the `TRESio_astropy.py` and `StuffHDF5.py` scripts for inspiration. And, the documentation on the HDF5 format in the [PSOAP docs](http://psoap.readthedocs.io/en/latest/configuration.html#processing-your-spectra-to-an-hdf5-file) might help.
