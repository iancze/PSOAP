import numpy as np
import EchelleJSON as ej
import matplotlib.pyplot as plt
from astropy.io import ascii, fits
from scipy.interpolate import interp1d

f = open("files.txt")
files = ["{}.json".format(ff[:-6]) for ff in f.readlines()]

def load_sens(filename):
    sfunc, hdr = fits.getdata(filename, header=True)
    wl = hdr['CDELT1'] * np.arange(len(sfunc)) + hdr['CRVAL1']
    return [wl, sfunc]

rdata = ascii.read("rectification.txt")

# Now insert these into a dictionary by order
rect_regions = {}
for row in rdata:
    rect_regions[row["order"]] = [row["wl0"], row["wl1"]]


def plot(order, BCV=True):

    fig,ax = plt.subplots(nrows=2, figsize=(12,8), sharex=True)

    for ff in files:
        if BCV:
            edict = ej.read("jsons_BCV/{}".format(ff))
        else:
            edict = ej.read("jsons/{}".format(ff))
        odict = edict["order_{}".format(order)]

        wl = odict["wl"]
        fl = odict["fl"]


        # Load the sensfunc for this ordre
        wl_s, s_func = load_sens('../flux/sens_BD+28.{:0>4d}.fits'.format(order + 1))
        f_func = 10**(0.4 * s_func)

        interp = interp1d(wl_s, f_func, kind="linear", fill_value="extrapolate")

        ax[0].plot(wl_s, f_func)

        # flux calibrate the spectrum
        fl /= interp(wl)

        # Normalize each order by the median flux in that order
        wl0, wl1 = rect_regions[order]
        ind = (wl > wl0) & (wl < wl1)

        med = np.median(fl[ind])

        fl /= med

        ax[1].plot(wl, fl)
        ax[1].errorbar(wl[ind], fl[ind], color="r")

    if BCV:
        fig.savefig("plots_BCV/{}.png".format(order), dpi=300)
    else:
        fig.savefig("plots/{}.png".format(order), dpi=300)

    plt.close('all')


if __name__=="__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Plot a single order, for all spectra.")
    parser.add_argument("order", type=int, help="The echelle order to plot (indexed from 0).")
    parser.add_argument("--BCV", action="store_true", help="Plot the BCV-corrected spectra?")
    args = parser.parse_args()

    plot(args.order, args.BCV)
