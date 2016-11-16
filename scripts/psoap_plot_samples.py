#!/usr/bin/env python

import argparse

parser = argparse.ArgumentParser(description="Measure statistics across multiple chains.")
parser.add_argument("--burn", type=int, default=0, help="How many samples to discard from the beginning of the chain for burn in.")
parser.add_argument("--config", help="name of the config file used for the run.", default="config.yaml")
parser.add_argument("--tri", help="Plot the triangle too.", action="store_true")
parser.add_argument("--interactive", action="store_true", help="Pop up the walker window so that you can zoom around.")
parser.add_argument("--cov", action="store_true", help="Estimate the optimal covariance to tune MH jumps.")

args = parser.parse_args()
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import psoap
from psoap import utils

# This first bit of code is run for every invocation of the script
flatchain = np.load("flatchain.npy")
flatchain = flatchain[args.burn:]

print("Last sample is ")
print(flatchain[-1])

# Load the lnprobabilities and truncate for burn in
lnprobs = np.load("lnprob.npy")
lnprobs = lnprobs[args.burn:]

niter, ndim = flatchain.shape

if args.cov:
    cov = utils.estimate_covariance(flatchain)
    np.save("opt_jump.npy", cov)

# If we can tell what type of model we were sampling, we can give everything appropriate labels.
import yaml
f = open(args.config)
config = yaml.load(f)
f.close()

model = config["model"]
labels = utils.get_labels(model, config["fix_params"])


import matplotlib.pyplot as plt

fig, ax = plt.subplots(nrows=(ndim + 1), ncols=1, figsize=(10, 1.5 * ndim))

iterations = np.arange(niter)

step = 100

#Plot the lnprob on top
ax[0].plot(iterations, lnprobs, color="b")
ax[0].set_ylabel("lnprob")

for i in range(ndim):
    ax[i +1].plot(iterations, flatchain[:, i], color="k")
    ax[i+1].set_ylabel(labels[i])

ax[-1].set_xlabel("Iteration")

if args.interactive:
    fig.subplots_adjust(bottom=0.0, top=1.0, right=1.0, hspace=0.0)
    plt.show()
else:
    fig.savefig("chain.png", dpi=300)

def hdi(samples, bins=40):

    hist, bin_edges = np.histogram(samples, bins=bins, density=True)
    # convert bin_edges into bin centroids
    bin_centers = bin_edges[:-1] + np.diff(bin_edges)

    dbin = bin_edges[1] - bin_edges[0]
    nbins = len(bin_centers)

    # Now, sort all of the bin heights in decreasing order of probability
    indsort = np.argsort(hist)[::-1]
    histsort = hist[indsort]
    binsort = bin_centers[indsort]

    binmax = binsort[0]

    prob = histsort[0] * dbin
    i = 0
    while prob < 0.683:
        i += 1
        prob = np.sum(histsort[:i] * dbin)

    level = histsort[i]

    indHDI = hist > level
    binHDI = bin_centers[indHDI]

    # print("Ranges: low: {}, max: {}, high: {}".format(binHDI[0], binmax, binHDI[-1]))
    # print("Diffs: max:{}, low:{}, high:{}, dbin:{}".format(binmax, binmax - binHDI[0], binHDI[-1]-binmax, dbin))

    # Now, return everything necessary to make a plot
    # "lower": lower confidence interval
    # "upper": upper confidence interval
    #
    # "plus": + errorbar
    # "minus": - errorbar

    plus = binHDI[-1]-binmax
    minus = binmax - binHDI[0]

    return {"bin_centers":bin_centers, "hist":hist, "max":binmax, "lower":binHDI[0], "upper":binHDI[-1], "plus":plus, "minus":minus, "dbin":dbin, "level":level}


def plot_hdis(flatchain, fname="hdi.png"):

    # Plot the bins with highlighted ranges
    fig,ax = plt.subplots(ncols=1, nrows=ndim, figsize=(6, 1.5 * ndim))

    for i in range(flatchain.shape[1]):
        vals = hdi(flatchain[:,i])

        ax[i].plot(vals["bin_centers"], vals["hist"], ls="steps-mid")
        ax[i].axhline(vals["level"], ls=":", color="k")
        ax[i].set_xlabel(labels[i])
        ax[i].set_ylabel("probability")

        print(labels[i])
        print("Ranges: low: {}, max: {}, high: {}".format(vals["lower"], vals["max"], vals["upper"]))
        print("Diffs: max:{}, low:{}, high:{}, dbin:{}".format(vals["max"], vals["minus"], vals["plus"], vals["dbin"]))
        print()

    fig.subplots_adjust(hspace=0.5, bottom=0.05, top=0.99)
    fig.savefig(fname)

try:
    plot_hdis(flatchain)
except IndexError:
    pass


# Make the triangle plot
if args.tri:
    import corner
    figure = corner.corner(flatchain, bins=30, labels=labels, quantiles=[0.16, 0.5, 0.84], plot_contours=True, plot_datapoints=False, show_titles=True)
    figure.savefig("triangle.png")
else:
    print("Not plotting triangle, no --tri flag.")
