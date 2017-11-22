#!/usr/bin/env julia

# Since numpy is so bad at drawing multivariate samples, this is a Julia routine to read in a mean vector and covariance matrix and then generate multivariate Gaussian draws.

using ArgParse

s = ArgParseSettings()
@add_arg_table s begin
    "mu"
    help = "Mean vector"
    "cov"
    help = "Covariance matrix."
    "--draws"
    help = "How many random draws to generate."
    default = 1
    "--out"
    help = "Name of output file to save draws."
    default = "draws.npy"
end

parsed_args = parse_args(ARGS, s)

using NPZ
using Distributions

# Load the mean vector and covariance matrix

mu = npzread(parsed_args["mu"])
cov = npzread(parsed_args["cov"])


dist = MvNormal(mu, cov)

x = rand(dist, parsed_args["draws"])

# save
npzwrite(parsed_args["out"], x)
