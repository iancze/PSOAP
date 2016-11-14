#!/usr/bin/env python

import argparse

parser = argparse.ArgumentParser(description="Initialize a new directory to do inference.")
parser.add_argument("model", choices=["SB1", "SB2", "ST3"], help="Which type of model to use, SB1, SB2, or SB3.")

args = parser.parse_args()

import inspect
import os
import shutil
import psoap

# Copy over the appropriate config.yaml file to current working directory
basedir = os.path.dirname(inspect.getfile(psoap))

shutil.copy(basedir + "/../data/config.{}.yaml".format(args.model), "config.yaml")
shutil.copy(basedir + "/../data/chunks.dat", "chunks.dat")
shutil.copy(basedir + "/../data/masks.dat", "masks.dat")
print("Copied config file for {} model to current working directory as config.yaml".format(args.model))
print("Copied chunks.dat and masks.dat to current working directory.")
