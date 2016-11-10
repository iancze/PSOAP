#!/usr/bin/env python

import argparse

parser = argparse.ArgumentParser(description="Initialize a new directory to do inference.")
parser.add_argument("model", help="Which type of model to use, SB1, SB2, or SB3.")

args = parser.parse_args()


# Copy over the appropriate config.yaml file to current working directory
