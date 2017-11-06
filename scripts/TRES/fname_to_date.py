import EchelleJSON as ej
import numpy as np

# Read all of the file names and convert the strings to UT and JD

f = open("files.txt")
files = ["{}.json".format(ff[:-6]) for ff in f.readlines()]

# Read the HJDN field
f = open("HJD.txt", "w")
for ff in files:
    edict = ej.read("jsons_BCV/{}".format(ff))
    HJD = edict["HJD"]
    f.write("{}\n".format(HJD))
f.close()
