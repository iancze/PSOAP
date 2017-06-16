import numpy as np


import psoap
import os

PSOAP_dir = os.path.dirname(psoap.__file__)[:-5]

##################################################
# Constants
##################################################
c_ang = 2.99792458e18 #A s^-1
c_kms = 2.99792458e5 #km s^-1

#n @ 3000: 1.0002915686329712
#n @ 6000: 1.0002769832562917
#n @ 8000: 1.0002750477973053

n_air = 1.000277
c_ang_air = c_ang/n_air
c_kms_air = c_kms/n_air

h = 6.6260755e-27 #erg s

G = 6.67259e-8 #cm3 g-1 s-2
M_sun = 1.99e33 #g
R_sun = 6.955e10 #cm
pc = 3.0856776e18 #cm
AU = 1.4959787066e13 #cm

day = 24 * 3600 # [s]
deg = np.pi / 180 # [radians]
km = 1e5 # [cm]

L_sun = 3.839e33 #erg/s
R_sun = 6.955e10 #cm
F_sun = L_sun / (4 * np.pi * R_sun ** 2) #bolometric flux of the Sun measured at the surface

chunk_fmt = "chunk_{:}_{:.0f}_{:.0f}" # order, wl0, wl1

# Default order used to estimate SNR, indexed from 0
snr_default = 22

class ChunkError(Exception):
    '''
    Raised when there was a problem evaluating a specific chunk.
    '''
    def __init__(self, msg):
        self.msg = msg
