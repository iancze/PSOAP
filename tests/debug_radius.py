import matplotlib.pyplot as plt
import numpy as np
from psoap import constants as C

dpc = 44.6 # pc
a = 0.0706 * dpc # [AU]
e = 0.9754
i = 49.7 # [deg]
omega = 127.31 # [deg]
Omega = 1.9 # [deg]
T0 = 2449571.037 # [Julian Date]
M_2 = 1.20 # [M_sun]
M_tot = 1.28 + M_2 # [M_sun]
gamma = 5.76 # [km/s]

def xy_B(f):
    # find the reduced radius
    r = a * (1 - e**2) / (1 + e * np.cos(f)) # [AU] # factor of pi here for omega_2
    r2 = -r * (M_tot - M_2) / M_tot # [AU]

    r_swap = a * (1 - e**2) / (1 + e * np.cos(f + np.pi))
    r2_swap = r_swap * (M_tot - M_2) / M_tot # [AU]

    return (r, r2, r_swap, r2_swap)

fs = np.linspace(0, 2 * np.pi, num=50)

rs, r2s, r_swaps, r2_swaps = xy_B(fs)

fig, ax = plt.subplots(nrows=4, sharex=True, sharey=True)
ax[0].plot(rs, ".")
ax[1].plot(r2s, ".")
ax[2].plot(r_swaps, ".")
ax[3].plot(r2_swaps, ".")
fig.savefig("plots/debug_f.png")
