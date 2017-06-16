import numpy as np
from scipy.optimize import fsolve

from psoap import constants as C

class Binary:
    '''
    Binary orbital model that can deliver absolute astrometric position, relative astrometric position (B relative to A), and radial velocities for A and B.

    Args:
        a (float): semi-major axis [AU]
        e (float): eccentricity (must be between ``[0.0, 1.0)``)
        i (float): inclination [deg]
        omega (float): argument of periastron [degrees]
        Omega (float): position angle of the ascending node [deg] east of north
        T0 (float): epoch of periastron passage [JD]
        M_tot (float): sum of the masses [M_sun]
        M_2 (float): mass of B [M_sun]
        gamma (float): systemic velocity (km/s)
        obs_dates (1D np.array): dates of observation (JD)
    '''
    def __init__(self, a, e, i, omega, Omega, T0, M_tot, M_2, gamma, obs_dates=None, **kwargs):
        assert (e >= 0.0) and (e < 1.0), "Eccentricity must be between [0, 1)"
        self.a = a # [AU] semi-major axis
        self.e = e # eccentricity
        self.i = i # [deg] inclination
        self.omega = omega # [deg] argument of periastron
        self.Omega = Omega # [deg] east of north
        self.T0 = T0 # [JD]
        self.M_tot = M_tot # [M_sun]
        self.M_2 = M_2 # [M_sun]
        self.gamma = gamma # [km/s]

        # Update the derived RV quantities
        self.q = self.M_2 / (self.M_tot - self.M_2) # [M2/M1]
        self.P = np.sqrt(4 * np.pi**2 / (C.G * self.M_tot * C.M_sun) * (self.a * C.AU)**3) / (60 * 60 * 24)# [days]
        self.K = np.sqrt(C.G/(1 - self.e**2)) * self.M_2 * C.M_sun * np.sin(self.i * np.pi/180.) / np.sqrt(self.M_tot * C.M_sun * self.a * C.AU) * 1e-5 # [km/s]

        # If we are going to be repeatedly predicting the orbit at a sequence of dates,
        # just store them to the object.
        self.obs_dates = obs_dates


    def _theta(self, t):
        '''Calculate the true anomoly for the A-B orbit.
        Input is in days.'''

        # t is input in seconds

        # Take a modulus of the period
        t = (t - self.T0) % self.P

        f = lambda E: E - self.e * np.sin(E) - 2 * np.pi * t/self.P
        E0 = 2 * np.pi * t / self.P

        E = fsolve(f, E0)[0]

        th = 2 * np.arctan(np.sqrt((1 + self.e)/(1 - self.e)) * np.tan(E/2.))

        if E < np.pi:
            return th
        else:
            return th + 2 * np.pi

    def _v1_f(self, f):
        '''Calculate the component of A's velocity based on only the inner orbit.
        f is the true anomoly of this inner orbit.'''

        return self.K * (np.cos(self.omega * np.pi/180 + f) + self.e * np.cos(self.omega * np.pi/180))

    def _v2_f(self, f):
        '''Calculate the component of B's velocity based on only the inner orbit.
        f is the true anomoly of this inner orbit.'''

        return -self.K/self.q * (np.cos(self.omega * np.pi/180 + f) + self.e * np.cos(self.omega * np.pi/180))

    # Get the position of A in the plane of the orbit
    def _xy_A(self, f):
        # find the reduced radius
        r = self.a * (1 - self.e**2) / (1 + self.e * np.cos(f)) # [AU]
        r1 = r * self.M_2 / self.M_tot # [AU]

        x = r1 * np.cos(f)
        y = r1 * np.sin(f)

        return (x,y)

    # Get the position of B in the plane of the orbit
    def _xy_B(self, f):
        # find the reduced radius
        r = self.a * (1 - self.e**2) / (1 + self.e * np.cos(f)) # [AU]
        r2 = -r * (self.M_tot - self.M_2) / self.M_tot # [AU]

        x = r2 * np.cos(f)
        y = r2 * np.sin(f)

        return (x,y)

    def _xy_AB(self, f):
        r = self.a * (1 - self.e**2) / (1 + self.e * np.cos(f)) # [AU]
        x = r * np.cos(f)
        y = r * np.sin(f)

        return (x,y)

    # position of A relative to center of mass
    def _XYZ_A(self, f):

        # find the reduced radius
        r = self.a * (1 - self.e**2) / (1 + self.e * np.cos(f)) # [AU]
        r1 = r * self.M_2 / self.M_tot # [AU]

        Omega = self.Omega * np.pi / 180
        omega = self.omega * np.pi / 180 # add in pi to swap the periapse
        i = self.i * np.pi / 180
        X = r1 * (np.cos(Omega) * np.cos(omega + f) - np.sin(Omega) * np.sin(omega + f) * np.cos(i))
        Y = r1 * (np.sin(Omega) * np.cos(omega + f) + np.cos(Omega) * np.sin(omega + f) * np.cos(i))
        Z = r1 * (np.sin(omega + f) * np.sin(i))

        return (X, Y, Z) # [AU]

    # position of B relative to center of mass
    def _XYZ_B(self, f):

        # find the reduced radius
        r = self.a * (1 - self.e**2) / (1 + self.e * np.cos(f)) # [AU]
        r2 = -r * (self.M_tot - self.M_2) / self.M_tot # [AU]

        Omega = self.Omega * np.pi / 180
        omega = self.omega * np.pi / 180
        i = self.i * np.pi / 180
        X = r2 * (np.cos(Omega) * np.cos(omega + f) - np.sin(Omega) * np.sin(omega + f) * np.cos(i))
        Y = r2 * (np.sin(Omega) * np.cos(omega + f) + np.cos(Omega) * np.sin(omega + f) * np.cos(i))
        Z = r2 * (np.sin(omega + f) * np.sin(i))

        return (X, Y, Z) # [AU]

    def _XYZ_AB(self, f):
        # radius of B to A
        r = self.a * (1 - self.e**2) / (1 + self.e * np.cos(f)) # [AU]
        Omega = self.Omega * np.pi / 180
        omega = self.omega * np.pi / 180
        i = self.i * np.pi / 180
        X = r * (np.cos(Omega) * np.cos(omega + f) - np.sin(Omega) * np.sin(omega + f) * np.cos(i))
        Y = r * (np.sin(Omega) * np.cos(omega + f) + np.cos(Omega) * np.sin(omega + f) * np.cos(i))
        Z = r * (np.sin(omega + f) * np.sin(i))

        # X is north, Y is east.
        return (X, Y, Z) # [AU]

    def _get_orbit_t(self, t):
        '''
        Given a time, calculate all of the orbital quantaties we might be interseted in.
        returns (v_A, v_B, (x,y) of A, (x,y) of B, and x,y of B relative to A)
        '''

        # Get the true anomoly "f" from time
        f = self._theta(t)

        # Feed this into the orbit equation and add the systemic velocity
        vA = self._v1_f(f) + self.gamma
        vB = self._v2_f(f) + self.gamma

        XYZ_A = self._XYZ_A(f)
        XYZ_B = self._XYZ_B(f)
        XYZ_AB = self._XYZ_AB(f)
        xy_A = self._xy_A(f)
        xy_B = self._xy_B(f)
        xy_AB = self._xy_AB(f)

        return (vA, vB, XYZ_A, XYZ_B, XYZ_AB, xy_A, xy_B, xy_AB)


    def get_orbit(self, dates=None):
        '''
        Deliver only the main quantities useful for performing a joint astrometric + RV fit to real data, namely
        the radial velocities ``vA``, ``vB``, the relative offsets ``rho_AB``, and relative position angles ``theta_AB``, for all dates provided. Relative offsets are provided in AU, and so must be converted to arcseconds after assuming a distance to the system. Relative position angles are given in degrees east of north.

        Args:
            dates (optional): if provided, calculate quantities at this new vector of dates, rather than the one provided when the object was initialized.

        Returns:
            np.array: A ``(4, npoints)`` shape array of ``[vA, vB, rho_AB, theta_AB]``
        '''

        if dates is None and self.obs_dates is None:
            raise RuntimeError("Must provide input dates or specify observation dates upon creation of orbit object.")

        if dates is None and self.obs_dates is not None:
            dates = self.obs_dates

        dates = np.atleast_1d(dates)
        N = len(dates)

        vAs = np.empty(N, dtype=np.float64)
        vBs = np.empty(N, dtype=np.float64)
        rho_ABs = np.empty(N, dtype=np.float64)
        theta_ABs = np.empty(N, dtype=np.float64)

        for i,date in enumerate(dates):
            vA, vB, XYZ_A, XYZ_B, XYZ_AB, xy_A, xy_B, xy_AB = self._get_orbit_t(date)
            vAs[i] = vA
            vBs[i] = vB

            # Calculate rho, theta from XY_AB
            X, Y, Z = XYZ_AB

            rho = np.sqrt(X**2 + Y**2) # [AU]
            theta = np.arctan2(Y, X) * 180/np.pi # [Deg]
            if theta < 0: # ensure that 0 <= theta <= 360
                theta += 360.

            rho_ABs[i] = rho
            theta_ABs[i] = theta

        return np.vstack((vAs, vBs, rho_ABs, theta_ABs))

    def get_full_orbit(self, dates=None):
        '''
        Deliver the full set of astrometric and radial velocity quantities, namely
        the radial velocities ``vA``, ``vB``, the position of A and B relative to the center of mass in the plane of the sky (``XY_A`` and ``XY_B``, respectively), the position of B relative to the position of A in the plane of the sky (``XY_AB``), the position of A and B in the plane of the orbit (``xy_A`` and ``xy_B``, respectively), and the position of B relative to the position of A in the plane of the orbit (``xy_AB``), for all dates provided. All positions are given in units of AU, and so must be converted to arcseconds after assuming a distance to the system.

        Args:
            dates (optional): if provided, calculate quantities at this new vector of dates, rather than the one provided when the object was initialized.

        Returns:
            np.array: A ``(8, npoints)`` shape array of ``[vA, vB, XYZ_A, XYZ_B, XYZ_AB, xy_A, xy_B, xy_AB]``
        '''


        if dates is None and self.obs_dates is None:
            raise RuntimeError("Must provide input dates or specify observation dates upon creation of orbit object.")

        if dates is None and self.obs_dates is not None:
            dates = self.obs_dates

        dates = np.atleast_1d(dates)
        N = len(dates)

        vAs = np.empty(N, dtype=np.float64)
        vBs = np.empty(N, dtype=np.float64)
        XYZ_As = np.empty((N, 3), dtype=np.float64)
        XYZ_Bs = np.empty((N, 3), dtype=np.float64)
        XYZ_ABs = np.empty((N, 3), dtype=np.float64)
        xy_As = np.empty((N, 2), dtype=np.float64)
        xy_Bs = np.empty((N, 2), dtype=np.float64)
        xy_ABs = np.empty((N, 2), dtype=np.float64)

        for i,date in enumerate(dates):
            vA, vB, XY_A, XY_B, XY_AB, xy_A, xy_B, xy_AB = self._get_orbit_t(date)
            vAs[i] = vA
            vBs[i] = vB
            XYZ_As[i] = np.array(XYZ_A)
            XYZ_Bs[i] = np.array(XYZ_B)
            XYZ_ABs[i] = np.array(XYZ_AB)
            xy_As[i] = np.array(xy_A)
            xy_Bs[i] = np.array(xy_B)
            xy_ABs[i] = np.array(xy_AB)


        return (vAs, vBs, XYZ_As, XYZ_Bs, XYZ_ABs, xy_As, xy_Bs, xy_ABs)


class Triple:
    '''
    Triple orbital model that can deliver absolute astrometric position, relative astrometric position (B relative to A, and C relative to A), and radial velocities for A, B, and C.

    Args:
        a_in (float): semi-major axis for inner orbit [AU]
        e_in (float): eccentricity for inner orbit (must be between ``[0.0, 1.0)``)
        i_in (float): inclination for inner orbit [deg]
        omega_in (float): argument of periastron for inner orbit [degrees]
        Omega_in (float): position angle of the ascending node [deg] east of north for inner orbit
        T0_in (float): epoch of periastron passage for inner orbit [JD]
        a_out (float): semi-major axis for outer orbit [AU]
        e_out (float): eccentricity for outer orbit (must be between ``[0.0, 1.0)``)
        i_out (float): inclination for outer orbit [deg]
        omega_out (float): argument of periastron for outer orbit [degrees]
        Omega_out (float): position angle of the ascending node [deg] east of north for outer orbit
        T0_out (float): epoch of periastron passage for outer orbit [JD]
        M_1 (float): mass of A [M_sun]
        M_2 (float): mass of B [M_sun]
        M_3 (float): mass of C [M_sun]
        gamma (float): systemic velocity (km/s)
        obs_dates (1D np.array): dates of observation (JD)
    '''
    def __init__(self, a_in, e_in, i_in, omega_in, Omega_in, T0_in, a_out, e_out, i_out, omega_out, Omega_out, T0_out, M_1, M_2, M_3, gamma, obs_dates=None, **kwargs):
        assert (e_in >= 0.0) and (e_in < 1.0), "Inner eccentricity must be between [0, 1)"
        assert (e_out >= 0.0) and (e_out < 1.0), "Outer eccentricity must be between [0, 1)"
        self.a_in = a_in # [AU]
        self.e_in = e_in #
        self.i_in = i_in # [deg]
        self.omega_in = omega_in # [deg]
        self.Omega_in = Omega_in # [deg]
        self.T0_in = T0_in # [JD]
        self.a_out = a_out # [AU]
        self.e_out = e_out
        self.i_out = i_out # [deg]
        self.omega_out = omega_out # [deg]
        self.Omega_out = Omega_out # [deg]
        self.T0_out = T0_out # [JD]
        self.M_1 = M_1 # [M_sun]
        self.M_2 = M_2 # [M_sun]
        self.M_3 = M_3 # [M_sun]
        self.gamma = gamma # [km/s]

        # Update the derived RV quantities
        self.P_in = np.sqrt(4 * np.pi**2 / (C.G * (self.M_1 + self.M_2) * C.M_sun) * (self.a_in * C.AU)**3) / (60 * 60 * 24)# [days]
        self.K_in = np.sqrt(C.G/(1 - self.e_in**2)) * self.M_2 * C.M_sun * np.sin(self.i_in * np.pi/180.) / np.sqrt((self.M_1 + self.M_2) * C.M_sun * self.a_in * C.AU) * 1e-5 # [km/s]

        self.P_out = np.sqrt(4 * np.pi**2 / (C.G * (self.M_1 + self.M_2 + self.M_3) * C.M_sun) * (self.a_out * C.AU)**3) / (60 * 60 * 24) # [days]

        self.K_out = np.sqrt(C.G/(1 - self.e_out**2)) * self.M_3 * C.M_sun * np.sin(self.i_out * np.pi/180.) / np.sqrt((self.M_1 + self.M_2 + self.M_3) * C.M_sun * self.a_out * C.AU) * 1e-5 # [km/s]

        # If we are going to be repeatedly predicting the orbit at a sequence of dates,
        # just store them to the object.
        self.obs_dates = obs_dates

    def _theta_in(self, t):
        '''Calculate the true anomoly for the A-B orbit.'''

        # t is input in seconds

        # Take a modulus of the period
        t = (t - self.T0_in) % self.P_in

        f = lambda E: E - self.e_in * np.sin(E) - 2 * np.pi * t/self.P_in
        E0 = 2 * np.pi * t / self.P_in

        E = fsolve(f, E0)[0]

        th = 2 * np.arctan(np.sqrt((1 + self.e_in)/(1 - self.e_in)) * np.tan(E/2.))

        if E < np.pi:
            return th
        else:
            return th + 2 * np.pi

    def _theta_out(self, t):
        '''Calculate the true anomoly for the (A-B) - C orbit.'''

        # t is input in seconds

        # Take a modulus of the period
        t = (t - self.T0_out) % self.P_out

        f = lambda E: E - self.e_out * np.sin(E) - 2 * np.pi * t/self.P_out
        E0 = 2 * np.pi * t / self.P_out

        E = fsolve(f, E0)[0]

        th = 2 * np.arctan(np.sqrt((1 + self.e_out)/(1 - self.e_out)) * np.tan(E/2.))

        if E < np.pi:
            return th
        else:
            return th + 2 * np.pi

    def _v1_f(self, f):
        '''Calculate the component of A's velocity based on only the inner orbit.
        f is the true anomoly of this inner orbit.'''

        return self.K_in * (np.cos(self.omega_in * np.pi/180 + f) + self.e_in * np.cos(self.omega_in * np.pi/180))

    def _v2_f(self, f):
        '''Calculate the component of B's velocity based on only the inner orbit.
        f is the true anomoly of this inner orbit.'''

        return -self.K_in * self.M_1/self.M_2 * (np.cos(self.omega_in * np.pi/180 + f) + self.e_in * np.cos(self.omega_in * np.pi/180))


    def _v3_f(self, f):
        '''Calculate the velocity of (A-B) based only on the outer orbit.
        f is the true anomoly of the outer orbit'''
        return  self.K_out * (np.cos(self.omega_out * np.pi/180 + f) + self.e_out * np.cos(self.omega_out * np.pi/180))


    def _v3_f_C(self, f):
        '''Calculate the velocity of C based only on the outer orbit.
        f is the true anomoly of the outer orbit
        '''
        return -self.K_out * (self.M_1 + self.M_2)/ self.M_3 * (np.cos(self.omega_out * np.pi/180 + f) + self.e_out * np.cos(self.omega_out * np.pi/180))

    # absolute position of the AB center of mass in the plane of the orbit
    def _xy_AB(self, f):
        # find the reduced radius
        r = self.a_out * (1 - self.e_out**2) / (1 + self.e_out * np.cos(f)) # [AU]
        r1 = r * self.M_3 / (self.M_1 + self.M_2 + self.M_3) # [AU]

        x = r1 * np.cos(f)
        y = r1 * np.sin(f)

        return (x,y)

    # absolute position of C in the plane of the orbit
    def _xy_C(self, f):
        # find the reduced radius
        r = self.a_out * (1 - self.e_out**2) / (1 + self.e_out * np.cos(f)) # [AU]
        r2 = -r * (self.M_1 + self.M_2) / (self.M_1 + self.M_2 + self.M_3) # [AU]

        x = r2 * np.cos(f)
        y = r2 * np.sin(f)

        return (x,y)

    # absolute position of AB center of mass
    def _XYZ_AB(self, f):

        # find the reduced radius
        r = self.a_out * (1 - self.e_out**2) / (1 + self.e_out * np.cos(f)) # [AU]
        r1 = r * self.M_3 / (self.M_1 + self.M_2 + self.M_3) # [AU]

        Omega = self.Omega_out * np.pi / 180
        omega = self.omega_out * np.pi / 180 # add in pi to swap the periapse
        i = self.i_out * np.pi / 180
        X = r1 * (np.cos(Omega) * np.cos(omega + f) - np.sin(Omega) * np.sin(omega + f) * np.cos(i))
        Y = r1 * (np.sin(Omega) * np.cos(omega + f) + np.cos(Omega) * np.sin(omega + f) * np.cos(i))
        Z = r1 * (np.sin(omega + f) * np.sin(i))

        return (X, Y, Z) # [AU]

    # absolute position of C
    def _XYZ_C(self, f):

        # find the reduced radius
        r = self.a_out * (1 - self.e_out**2) / (1 + self.e_out * np.cos(f)) # [AU]
        r2 = -r * (self.M_1 + self.M_2) / (self.M_1 + self.M_2 + self.M_3) # [AU]

        Omega = self.Omega_out * np.pi / 180
        omega = self.omega_out * np.pi / 180
        i = self.i_out * np.pi / 180
        X = r2 * (np.cos(Omega) * np.cos(omega + f) - np.sin(Omega) * np.sin(omega + f) * np.cos(i))
        Y = r2 * (np.sin(Omega) * np.cos(omega + f) + np.cos(Omega) * np.sin(omega + f) * np.cos(i))
        Z = r2 * (np.sin(omega + f) * np.sin(i))

        return (X, Y, Z) # [AU]


    # position of A relative to center of mass of AB in the plane of the orbit
    def _xy_A_loc(self, f):
        # find the reduced radius
        r = self.a_in * (1 - self.e_in**2) / (1 + self.e_in * np.cos(f)) # [AU]
        r1 = r * self.M_2 / (self.M_1 + self.M_2) # [AU]

        x = r1 * np.cos(f)
        y = r1 * np.sin(f)

        return (x,y)

    # position of B relative to center of mass of AB in the plane of the orbit
    def _xy_B_loc(self, f):
        # find the reduced radius
        r = self.a_in * (1 - self.e_in**2) / (1 + self.e_in * np.cos(f)) # [AU]
        r2 = -r * self.M_1 / (self.M_1 + self.M_2) # [AU]

        x = r2 * np.cos(f)
        y = r2 * np.sin(f)

        return (x,y)

    # position of A relative to center of mass of AB (projected)
    def _XYZ_A_loc(self, f):

        # find the reduced radius
        r = self.a_in * (1 - self.e_in**2) / (1 + self.e_in * np.cos(f)) # [AU]
        r1 = r * self.M_2 / (self.M_1 + self.M_2) # [AU]

        Omega = self.Omega_in * np.pi / 180
        omega = self.omega_in * np.pi / 180 # add in pi to swap the periapse
        i = self.i_in * np.pi / 180
        X = r1 * (np.cos(Omega) * np.cos(omega + f) - np.sin(Omega) * np.sin(omega + f) * np.cos(i))
        Y = r1 * (np.sin(Omega) * np.cos(omega + f) + np.cos(Omega) * np.sin(omega + f) * np.cos(i))
        Z = r1 * (np.sin(omega + f) * np.sin(i))

        return (X, Y, Z) # [AU]

    # position of B relative to center of mass of AB (projected)
    def _XYZ_B_loc(self, f):

        # find the reduced radius
        r = self.a_in * (1 - self.e_in**2) / (1 + self.e_in * np.cos(f)) # [AU]
        r2 = -r * self.M_1 / (self.M_1 + self.M_2) # [AU]

        Omega = self.Omega_in * np.pi / 180
        omega = self.omega_in * np.pi / 180
        i = self.i_in * np.pi / 180
        X = r2 * (np.cos(Omega) * np.cos(omega + f) - np.sin(Omega) * np.sin(omega + f) * np.cos(i))
        Y = r2 * (np.sin(Omega) * np.cos(omega + f) + np.cos(Omega) * np.sin(omega + f) * np.cos(i))
        Z = r2 * (np.sin(omega + f) * np.sin(i))

        return (X, Y, Z) # [AU]


    def _get_orbit_t(self, t):
        '''
        Given a time, calculate all of the orbital quantaties we might be interseted in.
        returns (v_A, v_B, (x,y) of A, (x,y) of B, and x,y of B relative to A)
        '''

        # Get the true anomoly "f" from time
        f_in = self._theta_in(t)
        f_out = self._theta_out(t)

        # Feed this into the orbit equation and add the systemic velocity
        vA = self._v1_f(f_in) + self._v3_f(f_out) + self.gamma
        vB = self._v2_f(f_in) + self._v3_f(f_out) + self.gamma
        vC = self._v3_f_C(f_out) + self.gamma

        # Absolute positions of AB center of mass, and C component.
        XYZ_AB = self._XYZ_AB(f_out)
        XYZ_C = self._XYZ_C(f_out)

        # Positions of A and B relative to AB center of mass.
        XYZ_A_loc = self._XYZ_A_loc(f_in)
        XYZ_B_loc = self._XYZ_B_loc(f_in)

        # Absolute positions of A and B
        XYZ_A = np.array(XYZ_A_loc) + np.array(XYZ_AB)
        XYZ_B = np.array(XYZ_B_loc) + np.array(XYZ_AB)

        # Orbital positions in the plane
        xy_AB = self._xy_AB(f_out)
        xy_C = self._xy_C(f_out)

        xy_A_loc = self._xy_A_loc(f_in)
        xy_B_loc = self._xy_B_loc(f_in)

        return (vA, vB, vC, XYZ_A, XYZ_B, XYZ_C, XYZ_AB, XYZ_A_loc, XYZ_B_loc, xy_C, xy_AB, xy_A_loc, xy_B_loc)



    def get_orbit(self, dates=None):
        '''
        Deliver only the main quantities useful for performing a joint astrometric + RV fit to real data, namely
        the radial velocities ``vA``, ``vB``, ``vC``, the relative offsets of B to A ``rho_AB``, and relative position angles ``theta_AB``, and the relative offsets of C to A ``rho_AC`` and ``theta_AC`` for all dates provided. Relative offsets are provided in AU, and so must be converted to arcseconds after assuming a distance to the system. Relative position angles are given in degrees east of north.

        Args:
            dates (optional): if provided, calculate quantities at this new vector of dates, rather than the one provided when the object was initialized.

        Returns:
            np.array: A ``(7, npoints)`` shape array of ``[vAs, vBs, vCs, rho_ABs, theta_ABs, rho_ACs, theta_ACs]``
        '''

        if dates is None and self.obs_dates is None:
            raise RuntimeError("Must provide input dates or specify observation dates upon creation of orbit object.")

        if dates is None and self.obs_dates is not None:
            dates = self.obs_dates

        dates = np.atleast_1d(dates)
        N = len(dates)

        vAs = np.empty(N, dtype=np.float64)
        vBs = np.empty(N, dtype=np.float64)
        vCs = np.empty(N, dtype=np.float64)

        rho_ABs = np.empty(N, dtype=np.float64)
        theta_ABs = np.empty(N, dtype=np.float64)

        rho_ACs = np.empty(N, dtype=np.float64)
        theta_ACs = np.empty(N, dtype=np.float64)


        for i,date in enumerate(dates):
            vA, vB, vC, XYZ_A, XYZ_B, XYZ_C, XYZ_AB, XYZ_A_loc, XYZ_B_loc, xy_C, xy_AB, xy_A_loc, xy_B_loc = self._get_orbit_t(date)

            vAs[i] = vA
            vBs[i] = vB
            vCs[i] = vC

            # For AB pair

            # Calculate rho, theta from XY_A, XY_B, and XY_C
            X_A, Y_A, Z_A = XYZ_A
            X_B, Y_B, Z_B = XYZ_B
            X_C, Y_C, Z_C = XYZ_C

            rho_AB = np.sqrt((X_B - X_A)**2 + (Y_B - Y_A)**2) # [AU]
            theta_AB = np.arctan2((Y_B - Y_A), (X_B - X_A)) * 180/np.pi # [Deg]
            if theta_AB < 0: # ensure that 0 <= theta <= 360
                theta_AB += 360.

            rho_ABs[i] = rho_AB
            theta_ABs[i] = theta_AB

            rho_AC = np.sqrt((X_C - X_A)**2 + (Y_C - Y_A)**2) # [AU]
            theta_AC = np.arctan2((Y_C - Y_A), (X_C - X_A)) * 180/np.pi # [Deg]
            if theta_AC < 0:
                theta_AC += 360.

            rho_ACs[i] = rho_AC
            theta_ACs[i] = theta_AC

        return np.vstack((vAs, vBs, vCs, rho_ABs, theta_ABs, rho_ACs, theta_ACs))

    def get_full_orbit(self, dates=None):
        '''
        Deliver the full set of astrometric and radial velocity quantities, namely
        the radial velocities ``vA``, ``vB``, ``vC``, the position of A, B, and C relative to the center of mass in the plane of the sky (``XY_A``, ``XY_B``, and ``XY_C``, respectively), the absolute position of the center of mass of (AB), (``XY_AB``), the position of A relative to the center of mass of AB (``XY_A_loc``), the position of B relative to the center of mass of (AB) (``XY_B_loc``), the absolute position of C in the plane of the orbit (``xy_C``), the absolute positon of the center of mass of AB in the plane of the orbit (``xy_AB``), the position of A in the plane of the orbit, relative to the center of mass of AB (``xy_A``), and the position of B in the plane of the orbit, relative to the center of mass of AB (``xy_B``), for all dates provided. All positions are given in units of AU, and so must be converted to arcseconds after assuming a distance to the system.

        Args:
            dates (optional): if provided, calculate quantities at this new vector of dates, rather than the one provided when the object was initialized.

        Returns:
            np.array: A ``(13, npoints)`` shape array of ``[vAs, vBs, vCs, XY_As, XY_Bs, XY_Cs, XY_ABs, XY_A_locs, XY_B_locs, xy_Cs, xy_ABs, xy_A_locs, xy_B_locs]``
        '''

        if dates is None and self.obs_dates is None:
            raise RuntimeError("Must provide input dates or specify observation dates upon creation of orbit object.")

        if dates is None and self.obs_dates is not None:
            dates = self.obs_dates

        dates = np.atleast_1d(dates)
        N = len(dates)

        vAs = np.empty(N, dtype=np.float64)
        vBs = np.empty(N, dtype=np.float64)
        vCs = np.empty(N, dtype=np.float64)

        XYZ_As = np.empty((N, 3), dtype=np.float64)
        XYZ_Bs = np.empty((N, 3), dtype=np.float64)
        XYZ_Cs = np.empty((N, 3), dtype=np.float64)
        XYZ_ABs = np.empty((N, 3), dtype=np.float64)
        XYZ_A_locs = np.empty((N, 3), dtype=np.float64)
        XYZ_B_locs = np.empty((N, 3), dtype=np.float64)

        xy_Cs = np.empty((N, 2), dtype=np.float64)
        xy_ABs = np.empty((N, 2), dtype=np.float64)
        xy_A_locs = np.empty((N, 2), dtype=np.float64)
        xy_B_locs = np.empty((N, 2), dtype=np.float64)

        for i,date in enumerate(dates):
            vA, vB, vC, XYZ_A, XYZ_B, XYZ_C, XYZ_AB, XYZ_A_loc, XYZ_B_loc, xy_C, xy_AB, xy_A_loc, xy_B_loc = self._get_orbit_t(date)
            vAs[i] = vA
            vBs[i] = vB
            vCs[i] = vC

            XYZ_As[i] = np.array(XYZ_A)
            XYZ_Bs[i] = np.array(XYZ_B)
            XYZ_Cs[i] = np.array(XYZ_C)

            XYZ_ABs[i] = np.array(XYZ_AB)
            XYZ_A_locs[i] = np.array(XYZ_A_loc)
            XYZ_B_locs[i] = np.array(XYZ_B_loc)

            xy_Cs[i] = np.array(xy_C)
            xy_ABs[i] = np.array(xy_AB)
            xy_A_locs[i] = np.array(xy_A_loc)
            xy_B_locs[i] = np.array(xy_B_loc)

        return (vAs, vBs, vCs, XYZ_As, XYZ_Bs, XYZ_Cs, XYZ_ABs, XYZ_A_locs, XYZ_B_locs, xy_Cs, xy_ABs, xy_A_locs, xy_B_locs)


models = {"Binary":Binary, "Triple":Triple}
