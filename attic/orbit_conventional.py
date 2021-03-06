import numpy as np
from scipy.optimize import fsolve, minimize

from psoap import constants as C

class Binary:
    '''
    A binary orbit that delivers astrometric position, relative astrometric position (B relative to A), and radial velocities of A and B.
    '''
    def __init__(self, a, e, i, omega, Omega, T0, M_tot, M_2, gamma, obs_dates=None, **kwargs):
        self.a = a # [AU] semi-major axis
        self.e = e # eccentricity
        self.i = i # [deg] inclination
        self.omega = omega # [deg] argument of periastron
        self.Omega = Omega # [deg] east of north
        self.T0 = T0 # [JD]
        self.M_tot = M_tot # [M_sun]
        self.M_2 = M_2 # [M_sun]
        self.gamma = gamma # [km/s]

        # Update the RV quantities
        self.recalculate()

        # If we are going to be repeatedly predicting the orbit at a sequence of dates,
        # just store them to the object.
        self.obs_dates = obs_dates

        self.param_dict = {"a":self.a, "e":self.e, "i":self.i, "omega":self.omega,
        "Omega":self.Omega, "T0":self.T0, "M_tot":self.M_tot, "M_2":self.M_2, "gamma":self.gamma}

    def recalculate(self):
        '''
        Recalculates derivative RV quantities when other parameters are updated.
        '''
        # Calculate the following RV quantities
        self.q = self.M_2 / (self.M_tot - self.M_2) # [M2/M1]
        self.P = np.sqrt(4 * np.pi**2 / (C.G * self.M_tot * C.M_sun) * (self.a * C.AU)**3) / (60 * 60 * 24)# [days]
        self.K = np.sqrt(C.G/(1 - self.e**2)) * self.M_2 * C.M_sun * np.sin(self.i * np.pi/180.) / np.sqrt(self.M_tot * C.M_sun * self.a * C.AU) * 1e-5 # [km/s]

    def update_parameters(self, param_values, param_list):
        '''
        param_values is numpy array of values
        param_list is list of strings of the names of the parameters
        '''
        for (value, key) in zip(param_values, param_list):
            self.param_dict[key] = value

    def theta(self, t):
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

    def v1_f(self, f):
        '''Calculate the component of A's velocity based on only the inner orbit.
        f is the true anomoly of this inner orbit.'''

        return self.K * (np.cos(self.omega * np.pi/180 + f) + self.e * np.cos(self.omega * np.pi/180))

    def v2_f(self, f):
        '''Calculate the component of B's velocity based on only the inner orbit.
        f is the true anomoly of this inner orbit.'''

        return -self.K/self.q * (np.cos(self.omega * np.pi/180 + f) + self.e * np.cos(self.omega * np.pi/180))

    # Get the position of A in the plane of the orbit
    def xy_A(self, f):
        # find the reduced radius
        r = self.a * (1 - self.e**2) / (1 + self.e * np.cos(f)) # [AU]
        r1 = r * self.M_2 / self.M_tot # [AU]

        x = r1 * np.cos(f)
        y = r1 * np.sin(f)

        return (x,y)

    # Get the position of B in the plane of the orbit
    def xy_B(self, f):
        # find the reduced radius
        r = self.a * (1 - self.e**2) / (1 + self.e * np.cos(f)) # [AU]
        r2 = -r * (self.M_tot - self.M_2) / self.M_tot # [AU]

        x = r2 * np.cos(f)
        y = r2 * np.sin(f)

        return (x,y)

    def xy_AB(self, f):
        r = self.a * (1 - self.e**2) / (1 + self.e * np.cos(f)) # [AU]
        x = r * np.cos(f)
        y = r * np.sin(f)

        return (x,y)

    # position of A relative to center of mass
    def XY_A(self, f):

        Omega = self.Omega * np.pi / 180
        omega = self.omega * np.pi / 180 # add in pi to swap the periapse
        i = self.i * np.pi / 180

        # find the reduced semi-major axis
        a1 = self.a  * self.M_2 / self.M_tot
        r1 = a1 * (1 - self.e**2) / (1 + self.e * np.cos(f)) # [AU]

        x = r1 / a1 * np.cos(f)
        y = r1 / a1 * np.sin(f)

        # Calculate Thiele-Innes elements
        A = a1 * (np.cos(omega) * np.cos(Omega) - np.sin(omega) * np.sin(Omega) * np.cos(i))
        B = a1 * (np.cos(omega) * np.sin(Omega) + np.sin(omega) * np.cos(Omega) * np.cos(i))
        F = a1 * (-np.sin(omega) * np.cos(Omega) - np.cos(omega) * np.sin(Omega) * np.cos(i))
        G = a1 * (-np.sin(omega) * np.sin(Omega) + np.cos(omega) * np.cos(Omega) * np.cos(i))

        X = A * x + F * y
        Y = B * x + G * y

        return (X, Y) # [AU]

    # position of B relative to center of mass
    def XY_B(self, f):

        Omega = self.Omega * np.pi / 180
        omega = self.omega * np.pi / 180
        i = self.i * np.pi / 180

        # find the reduced radius
        a2 = self.a * (self.M_tot - self.M_2) / self.M_tot
        r2 = a2 * (1 - self.e**2) / (1 + self.e * np.cos(f)) # [AU]

        x = r2 / a2 * np.cos(f)
        y = r2 / a2 * np.sin(f)

        # Calculate Thiele-Innes elements
        A = a2 * (np.cos(omega) * np.cos(Omega) - np.sin(omega) * np.sin(Omega) * np.cos(i))
        B = a2 * (np.cos(omega) * np.sin(Omega) + np.sin(omega) * np.cos(Omega) * np.cos(i))
        F = a2 * (-np.sin(omega) * np.cos(Omega) - np.cos(omega) * np.sin(Omega) * np.cos(i))
        G = a2 * (-np.sin(omega) * np.sin(Omega) + np.cos(omega) * np.cos(Omega) * np.cos(i))

        X = A * x + F * y
        Y = B * x + G * y

        return (X, Y) # [AU]

    def XY_AB(self, f):

        Omega = self.Omega * np.pi / 180
        omega = self.omega * np.pi / 180
        i = self.i * np.pi / 180

        r = self.a * (1 - self.e**2) / (1 + self.e * np.cos(f))
        x = r / self.a * np.cos(f)
        y = r / self.a * np.sin(f)

        # Calculate Thiele-Innes elements
        A = self.a * (np.cos(omega) * np.cos(Omega) - np.sin(omega) * np.sin(Omega) * np.cos(i))
        B = self.a * (np.cos(omega) * np.sin(Omega) + np.sin(omega) * np.cos(Omega) * np.cos(i))
        F = self.a * (-np.sin(omega) * np.cos(Omega) - np.cos(omega) * np.sin(Omega) * np.cos(i))
        G = self.a * (-np.sin(omega) * np.sin(Omega) + np.cos(omega) * np.cos(Omega) * np.cos(i))

        X = A * x + F * y
        Y = B * x + G * y

        # X is north, Y is east.
        return (X, Y) # [AU]


    def get_orbit(self, t):
        '''
        Given a time, calculate all of the orbital quantaties we might be interseted in.
        returns (v_A, v_B, (x,y) of A, (x,y) of B, and x,y of B relative to A)
        '''

        # Get the true anomoly "f" from time
        f = self.theta(t)

        # Feed this into the orbit equation and add the systemic velocity
        vA = self.v1_f(f) + self.gamma
        vB = self.v2_f(f) + self.gamma

        XY_A = self.XY_A(f)
        XY_B = self.XY_B(f)
        XY_AB = self.XY_AB(f)
        xy_A = self.xy_A(f)
        xy_B = self.xy_B(f)
        xy_AB = self.xy_AB(f)

        return (vA, vB, XY_A, XY_B, XY_AB, xy_A, xy_B, xy_AB)

    def get_component_orbits(self, dates=None):
        '''
        Return both vA and vB for all dates provided.
        '''

        if dates is None and self.obs_dates is None:
            raise RuntimeError("Must provide input dates or specify observation dates upon creation of orbit object.")

        if dates is None and self.obs_dates is not None:
            dates = self.obs_dates

        dates = np.atleast_1d(dates)
        N = len(dates)

        vAs = np.empty(N, dtype=np.float64)
        vBs = np.empty(N, dtype=np.float64)
        XY_As = np.empty((N, 2), dtype=np.float64)
        XY_Bs = np.empty((N, 2), dtype=np.float64)
        XY_ABs = np.empty((N, 2), dtype=np.float64)
        xy_As = np.empty((N, 2), dtype=np.float64)
        xy_Bs = np.empty((N, 2), dtype=np.float64)
        xy_ABs = np.empty((N, 2), dtype=np.float64)

        for i,date in enumerate(dates):
            vA, vB, XY_A, XY_B, XY_AB, xy_A, xy_B, xy_AB = self.get_orbit(date)
            vAs[i] = vA
            vBs[i] = vB
            XY_As[i] = np.array(XY_A)
            XY_Bs[i] = np.array(XY_B)
            XY_ABs[i] = np.array(XY_AB)
            xy_As[i] = np.array(xy_A)
            xy_Bs[i] = np.array(xy_B)
            xy_ABs[i] = np.array(xy_AB)


        return (vAs, vBs, XY_As, XY_Bs, XY_ABs, xy_As, xy_Bs, xy_ABs)

    def get_component_fits(self, dates=None):
        '''
        Return both vA, vB, rho_AB, and theta_AB, for all dates provided.
        These are mainly as inputs to a fit.
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
            vA, vB, XY_A, XY_B, XY_AB, xy_A, xy_B, xy_AB = self.get_orbit(date)
            vAs[i] = vA
            vBs[i] = vB

            # Calculate rho, theta from XY_AB
            X, Y = XY_AB

            rho = np.sqrt(X**2 + Y**2) # [AU]
            theta = np.arctan2(Y, X) * 180/np.pi # [Deg]
            if theta < 0: # ensure that 0 <= theta <= 360
                theta += 360.

            rho_ABs[i] = rho
            theta_ABs[i] = theta

        return (vAs, vBs, rho_ABs, theta_ABs)

class Triple:
    '''
    Techniques describing solving for a triple star orbit.
    '''
    def __init__(self, a_in, e_in, i_in, omega_in, Omega_in, T0_in, a_out, e_out, i_out, omega_out, Omega_out, T0_out, M_1, M_2, M_3, gamma, obs_dates=None, **kwargs):
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

        self.recalculate()

        # If we are going to be repeatedly predicting the orbit at a sequence of dates,
        # just store them to the object.
        self.obs_dates = obs_dates

        self.param_dict = {"a_in":self.a_in, "e_in":self.e_in, "i_in":self.i_in, "omega_in":self.omega_in, "Omega_in":self.Omega_in, "T0_in":self.T0_in, "a_out":self.a_out, "e_out":self.e_out, "i_out":self.i_out, "omega_out":self.omega_out, "Omega_out":self.Omega_out, "T0_out":self.T0_out, "M_1":self.M_1, "M_2":self.M_2, "M_3":self.M_3, "gamma":self.gamma}

    def recalculate(self):
        '''
        Update all of the derived quantities.
        '''
        # Calculate the following RV quantities
        self.P_in = np.sqrt(4 * np.pi**2 / (C.G * (self.M_1 + self.M_2) * C.M_sun) * (self.a_in * C.AU)**3) / (60 * 60 * 24)# [days]
        self.K_in = np.sqrt(C.G/(1 - self.e_in**2)) * self.M_2 * C.M_sun * np.sin(self.i_in * np.pi/180.) / np.sqrt((self.M_1 + self.M_2) * C.M_sun * self.a_in * C.AU) * 1e-5 # [km/s]

        self.P_out = np.sqrt(4 * np.pi**2 / (C.G * (self.M_1 + self.M_2 + self.M_3) * C.M_sun) * (self.a_out * C.AU)**3) / (60 * 60 * 24) # [days]

        self.K_out = np.sqrt(C.G/(1 - self.e_out**2)) * self.M_3 * C.M_sun * np.sin(self.i_out * np.pi/180.) / np.sqrt((self.M_1 + self.M_2 + self.M_3) * C.M_sun * self.a_out * C.AU) * 1e-5 # [km/s]

    def update_parameters(self, param_values, param_list):
        '''
        param_values is numpy array of values
        param_list is list of strings of the names of the parameters
        '''
        for (value, key) in zip(param_values, param_list):
            self.param_dict[key] = value

    def theta_in(self, t):
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

    def theta_out(self, t):
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

    def v1_f(self, f):
        '''Calculate the component of A's velocity based on only the inner orbit.
        f is the true anomoly of this inner orbit.'''

        return self.K_in * (np.cos(self.omega_in * np.pi/180 + f) + self.e_in * np.cos(self.omega_in * np.pi/180))

    def v2_f(self, f):
        '''Calculate the component of B's velocity based on only the inner orbit.
        f is the true anomoly of this inner orbit.'''

        return -self.K_in * self.M_1/self.M_2 * (np.cos(self.omega_in * np.pi/180 + f) + self.e_in * np.cos(self.omega_in * np.pi/180))


    def v3_f(self, f):
        '''Calculate the velocity of (A-B) based only on the outer orbit.
        f is the true anomoly of the outer orbit'''
        return  self.K_out * (np.cos(self.omega_out * np.pi/180 + f) + self.e_out * np.cos(self.omega_out * np.pi/180))


    def v3_f_C(self, f):
        '''Calculate the velocity of C based only on the outer orbit.
        f is the true anomoly of the outer orbit
        '''
        return -self.K_out * (self.M_1 + self.M_2)/ self.M_3 * (np.cos(self.omega_out * np.pi/180 + f) + self.e_out * np.cos(self.omega_out * np.pi/180))

    # absolute position of the AB center of mass in the plane of the orbit
    def xy_AB(self, f):
        # find the reduced radius
        r = self.a_out * (1 - self.e_out**2) / (1 + self.e_out * np.cos(f)) # [AU]
        r1 = r * self.M_3 / (self.M_1 + self.M_2 + self.M_3) # [AU]

        x = r1 * np.cos(f)
        y = r1 * np.sin(f)

        return (x,y)

    # absolute position of C in the plane of the orbit
    def xy_C(self, f):
        # find the reduced radius
        r = self.a_out * (1 - self.e_out**2) / (1 + self.e_out * np.cos(f)) # [AU]
        r2 = -r * (self.M_1 + self.M_2) / (self.M_1 + self.M_2 + self.M_3) # [AU]

        x = r2 * np.cos(f)
        y = r2 * np.sin(f)

        return (x,y)

    # absolute position of AB center of mass
    def XY_AB(self, f):

        # find the reduced radius
        r = self.a_out * (1 - self.e_out**2) / (1 + self.e_out * np.cos(f)) # [AU]
        r1 = r * self.M_3 / (self.M_1 + self.M_2 + self.M_3) # [AU]

        Omega = self.Omega_out * np.pi / 180
        omega = self.omega_out * np.pi / 180 # add in pi to swap the periapse
        i = self.i_out * np.pi / 180
        X = r1 * (np.cos(Omega) * np.cos(omega + f) - np.sin(Omega) * np.sin(omega + f) * np.cos(i))
        Y = r1 * (np.sin(Omega) * np.cos(omega + f) + np.cos(Omega) * np.sin(omega + f) * np.cos(i))

        return (X, Y) # [AU]

    # absolute position of C
    def XY_C(self, f):

        # find the reduced radius
        r = self.a_out * (1 - self.e_out**2) / (1 + self.e_out * np.cos(f)) # [AU]
        r2 = -r * (self.M_1 + self.M_2) / (self.M_1 + self.M_2 + self.M_3) # [AU]

        Omega = self.Omega_out * np.pi / 180
        omega = self.omega_out * np.pi / 180
        i = self.i_out * np.pi / 180
        X = r2 * (np.cos(Omega) * np.cos(omega + f) - np.sin(Omega) * np.sin(omega + f) * np.cos(i))
        Y = r2 * (np.sin(Omega) * np.cos(omega + f) + np.cos(Omega) * np.sin(omega + f) * np.cos(i))

        return (X, Y) # [AU]


    # position of A relative to center of mass of AB in the plane of the orbit
    def xy_A_loc(self, f):
        # find the reduced radius
        r = self.a_in * (1 - self.e_in**2) / (1 + self.e_in * np.cos(f)) # [AU]
        r1 = r * self.M_2 / (self.M_1 + self.M_2) # [AU]

        x = r1 * np.cos(f)
        y = r1 * np.sin(f)

        return (x,y)

    # position of B relative to center of mass of AB in the plane of the orbit
    def xy_B_loc(self, f):
        # find the reduced radius
        r = self.a_in * (1 - self.e_in**2) / (1 + self.e_in * np.cos(f)) # [AU]
        r2 = -r * self.M_1 / (self.M_1 + self.M_2) # [AU]

        x = r2 * np.cos(f)
        y = r2 * np.sin(f)

        return (x,y)

    # position of A relative to center of mass of AB (projected)
    def XY_A_loc(self, f):

        # find the reduced radius
        r = self.a_in * (1 - self.e_in**2) / (1 + self.e_in * np.cos(f)) # [AU]
        r1 = r * self.M_2 / (self.M_1 + self.M_2) # [AU]

        Omega = self.Omega_in * np.pi / 180
        omega = self.omega_in * np.pi / 180 # add in pi to swap the periapse
        i = self.i_in * np.pi / 180
        X = r1 * (np.cos(Omega) * np.cos(omega + f) - np.sin(Omega) * np.sin(omega + f) * np.cos(i))
        Y = r1 * (np.sin(Omega) * np.cos(omega + f) + np.cos(Omega) * np.sin(omega + f) * np.cos(i))

        return (X, Y) # [AU]

    # position of B relative to center of mass of AB (projected)
    def XY_B_loc(self, f):

        # find the reduced radius
        r = self.a_in * (1 - self.e_in**2) / (1 + self.e_in * np.cos(f)) # [AU]
        r2 = -r * self.M_1 / (self.M_1 + self.M_2) # [AU]

        Omega = self.Omega_in * np.pi / 180
        omega = self.omega_in * np.pi / 180
        i = self.i_in * np.pi / 180
        X = r2 * (np.cos(Omega) * np.cos(omega + f) - np.sin(Omega) * np.sin(omega + f) * np.cos(i))
        Y = r2 * (np.sin(Omega) * np.cos(omega + f) + np.cos(Omega) * np.sin(omega + f) * np.cos(i))

        return (X, Y) # [AU]


    def get_orbit(self, t):
        '''
        Given a time, calculate all of the orbital quantaties we might be interseted in.
        returns (v_A, v_B, (x,y) of A, (x,y) of B, and x,y of B relative to A)
        '''

        # Get the true anomoly "f" from time
        f_in = self.theta_in(t)
        f_out = self.theta_out(t)

        # Feed this into the orbit equation and add the systemic velocity
        vA = self.v1_f(f_in) + self.v3_f(f_out) + self.gamma
        vB = self.v2_f(f_in) + self.v3_f(f_out) + self.gamma
        vC = self.v3_f_C(f_out) + self.gamma

        # Absolute positions of AB center of mass, and C component.
        XY_AB = self.XY_AB(f_out)
        XY_C = self.XY_C(f_out)

        # Positions of A and B relative to AB center of mass.
        XY_A_loc = self.XY_A_loc(f_in)
        XY_B_loc = self.XY_B_loc(f_in)

        # Absolute positions of A and B
        XY_A = np.array(XY_A_loc) + np.array(XY_AB)
        XY_B = np.array(XY_B_loc) + np.array(XY_AB)

        # Orbital positions in the plane
        xy_AB = self.xy_AB(f_out)
        xy_C = self.xy_C(f_out)

        xy_A_loc = self.xy_A_loc(f_in)
        xy_B_loc = self.xy_B_loc(f_in)

        return (vA, vB, vC, XY_A, XY_B, XY_C, XY_AB, XY_A_loc, XY_B_loc, xy_C, xy_AB, xy_A_loc, xy_B_loc)


    def get_component_orbits(self, dates=None):
        '''
        Return both vA and vB for all dates provided.
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

        XY_As = np.empty((N, 2), dtype=np.float64)
        XY_Bs = np.empty((N, 2), dtype=np.float64)
        XY_Cs = np.empty((N, 2), dtype=np.float64)
        XY_ABs = np.empty((N, 2), dtype=np.float64)
        XY_A_locs = np.empty((N, 2), dtype=np.float64)
        XY_B_locs = np.empty((N, 2), dtype=np.float64)

        xy_Cs = np.empty((N, 2), dtype=np.float64)
        xy_ABs = np.empty((N, 2), dtype=np.float64)
        xy_A_locs = np.empty((N, 2), dtype=np.float64)
        xy_B_locs = np.empty((N, 2), dtype=np.float64)

        for i,date in enumerate(dates):
            vA, vB, vC, XY_A, XY_B, XY_C, XY_AB, XY_A_loc, XY_B_loc, xy_C, xy_AB, xy_A_loc, xy_B_loc = self.get_orbit(date)
            vAs[i] = vA
            vBs[i] = vB
            vCs[i] = vC

            XY_As[i] = np.array(XY_A)
            XY_Bs[i] = np.array(XY_B)
            XY_Cs[i] = np.array(XY_C)

            XY_ABs[i] = np.array(XY_AB)
            XY_A_locs[i] = np.array(XY_A_loc)
            XY_B_locs[i] = np.array(XY_B_loc)

            xy_Cs[i] = np.array(xy_C)
            xy_ABs[i] = np.array(xy_AB)
            xy_A_locs[i] = np.array(xy_A_loc)
            xy_B_locs[i] = np.array(xy_B_loc)

        return (vAs, vBs, vCs, XY_As, XY_Bs, XY_Cs, XY_ABs, XY_A_locs, XY_B_locs, xy_Cs, xy_ABs, xy_A_locs, xy_B_locs)

    def get_component_fits(self, dates=None):
        '''
        Return the vA, vB, vC, rho_AB, theta_AB, rho_AC, theta_AC for all dates provided.
        These are mainly as inputs to a fit.
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
            vA, vB, vC, XY_A, XY_B, XY_C, XY_AB, XY_A_loc, XY_B_loc, xy_C, xy_AB, xy_A_loc, xy_B_loc = self.get_orbit(date)

            vAs[i] = vA
            vBs[i] = vB
            vCs[i] = vC

            # For AB pair

            # Calculate rho, theta from XY_A, XY_B, and XY_C
            X_A, Y_A = XY_A
            X_B, Y_B = XY_B
            X_C, Y_C = XY_C

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

        return (vAs, vBs, vCs, rho_ABs, theta_ABs, rho_ACs, theta_ACs)


models = {"Binary":Binary, "Triple":Triple}
