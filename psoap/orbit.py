import numpy as np
from scipy.optimize import fsolve, minimize

from psoap import constants as C

class SB1:
    '''
    Describing a single-line Spectroscopic binary.
    '''
    def __init__(self, K, e, omega, P, T0, gamma, obs_dates=None, **kwargs):
        self._K = K # [km/s]
        self._e = e
        self._omega = omega # [deg]
        self._P = P # [days]
        self._T0 = T0 # [JD]
        self._gamma = gamma # [km/s]

        # If we are going to be repeatedly predicting the orbit at a sequence of dates,
        # just store them to the object.
        self.obs_dates = obs_dates

    # Properties so that we can easily update subsets of the orbit.
    @property
    def K(self):
        return self._K

    @K.setter
    def K(self, value):
        self._K = value

    @property
    def e(self):
        return self._e

    @e.setter
    def e(self, value):
        self._e = value

    @property
    def omega(self):
        return self._omega

    @omega.setter
    def omega(self, value):
        self._omega = value

    @property
    def P(self):
        return self._P

    @P.setter
    def P(self, value):
        self._P = value

    @property
    def T0(self):
        return self._T0

    @T0.setter
    def T0(self, value):
        self._T0 = value

    @property
    def gamma(self):
        return self._gamma

    @gamma.setter
    def gamma(self, value):
        self._gamma = value

    def theta(self, t):
        '''Calculate the true anomoly for the A-B orbit.
        Input is in days.'''

        # t is input in seconds

        # Take a modulus of the period
        t = (t - self._T0) % self._P

        f = lambda E: E - self._e * np.sin(E) - 2 * np.pi * t/self._P
        E0 = 2 * np.pi * t / self._P

        E = fsolve(f, E0)[0]

        th = 2 * np.arctan(np.sqrt((1 + self._e)/(1 - self._e)) * np.tan(E/2.))

        if E < np.pi:
            return th
        else:
            return th + 2 * np.pi

    def v1_f(self, f):
        '''Calculate the component of A's velocity based on only the inner orbit.
        f is the true anomoly of this inner orbit.'''

        return self._K * (np.cos(self._omega * np.pi/180 + f) + self._e * np.cos(self._omega * np.pi/180))

    def vA_t(self, t):
        '''

        '''
        # Get the true anomoly "f" from time
        f = self.theta(t)

        # Feed this into the orbit equation and add the systemic velocity
        return self.v1_f(f) + self._gamma


    def get_component_velocities(self, dates=None):
        '''
        Return both vA and vB for all dates provided.
        '''

        if dates is None and self.obs_dates is None:
            raise RuntimeError("Must provide input dates or specify observation dates upon creation of orbit object.")

        if dates is None and self.obs_dates is not None:
            dates = self.obs_dates

        dates = np.atleast_1d(dates)

        vAs = np.array([self.vA_t(date) for date in dates])

        return vAs

class SB2:
    '''
    Techniques describing solving for a binary orbit.
    '''
    def __init__(self, q, K, e, omega, P, T0, gamma, obs_dates=None, **kwargs):
        self._q = q # [M2/M1]
        self._K = K # [km/s]
        self._e = e
        self._omega = omega # [deg]
        self._P = P # [days]
        self._T0 = T0 # [JD]
        self._gamma = gamma # [km/s]

        # If we are going to be repeatedly predicting the orbit at a sequence of dates,
        # just store them to the object.
        self.obs_dates = obs_dates

    # Properties so that we can easily update subsets of the orbit.
    @property
    def q(self):
        return self._q

    @q.setter
    def q(self, value):
        self._q = value

    @property
    def K(self):
        return self._K

    @K.setter
    def K(self, value):
        self._K = value

    @property
    def e(self):
        return self._e

    @e.setter
    def e(self, value):
        self._e = value

    @property
    def omega(self):
        return self._omega

    @omega.setter
    def omega(self, value):
        self._omega = value

    @property
    def P(self):
        return self._P

    @P.setter
    def P(self, value):
        self._P = value

    @property
    def T0(self):
        return self._T0

    @T0.setter
    def T0(self, value):
        self._T0 = value

    @property
    def gamma(self):
        return self._gamma

    @gamma.setter
    def gamma(self, value):
        self._gamma = value

    def theta(self, t):
        '''Calculate the true anomoly for the A-B orbit.
        Input is in days.'''

        # t is input in seconds

        # Take a modulus of the period
        t = (t - self._T0) % self._P

        f = lambda E: E - self._e * np.sin(E) - 2 * np.pi * t/self._P
        E0 = 2 * np.pi * t / self._P

        E = fsolve(f, E0)[0]

        th = 2 * np.arctan(np.sqrt((1 + self._e)/(1 - self._e)) * np.tan(E/2.))

        if E < np.pi:
            return th
        else:
            return th + 2 * np.pi

    def v1_f(self, f):
        '''Calculate the component of A's velocity based on only the inner orbit.
        f is the true anomoly of this inner orbit.'''

        return self._K * (np.cos(self._omega * np.pi/180 + f) + self._e * np.cos(self._omega * np.pi/180))

    def vA_t(self, t):
        '''

        '''
        # Get the true anomoly "f" from time
        f = self.theta(t)

        # Feed this into the orbit equation and add the systemic velocity
        return self.v1_f(f) + self._gamma


    def v2_f(self, f):
        '''Calculate the component of B's velocity based on only the inner orbit.
        f is the true anomoly of this inner orbit.'''

        return -self._K/self._q * (np.cos(self._omega * np.pi/180 + f) + self._e * np.cos(self._omega * np.pi/180))

    def vB_t(self, t):
        f = self.theta(t)

        # Feed this into the orbit equation and add the systemic velocity
        return self.v2_f(f) + self._gamma


    def get_component_velocities(self, dates=None):
        '''
        Return both vA and vB for all dates provided.
        '''

        if dates is None and self.obs_dates is None:
            raise RuntimeError("Must provide input dates or specify observation dates upon creation of orbit object.")

        if dates is None and self.obs_dates is not None:
            dates = self.obs_dates

        dates = np.atleast_1d(dates)

        vAs = np.array([self.vA_t(date) for date in dates])
        vBs = np.array([self.vB_t(date) for date in dates])

        return (vAs, vBs)


class ST3:
    '''
    Techniques describing solving for a triple star orbit.
    '''
    def __init__(self, q_in, K_in, e_in, omega_in, P_in, T0_in, q_out, K_out, e_out, omega_out, P_out, T0_out, gamma, obs_dates=None, **kwargs):
        self._q_in = q_in # [M2/M1]
        self._K_in = K_in # [km/s]
        self._e_in = e_in
        self._omega_in = omega_in # [deg]
        self._P_in = P_in # [days]
        self._T0_in = T0_in # [JD]
        self._q_out = q_out # [M2/M1]
        self._K_out = K_out # [km/s]
        self._e_out = e_out
        self._omega_out = omega_out # [deg]
        self._P_out = P_out # [days]
        self._T0_out = T0_out # [JD]
        self._gamma = gamma # [km/s]

        # If we are going to be repeatedly predicting the orbit at a sequence of dates,
        # just store them to the object.
        self.obs_dates = obs_dates

    # Properties so that we can easily update subsets of the orbit.
    @property
    def q_in(self):
        return self._q_in

    @q_in.setter
    def q_in(self, value):
        self._q_in = value

    @property
    def K_in(self):
        return self._K_in

    @K_in.setter
    def K_in(self, value):
        self._K_in = value

    @property
    def e_in(self):
        return self._e_in

    @e_in.setter
    def e_in(self, value):
        self._e_in = value

    @property
    def omega_in(self):
        return self._omega_in

    @omega_in.setter
    def omega_in(self, value):
        self._omega_in = value

    @property
    def P_in(self):
        return self._P_in

    @P_in.setter
    def P_in(self, value):
        self._P_in = value

    @property
    def T0_in(self):
        return self._T0_in

    @T0_in.setter
    def T0_in(self, value):
        self._T0_in = value

    @property
    def q_out(self):
        return self._q_out

    @q_out.setter
    def q_out(self, value):
        self._q_out = value

    @property
    def K_out(self):
        return self._K_out

    @K_out.setter
    def K_out(self, value):
        self._K_out = value

    @property
    def e_out(self):
        return self._e_out

    @e_out.setter
    def e_out(self, value):
        self._e_out = value

    @property
    def omega_out(self):
        return self._omega_out

    @omega_out.setter
    def omega_out(self, value):
        self._omega_out = value

    @property
    def P_out(self):
        return self._P_out

    @P_out.setter
    def P_out(self, value):
        self._P_out = value

    @property
    def T0_out(self):
        return self._T0_out

    @T0_out.setter
    def T0_out(self, value):
        self._T0_out = value

    @property
    def gamma(self):
        return self._gamma

    @gamma.setter
    def gamma(self, value):
        self._gamma = value

    def theta_in(self, t):
        '''Calculate the true anomoly for the A-B orbit.'''

        # t is input in seconds

        # Take a modulus of the period
        t = (t - self._T0_in) % self._P_in

        f = lambda E: E - self._e_in * np.sin(E) - 2 * np.pi * t/self._P_in
        E0 = 2 * np.pi * t / self._P_in

        E = fsolve(f, E0)[0]

        th = 2 * np.arctan(np.sqrt((1 + self._e_in)/(1 - self._e_in)) * np.tan(E/2.))

        if E < np.pi:
            return th
        else:
            return th + 2 * np.pi

    def theta_out(self, t):
        '''Calculate the true anomoly for the (A-B) - C orbit.'''

        # t is input in seconds

        # Take a modulus of the period
        t = (t - self._T0_out) % self._P_out

        f = lambda E: E - self._e_out * np.sin(E) - 2 * np.pi * t/self._P_out
        E0 = 2 * np.pi * t / self._P_out

        E = fsolve(f, E0)[0]

        th = 2 * np.arctan(np.sqrt((1 + self._e_out)/(1 - self._e_out)) * np.tan(E/2.))

        if E < np.pi:
            return th
        else:
            return th + 2 * np.pi

    def v1_f(self, f):
        '''Calculate the component of A's velocity based on only the inner orbit.
        f is the true anomoly of this inner orbit.'''

        return self._K_in * (np.cos(self._omega_in * np.pi/180 + f) + self._e_in * np.cos(self._omega_in * np.pi/180))

    def v2_f(self, f):
        '''Calculate the component of B's velocity based on only the inner orbit.
        f is the true anomoly of this inner orbit.'''

        return -self._K_in/self._q_in * (np.cos(self._omega_in * np.pi/180 + f) + self._e_in * np.cos(self._omega_in * np.pi/180))


    def v3_f(self, f):
        '''Calculate the velocity of (A-B) based only on the outer orbit.
        f is the true anomoly of the outer orbit'''
        return  self._K_out * (np.cos(self._omega_out * np.pi/180 + f) + self._e_out * np.cos(self._omega_out * np.pi/180))


    def v3_f_C(self, f):
        '''Calculate the velocity of C based only on the outer orbit.
        f is the true anomoly of the outer orbit
        '''
        return -self._K_out / self._q_out * (np.cos(self._omega_out * np.pi/180 + f) + self._e_out * np.cos(self._omega_out * np.pi/180))


    def vA_t(self, t):

        # Get the true anomoly "f" from time
        f_in = self.theta_in(t)
        f_out = self.theta_out(t)

        v1 = self.v1_f(f_in)
        v3 = self.v3_f(f_out)

        return v1 + v3 + self.gamma

    def vB_t(self, t):

        # Get the true anolomy "f" from time
        f_in = self.theta_in(t)
        f_out = self.theta_out(t)

        v2 = self.v2_f(f_in)
        v3 = self.v3_f(f_out)

        return v2 + v3 + self.gamma

    def vC_t(self, t):

        # Get the true anolomy "f" from time
        f_out = self.theta_out(t)

        v3 = self.v3_f_C(f_out)

        return v3 + self.gamma


    def get_component_velocities(self, dates=None):
        '''
        Return vA, vB, and vC for all dates provided.
        '''

        if dates is None and self.obs_dates is None:
            raise RuntimeError("Must provide input dates or specify observation dates upon creation of orbit object.")

        if dates is None and self.obs_dates is not None:
            dates = self.obs_dates

        dates = np.atleast_1d(dates)


        vAs = np.array([self.vA_t(date) for date in dates])
        vBs = np.array([self.vB_t(date) for date in dates])
        vCs = np.array([self.vC_t(date) for date in dates])

        return (vAs, vBs, vCs)


models = {"SB1":SB1, "SB2":SB2, "ST3":ST3}
