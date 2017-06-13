import numpy as np
from scipy.optimize import fsolve


class SB1:
    '''
    Describing a single-line Spectroscopic binary.
    '''
    def __init__(self, K, e, omega, P, T0, gamma, obs_dates=None, **kwargs):
        self.K = K # [km/s]
        self.e = e
        self.omega = omega # [deg]
        self.P = P # [days]
        self.T0 = T0 # [JD]
        self.gamma = gamma # [km/s]

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

    def _vA_t(self, t):
        '''

        '''
        # Get the true anomoly "f" from time
        f = self._theta(t)

        # Feed this into the orbit equation and add the systemic velocity
        return self._v1_f(f) + self.gamma


    def get_velocities(self, dates=None):
        '''
        Return vA for all dates provided.
        '''

        if dates is None and self.obs_dates is None:
            raise RuntimeError("Must provide input dates or specify observation dates upon creation of orbit object.")

        if dates is None and self.obs_dates is not None:
            dates = self.obs_dates

        dates = np.atleast_1d(dates)

        vAs = np.array([self._vA_t(date) for date in dates])

        return np.atleast_2d(vAs)

class SB2(SB1):
    '''
    Methods for solving a double-lined spectroscopic orbit.
    '''
    def __init__(self, q, K, e, omega, P, T0, gamma, obs_dates=None, **kwargs):
        super().__init__(K, e, omega, P, T0, gamma, obs_dates=obs_dates, **kwargs)
        self.q = q

    def _v2_f(self, f):
        '''Calculate the component of B's velocity based on only the inner orbit.
        f is the true anomoly of this inner orbit.'''

        return -self.K/self.q * (np.cos(self.omega * np.pi/180 + f) + self.e * np.cos(self.omega * np.pi/180))

    def _vB_t(self, t):
        f = self._theta(t)

        # Feed this into the orbit equation and add the systemic velocity
        return self._v2_f(f) + self.gamma

    def get_velocities(self, dates=None):
        '''
        Return both vA and vB for all dates provided.
        '''

        if dates is None and self.obs_dates is None:
            raise RuntimeError("Must provide input dates or specify observation dates upon creation of orbit object.")

        if dates is None and self.obs_dates is not None:
            dates = self.obs_dates

        dates = np.atleast_1d(dates)

        vAs = np.array([self._vA_t(date) for date in dates])
        vBs = np.array([self._vB_t(date) for date in dates])

        return (vAs, vBs)

class ST1:
    '''
    A hierarchical triple star orbit for which we only see the primary lines.
    '''
    def __init__(self, K_in, e_in, omega_in, P_in, T0_in, K_out, e_out, omega_out, P_out, T0_out, gamma, obs_dates=None, **kwargs):
        self.K_in = K_in # [km/s]
        self.e_in = e_in
        self.omega_in = omega_in # [deg]
        self.P_in = P_in # [days]
        self.T0_in = T0_in # [JD]
        self.K_out = K_out # [km/s]
        self.e_out = e_out
        self.omega_out = omega_out # [deg]
        self.P_out = P_out # [days]
        self.T0_out = T0_out # [JD]
        self.gamma = gamma # [km/s]

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


    def _v3_f(self, f):
        '''Calculate the velocity of (A-B) based only on the outer orbit.
        f is the true anomoly of the outer orbit'''
        return  self.K_out * (np.cos(self.omega_out * np.pi/180 + f) + self.e_out * np.cos(self.omega_out * np.pi/180))


    def _vA_t(self, t):

        # Get the true anomoly "f" from time
        f_in = self._theta_in(t)
        f_out = self._theta_out(t)

        v1 = self._v1_f(f_in)
        v3 = self._v3_f(f_out)

        return v1 + v3 + self.gamma

    def get_velocities(self, dates=None):
        '''
        Return vA for all dates provided.
        '''

        if dates is None and self.obs_dates is None:
            raise RuntimeError("Must provide input dates or specify observation dates upon creation of orbit object.")

        if dates is None and self.obs_dates is not None:
            dates = self.obs_dates

        dates = np.atleast_1d(dates)

        vAs = np.array([self._vA_t(date) for date in dates])

        return np.atleast_2d(vAs)

class ST2(ST1):
    '''
    Solving a triple star orbit for which we see both sets of lines of the inner binary.
    '''
    def __init__(self, q_in, K_in, e_in, omega_in, P_in, T0_in, K_out, e_out, omega_out, P_out, T0_out, gamma, obs_dates=None, **kwargs):
        super().__init__(K_in, e_in, omega_in, P_in, T0_in, K_out, e_out, omega_out, P_out, T0_out, gamma, obs_dates=obs_dates, **kwargs)

        self.q_in = q_in # [M2/M1]

    def _v2_f(self, f):
        '''Calculate the component of B's velocity based on only the inner orbit.
        f is the true anomoly of this inner orbit.'''

        return -self.K_in/self.q_in * (np.cos(self.omega_in * np.pi/180 + f) + self.e_in * np.cos(self.omega_in * np.pi/180))

    def _vB_t(self, t):

        # Get the true anolomy "f" from time
        f_in = self._theta_in(t)
        f_out = self._theta_out(t)

        v2 = self._v2_f(f_in)
        v3 = self._v3_f(f_out)

        return v2 + v3 + self.gamma

    def get_velocities(self, dates=None):
        '''
        Return vA and vB for all dates provided.
        '''

        if dates is None and self.obs_dates is None:
            raise RuntimeError("Must provide input dates or specify observation dates upon creation of orbit object.")

        if dates is None and self.obs_dates is not None:
            dates = self.obs_dates

        dates = np.atleast_1d(dates)


        vAs = np.array([self._vA_t(date) for date in dates])
        vBs = np.array([self._vB_t(date) for date in dates])

        return (vAs, vBs)


class ST3(ST2):
    '''
    Techniques describing solving for a triple star orbit for which we see all lines.
    '''
    def __init__(self, q_in, K_in, e_in, omega_in, P_in, T0_in, q_out, K_out, e_out, omega_out, P_out, T0_out, gamma, obs_dates=None, **kwargs):
        super().__init__(q_in, K_in, e_in, omega_in, P_in, T0_in, K_out, e_out, omega_out, P_out, T0_out, gamma, obs_dates=obs_dates, **kwargs)

        self.q_out = q_out # [M2/M1]

    def _v3_f_C(self, f):
        '''Calculate the velocity of C based only on the outer orbit.
        f is the true anomoly of the outer orbit
        '''
        return -self.K_out / self.q_out * (np.cos(self.omega_out * np.pi/180 + f) + self.e_out * np.cos(self.omega_out * np.pi/180))


    def _vC_t(self, t):

        # Get the true anolomy "f" from time
        f_out = self._theta_out(t)

        v3 = self._v3_f_C(f_out)

        return v3 + self.gamma


    def get_velocities(self, dates=None):
        '''
        Return vA, vB, and vC for all dates provided.
        '''

        if dates is None and self.obs_dates is None:
            raise RuntimeError("Must provide input dates or specify observation dates upon creation of orbit object.")

        if dates is None and self.obs_dates is not None:
            dates = self.obs_dates

        dates = np.atleast_1d(dates)


        vAs = np.array([self._vA_t(date) for date in dates])
        vBs = np.array([self._vB_t(date) for date in dates])
        vCs = np.array([self._vC_t(date) for date in dates])

        return (vAs, vBs, vCs)


models = {"SB1":SB1, "SB2":SB2, "ST1":ST1, "ST2":ST2, "ST3":ST3}
