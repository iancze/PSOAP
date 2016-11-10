import numpy as np
from scipy.optimize import fsolve, minimize

from psoap import constants as C

class SB1:
    '''
    Describing a single-line Spectroscopic binary.
    '''
    def __init__(self, K, e, omega, P, T0, gamma, obs_dates=None):
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
    def __init__(self, q, K, e, omega, P, T0, gamma, obs_dates=None):
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


class SB3:
    '''
    Techniques describing solving for a triple star orbit.
    '''
    def __init__(self, q_inner, K_inner, e_inner, omega_inner, P_inner, T0_inner, q_outer, K_outer, e_outer, omega_outer, P_outer, T0_outer, gamma, obs_dates=None):
        self.q_inner = q_inner # [M2/M1]
        self.K_inner = K_inner # [km/s]
        self.e_inner = e_inner
        self.omega_inner = omega_inner # [deg]
        self.P_inner = P_inner # [days]
        self.T0_inner = T0_inner # [JD]
        self.q_outer = q_outer # [M2/M1]
        self.K_outer = K_outer # [km/s]
        self.e_outer = e_outer
        self.omega_outer = omega_outer # [deg]
        self.P_outer = P_outer # [days]
        self.T0_outer = T0_outer # [JD]
        self.gamma = gamma # [km/s]

        # If we are going to be repeatedly predicting the orbit at a sequence of dates,
        # just store them to the object.
        self.obs_dates = obs_dates

    # Properties so that we can easily update subsets of the orbit.
    @property
    def q_inner(self):
        return self.q_inner

    @q_inner.setter
    def q_inner(self, value):
        self.q_inner = value

    @property
    def K_inner(self):
        return self.K_inner

    @K_inner.setter
    def K_inner(self, value):
        self.K_inner = value

    @property
    def e_inner(self):
        return self.e_inner

    @e_inner.setter
    def e_inner(self, value):
        self.e_inner = value

    @property
    def omega_inner(self):
        return self.omega_inner

    @omega_inner.setter
    def omega_inner(self, value):
        self.omega_inner = value

    @property
    def P_inner(self):
        return self.P_inner

    @P_inner.setter
    def P_inner(self, value):
        self.P_inner = value

    @property
    def T0_inner(self):
        return self.T0_inner

    @T0_inner.setter
    def T0_inner(self, value):
        self.T0_inner = value

    @property
    def q_outer(self):
        return self.q_outer

    @q_outer.setter
    def q_outer(self, value):
        self.q_outer = value

    @property
    def K_outer(self):
        return self.K_outer

    @K_outer.setter
    def K_outer(self, value):
        self.K_outer = value

    @property
    def e_outer(self):
        return self.e_outer

    @e_outer.setter
    def e_outer(self, value):
        self.e_outer = value

    @property
    def omega_outer(self):
        return self.omega_outer

    @omega_outer.setter
    def omega_outer(self, value):
        self.omega_outer = value

    @property
    def P_outer(self):
        return self.P_outer

    @P_outer.setter
    def P_outer(self, value):
        self.P_outer = value

    @property
    def T0_outer(self):
        return self.T0_outer

    @T0_outer.setter
    def T0_outer(self, value):
        self.T0_outer = value


    @property
    def gamma(self):
        return self.gamma

    @gamma.setter
    def gamma(self, value):
        self.gamma = value


    def theta_inner(self, t):
        '''Calculate the true anomoly for the A-B orbit.'''

        # t is input in seconds

        # Take a modulus of the period
        t = (t - self.T0_inner) % self.P_inner

        f = lambda E: E - self.e_inner * np.sin(E) - 2 * np.pi * t/self.P_inner
        E0 = 2 * np.pi * t / self.P_inner

        E = fsolve(f, E0)[0]

        th = 2 * np.arctan(np.sqrt((1 + self.e_inner)/(1 - self.e_inner)) * np.tan(E/2.))

        if E < np.pi:
            return th
        else:
            return th + 2 * np.pi

    def theta_outer(self, t):
        '''Calculate the true anomoly for the (A-B) - C orbit.'''

        # t is input in seconds

        # Take a modulus of the period
        t = (t - self.T0_outer) % self.P_outer

        f = lambda E: E - e_outer * np.sin(E) - 2 * np.pi * t/self.P_outer
        E0 = 2 * np.pi * t / self.P_outer

        E = fsolve(f, E0)[0]

        th = 2 * np.arctan(np.sqrt((1 + self.e_outer)/(1 - self.e_outer)) * np.tan(E/2.))

        if E < np.pi:
            return th
        else:
            return th + 2 * np.pi


    def v3_f(self, f):
        '''Calculate the velocity of (A-B) based only on the outer orbit.
        f is the true anomoly of the outer orbit'''
        return  self.K_outer * (np.cos(self.omega_outer * np.pi/180 + f) + self.e_outer * np.cos(self.omega_outer * np.pi/180))


    def v3_f_C(self, f):
        '''Calculate the velocity of C based only on the outer orbit.
        f is the true anomoly of the outer orbit
        '''
        return -self.K_outer / self.q_outer * (np.cos(self.omega_outer * np.pi/180 + f) + self.e_outer * np.cos(self.omega_outer * np.pi/180))


    def vA_t(self, t):

        # Get the true anomoly "f" from time
        f_inner = self.theta_inner(t)
        f_outer = self.theta_outer(t)

        v1 = self.v1_f(f_inner)
        v3 = self.v3_f(f_outer)

        return v1 + v3 + self.gamma

    def vB_t(self, t):

        # Get the true anolomy "f" from time
        f_inner = self.theta_inner(t)
        f_outer = self.theta_outer(t)

        v2 = self.v2_f(f_inner)
        v3 = self.v3_f(f_outer)

        return v2 + v3 + self.gamma

    def vC_t(self, t):

        # Get the true anolomy "f" from time
        f_outer = self.theta_outer(t)

        v3 = self.v3_f_C(f_outer)

        return v3 + self.gamma


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
        vCs = np.array([self.vC_t(date) for date in dates])

        return (vAs, vBs, vCs)


def main():

    # Make some fake observations and parameters and see how they compare.

    # Systemic velocity measured from the relative point of setting the highest S/N epoch (the latest one) to zero.
    gamma = 5.0 # km/s

    # Primary - Secondary Orbit
    K_1 = 3.5 # [km/s]
    T0_inner = 2452000.0 * day # [s] JD
    P_inner = 3.3 * day # [s]
    e_inner = 0.1
    omega_inner = 0.0 # [radians]

    q_inner = 0.2 # The mass ratio between component 2 and 1, q = M_2 / M_1
    q_outer = 0.2

    # Third component orbit around (Primary + Secondary)
    K_3 = 2.04 # [km/s]
    T0_outer = 2453535. * day # [s] JD
    P_outer = 35 * day # [s] JD
    e_outer = 0.061
    omega_outer = 276 * deg # [radians]

    dates = np.load("dates.npy")

    vAs = get_vA(dates)
    vBs = get_vB(dates)
    vCs = get_vC(dates)

    dates_fine = np.linspace(dates[0], dates[-1], num=100)
    vA_fine = get_vA(dates_fine)
    vB_fine = get_vB(dates_fine)
    vC_fine = get_vC(dates_fine)

    np.save("vAs.npy", vAs)
    np.save("vBs.npy", vBs)
    np.save("vCs.npy", vCs)

    fig, ax = plt.subplots()
    ax.plot(dates_fine - 2400000, vA_fine, "b")
    ax.plot(dates - 2400000, vAs, "bo")

    ax.plot(dates_fine - 2400000, vB_fine, "g")
    ax.plot(dates - 2400000, vBs, "go")

    ax.plot(dates_fine - 2400000, vC_fine, "r")
    ax.plot(dates - 2400000, vCs, "ro")

    ax.axhline(gamma, ls="-.", color="0.5")

    ax.set_xlabel("JD - 2400000 [day]")
    ax.set_ylabel(r"$v$ [km/s]")

    fig.savefig("orbits.png")

if __name__ == "__main__":
    main()
