# example of a script that could provide a user-defined prior
# this function must take a numpy array ``p``, and return the natural log (ln) of the prior value.
# since this code will be imported by ``psoap-sample`` before being run, no import numpy or other statements are necessary.
def prior(p):
    '''
    Sample of a user-defined prior for a double-lined spectroscopic binary.
    '''
    (q, K, e, omega, P, T0, gamma), (amp_f, l_f, amp_g, l_g) = convert_vector_p(p)

    if q < 0.0 or K < 0.0 or e < 0.0 or e > 1.0 or P < 0.0 or omega < -90 or omega > 450 or amp_f < 0.0 or l_f < 0.0 or amp_g < 0.0 or l_g < 0.0:
        return -np.inf
    else:
        return 0.0
