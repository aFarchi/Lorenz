#!/usr/bin/env python

#__________________________________________________
# pyLorenz/utils/sampling/
# sampling.py
#__________________________________________________
# author        : colonel
# last modified : 2016/11/18
#__________________________________________________
#
# sampling methods
# all methods takes the log-weights as first argument, the number of indices to sample as second argument,
# the RNG as third argument and check as last argument
# and return the sampling indices
#

import numpy as np

#__________________________________________________

class ZeroWeightError(Exception):
    pass

#__________________________________________________

def accumulate_weights_check(t_w, t_check = True):
    # cumulative weights
    wc = t_w.cumsum()
    if t_check and wc[-1] == 0.0:
        # all weights are nearly zero
        # in principle this should never happen
        raise ZeroWeightError
    return wc

#__________________________________________________

def probabilistic_sampling(t_w, t_Ns, t_rng, t_check = True):
    # cumulative weights
    wc = accumulate_weights_check(t_w, t_check)
    return wc.searchsorted(t_rng.rand(t_Ns)*wc[-1])

#__________________________________________________

def stochastic_universal_sampling(t_w, t_Ns, t_rng, t_check = True):
    # cumulative weights
    wc   = accumulate_weights_check(t_w, t_check)
    # random comb
    comb = ( np.arange(t_Ns) + t_rng.rand() ) * ( wc[-1] / t_Ns )
    return wc.searchsorted(comb)

#__________________________________________________

def monte_carlo_metropolis_hastings_sampling(t_w, t_Ns, t_rng, t_check = True):
    # note: with this method, Ns must be equal to weights.size
    i = np.arange(t_Ns)

    if t_check and t_w.sum() == 0.0: 
        # all weights are nearly zero
        # in principle this should never happen
        raise ZeroWeightError

    # arbitrarily choose first sample
    sample = 0
    for ns in range(t_Ns):
        # accept next sample with proba w[next]/w[previous]
        # here 'previous' = 'sample' and 'next' = 'ns'
        if t_w[sample] * t_rng.rand() <= t_w[ns]:
            sample = ns
        # else duplicate previous sample
        else:
            i[ns]  = sample

    return i

#__________________________________________________

