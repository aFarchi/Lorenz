#! /usr/bin/env python

#__________________________________________________
# pyLorenz/utils/sampling/
# regulariser.py
#__________________________________________________
# author        : colonel
# last modified : 2016/12/17
#__________________________________________________
#
# Regulatisations classes
#

import numpy as np

#__________________________________________________

class NoRegulariser(object):

    #_________________________

    def __init__(self):
        pass

    #_________________________

    def regularisation(self, t_x, t_resampling_indices, t_weights):
        t_x[:] = t_x[t_resampling_indices]

#__________________________________________________

class JitterRegulariser(object):

    #_________________________

    def __init__(self, t_rng, t_jitter_std):
        # rng
        self.m_rng        = t_rng
        # jitter standard deviation
        self.m_jitter_std = t_jitter_std

    #_________________________

    def regularisation(self, t_x, t_resampling_indices, t_weights):
        # sample errors
        errors = self.m_jitter_std * self.m_rng.standard_normal(t_x.shape)
        # return error and remove sample mean
        t_x[:] = t_x[t_resampling_indices] + errors - errors.mean(axis=0)

#__________________________________________________

class UnivariateGaussianKernelRegulariser(object):

    def __init__(self, t_rng, t_variance_min, t_bandwidth_scale):
        # minimum of regularisation deviation
        self.m_variance_min = t_variance_min
        # unit bandwidth
        self.m_h            = t_bandwidth_scale * np.power(4/3, 1/5)
        # rng
        self.m_rng          = t_rng

    #_________________________

    def regularisation(self, t_x, t_resampling_indices, t_weights):
        # compute univariate std
        mean   = np.average(t_x, axis=0, weights=t_weights)
        var    = np.maximum(np.average((t_x-mean)**2, axis=0, weights=t_weights), self.m_variance_min)
        # adaptative bandwidth
        h      = ( self.m_h / np.power( t_x.shape[0] , 0.2 ) ) * np.sqrt(var)
        # sample errors
        errors = h * self.m_rng.standard_normal(t_x.shape)
        # add error and remove sample mean
        t_x    = t_x[t_resampling_indices] + errors - errors.mean(axis = 0)

#__________________________________________________

