#! /usr/bin/env python

#__________________________________________________
# pyLorenz/model/
# lorenz63.py
#__________________________________________________
# author        : colonel
# last modified : 2016/9/21
#__________________________________________________
#
# classes to handle the a Lorenz 1963 model
# i.e. process compute the dervative at a given point according to a Lorenz 1963 model
#

import numpy as np

from ..utils.process.abstractprocess       import AbstractStochasticProcess
from ..utils.process.abstractprocess       import AbstractDeterministicProcess
from ..utils.random.independantgaussianrng import IndependantGaussianRNG

#__________________________________________________

class AbstractLorenz63Model(object):

    #_________________________

    def __init__(self, t_sigma = 10.0, t_beta = 8.0 / 3.0, t_rho = 28.0):
        self.m_stateDimension = 3
        self.setLorenz63ModelParameters(t_sigma, t_beta, t_rho)

    #_________________________

    def setLorenz63ModelParameters(self, t_sigma = 10.0, t_beta = 8.0 / 3.0, t_rho = 28.0):
        # set model parameters
        self.m_sigma = t_sigma
        self.m_beta  = t_beta
        self.m_rho   = t_rho

    #_________________________

    def deterministicProcess(self, t_x, t_t):
        # compute dx at point (x,t) according to the model
        # here dxonly depens on x
        shape = t_x.shape
        if len(shape) == 1:
            # for one point
            dx    = np.zeros(3)
            dx[0] = self.m_sigma * ( t_x[1] - t_x[0] )
            dx[1] = ( self.m_rho - t_x[2] ) * t_x[0] - t_x[1]
            dx[2] = t_x[0] * t_x[1] - self.m_beta * t_x[2]
        else:
            # for multiple points
            Ns       = shape[0]
            dx       = np.zeros(shape=(Ns, 3))
            dx[:, 0] = self.m_sigma * ( t_x[:, 1] - t_x[:, 0] )
            dx[:, 1] = ( self.m_rho - t_x[:, 2] ) * t_x[:, 0] - t_x[:, 1]
            dx[:, 2] = t_x[:, 0] * t_x[:, 1] - self.m_beta * t_x[:, 2]
        return dx

#__________________________________________________

class StochasticLorenz63Model(AbstractLorenz63Model, AbstractStochasticProcess):

    #_________________________

    def __init__(self, t_sigma = 10.0, t_beta = 8.0 / 3.0, t_rho = 28.0, t_eg = IndependantGaussianRNG()):
        AbstractLorenz63Model.__init__(self, t_sigma, t_beta, t_rho)
        AbstractStochasticProcess.__init__(self, t_eg)

    #_________________________

    def deterministicModel(self):
        return DeterministicLorenz63Model(self.m_sigma, self.m_beta, self.m_rho)

#__________________________________________________

class DeterministicLorenz63Model(AbstractLorenz63Model, AbstractDeterministicProcess):

    #_________________________

    def __init__(self, t_sigma = 10.0, t_beta = 8.0 / 3.0, t_rho = 28.0):
        AbstractLorenz63Model.__init__(self, t_sigma, t_beta, t_rho)
        AbstractDeterministicProcess.__init__(self)

    #_________________________

    def deterministicModel(self):
        return self

#__________________________________________________

