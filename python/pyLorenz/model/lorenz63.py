#! /usr/bin/env python

#__________________________________________________
# pyLorenz/model/
# lorenz63.py
#__________________________________________________
# author        : colonel
# last modified : 2016/9/20
#__________________________________________________
#
# class to handle the step function of a Lorenz 1963 model
#

import numpy as np
from ..utils.random.abstractStochasticProcess import AbstractStochasticProcess

#__________________________________________________

class Lorenz63Model(AbstractStochasticProcess):

    #_________________________

    def __init__(self):
        # constructor
        self.m_stateDimension = 3
        self.setParameters()

    #_________________________

    def setParameters(self, t_sigma = 10.0, t_beta = 8.0 / 3.0, t_rho = 28.0):
        # set model parameters
        self.m_sigma = t_sigma
        self.m_beta  = t_beta
        self.m_rho   = t_rho

    #_________________________

    def setIntegrator(self, t_integrator):
        # set integrator
        self.m_integrator = t_integrator

    #_________________________

    def computeDerivate(self, t_x):
        # compute dx according to the model
        # t_x is not modified
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

    #_________________________

    def deterministicProcessForward(self, t_xn, **kwargs):
        # step forward : xnpp = model(t_xn)
        # t_xn is not modified
        if 't_stochastic' in kwargs:
            if kwargs['t_stochastic']:
                return self.m_integrator.stochasticProcessForward(self, t_xn)
        return self.m_integrator.deterministicProcessForward(self, t_xn)

#__________________________________________________

