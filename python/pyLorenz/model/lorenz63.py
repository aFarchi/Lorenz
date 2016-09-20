#! /usr/bin/env python

#__________________________________________________
# pyLorenz/model/
# lorenz63.py
#__________________________________________________
# author        : colonel
# last modified : 2016/9/20
#__________________________________________________
#
# classes to handle the a Lorenz 1963 model
# i.e. process compute the dervative at a given point according to a Lorenz 1963 model
#

import numpy as np

from ..utils.process.abstractprocess import AbstractStochasticProcess
from ..utils.process.abstractprocess import AbstractDeterministicProcess

#__________________________________________________

class AbstractLorenz63Model:

    #_________________________

    def __init__(self):
        # constructor
        self.m_stateDimension = 3

    #_________________________

    def setParameters(self, t_sigma = 10.0, t_beta = 8.0 / 3.0, t_rho = 28.0):
        # set model parameters
        self.m_sigma = t_sigma
        self.m_beta  = t_beta
        self.m_rho   = t_rho

    #_________________________

    def deterministicProcess(self, t_x):
        # compute dx according to the model
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
    pass

#__________________________________________________

class DeterministicLorenz63Model(AbstractLorenz63Model, AbstractDeterministicProcess):
    pass

#__________________________________________________

