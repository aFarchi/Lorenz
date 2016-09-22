#! /usr/bin/env python

#__________________________________________________
# pyLorenz/utils/integration/
# eulerexplintegrator.py
#__________________________________________________
# author        : colonel
# last modified : 2016/9/21
#__________________________________________________
#
# classes to handle the integration step according to euler's explicit scheme
#

import numpy as np

from abstractintegrator              import AbstractStochasticIntegrator
from abstractintegrator              import AbstractDeterministicIntegrator
from ..random.independantgaussianrng import IndependantGaussianRNG
from ...model.lorenz63               import DeterministicLorenz63Model

#__________________________________________________

class AbstractEulerExplIntegrator(object):

    #_________________________

    def __init__(self):
        self.m_deterministicIntegratorClass = DeterministicEulerExplIntegrator
    
    #_________________________

    def deterministicProcess(self, t_xn, t_nt):
        # integrates xn
        dx = self.m_model.process(t_xn, t_nt*self.m_dt)
        return t_xn + dx * self.m_dt

#__________________________________________________

class StochasticEulerExplIntegrator(AbstractEulerExplIntegrator, AbstractStochasticIntegrator):

    #_________________________

    def __init__(self, t_eg = IndependantGaussianRNG(), t_dt = 0.01, t_model = DeterministicLorenz63Model()):
        AbstractEulerExplIntegrator.__init__(self)
        AbstractStochasticIntegrator.__init__(self, t_eg, t_dt, t_model)

#__________________________________________________

class DeterministicEulerExplIntegrator(AbstractEulerExplIntegrator, AbstractDeterministicIntegrator):

    #_________________________

    def __init__(self, t_dt = 0.01, t_model = DeterministicLorenz63Model()):
        AbstractEulerExplIntegrator.__init__(self)
        AbstractDeterministicIntegrator.__init__(self, t_dt, t_model)

#__________________________________________________

