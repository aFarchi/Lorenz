#! /usr/bin/env python

#__________________________________________________
# pyLorenz/utils/integration/
# rk2integrator.py
#__________________________________________________
# author        : colonel
# last modified : 2016/9/21
#__________________________________________________
#
# classes to handle the integration step according to RK2 scheme
#

import numpy as np

from abstractintegrator              import AbstractStochasticIntegrator
from abstractintegrator              import AbstractMultiStochasticIntegrator
from abstractintegrator              import AbstractDeterministicIntegrator
from ..random.independantgaussianrng import IndependantGaussianRNG
from ...model.lorenz63               import DeterministicLorenz63Model

#__________________________________________________

class AbstractRK2Integrator:

    #_________________________

    def __init__(self):
        self.m_deterministicIntegratorClass = DeterministicRK2Integrator

    #_________________________

    def deterministicProcess(self, t_xn, t_nt):
        # integrates xn
        dx = self.m_model.process(t_xn, t_nt*self.m_dt)
        x  = self.potentiallyAddError(t_xn + dx * self.m_dt / 2.0)
        dx = self.m_model.process(x, (t_nt+0.5)*self.m_dt)
        return self.potentiallyAddError(t_xn + dx * self.m_dt)

#__________________________________________________

class StochasticRK2Integrator(AbstractRK2Integrator, AbstractStochasticIntegrator):

    #_________________________

    def __init__(self, t_eg = IndependantGaussianRNG(), t_dt = 0.01, t_model = DeterministicLorenz63Model()):
        AbstractRK2Integrator.__init__(self)
        AbstractStochasticIntegrator.__init__(self, t_eg, t_dt, t_model)

#__________________________________________________
 
class MultiStochasticRK2Integrator(AbstractRK2Integrator, AbstractMultiStochasticIntegrator):

    #_________________________

    def __init__(self, t_eg = IndependantGaussianRNG(), t_dt = 0.01, t_model = DeterministicLorenz63Model()):
        AbstractRK2Integrator.__init__(self)
        AbstractMultiStochasticIntegrator.__init__(self, t_eg, t_dt, t_model)

#__________________________________________________
 
class DeterministicRK2Integrator(AbstractRK2Integrator, AbstractDeterministicIntegrator):

    #_________________________

    def __init__(self, t_dt = 0.01, t_model = DeterministicLorenz63Model()):
        AbstractRK2Integrator.__init__(self)
        AbstractDeterministicIntegrator.__init__(self, t_dt, t_model)

#__________________________________________________

