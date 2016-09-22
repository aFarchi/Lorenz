#! /usr/bin/env python

#__________________________________________________
# pyLorenz/utils/integration/
# rk4integrator.py
#__________________________________________________
# author        : colonel
# last modified : 2016/9/21
#__________________________________________________
#
# classes to handle the integration step according to RK4 scheme
#

import numpy as np

from abstractintegrator              import AbstractStochasticIntegrator
from abstractintegrator              import AbstractMultiStochasticIntegrator
from abstractintegrator              import AbstractDeterministicIntegrator
from ..random.independantgaussianrng import IndependantGaussianRNG
from ...model.lorenz63               import DeterministicLorenz63Model

#__________________________________________________

class AbstractRK4Integrator:

    #_________________________

    def __init__(self):
        self.m_deterministicIntegratorClass = DeterministicRK4Integrator

    #_________________________

    def deterministicProcess(self, t_xn, t_nt):
        # integrates xn
        dx1 = self.m_model.process(t_xn, t_nt*self.m_dt)
        x1  = self.potentiallyAddError(t_xn + dx1 * self.m_dt / 2.0)
        dx2 = self.m_model.process(x1, (t_nt+0.5)*self.m_dt)
        x2  = self.potentiallyAddError(t_xn + dx2 * self.m_dt / 2.0)
        dx3 = self.m_model.process(x2, (t_nt+0.5)*self.m_dt)
        x3  = self.potentiallyAddError(t_xn + dx3 * self.m_dt)
        dx4 = self.m_model.process(x3, (t_nt+1.0)*self.m_dt)
        dx  = ( dx1 + 2.0 * dx2 + 2.0 * dx3 + dx4 ) / 6.0
        return self.potentiallyAddError(t_xn + dx * self.m_dt)

#__________________________________________________

class StochasticRK4Integrator(AbstractRK4Integrator, AbstractStochasticIntegrator):

    #_________________________

    def __init__(self, t_eg = IndependantGaussianRNG(), t_dt = 0.01, t_model = DeterministicLorenz63Model()):
        AbstractRK4Integrator.__init__(self)
        AbstractStochasticIntegrator.__init__(self, t_eg, t_dt, t_model)

#__________________________________________________
 
class MultiStochasticRK4Integrator(AbstractRK4Integrator, AbstractMultiStochasticIntegrator):

    #_________________________

    def __init__(self, t_eg = IndependantGaussianRNG(), t_dt = 0.01, t_model = DeterministicLorenz63Model()):
        AbstractRK4Integrator.__init__(self)
        AbstractMultiStochasticIntegrator.__init__(self, t_eg, t_dt, t_model)

#__________________________________________________
 
class DeterministicRK4Integrator(AbstractRK4Integrator, AbstractDeterministicIntegrator):

    #_________________________

    def __init__(self, t_dt = 0.01, t_model = DeterministicLorenz63Model()):
        AbstractRK4Integrator.__init__(self)
        AbstractDeterministicIntegrator.__init__(self, t_dt, t_model)

#__________________________________________________

