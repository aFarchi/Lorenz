#! /usr/bin/env python

#__________________________________________________
# pyLorenz/utils/integration/
# rk4integrator.py
#__________________________________________________
# author        : colonel
# last modified : 2016/10/6
#__________________________________________________
#
# classes to handle the integration step according to RK4 scheme
#

import numpy as np

from abstractintegrator              import AbstractIntegrator
from ..random.independantgaussianrng import IndependantGaussianRNG
from ...model.lorenz63               import DeterministicLorenz63Model
from ..process.abstractprocess       import AbstractStochasticProcess
from ..process.abstractprocess       import AbstractMultiStochasticProcess
from ..process.abstractprocess       import AbstractDeterministicProcess

#__________________________________________________

class AbstractRK4Integrator(AbstractIntegrator):

    #_________________________

    def __init__(self, t_dt = 0.01, t_model = DeterministicLorenz63Model()):
        AbstractIntegrator.__init__(self, t_dt, t_model)

    #_________________________

    def deterministicProcess(self, t_x, t_t):
        # integrates xn
        dx1 = self.m_model.process(t_x, t_t)
        x1  = t_x + dx1 * self.m_dt / 2.0                 # + potential errors
        dx2 = self.m_model.process(x1, t_t+self.m_dt/2.0)
        x2  = t_x + dx2 * self.m_dt / 2.0                 # + potential errors
        dx3 = self.m_model.process(x2, t_t+self.m_dt/2.0)
        x3  = t_x + dx3 * self.m_dt                       # + potential errors
        dx4 = self.m_model.process(x3, t_t+self.m_dt)
        dx  = ( dx1 + 2.0 * dx2 + 2.0 * dx3 + dx4 ) / 6.0
        return t_x + dx * self.m_dt                       # + potential errors

    #_________________________

    def multiStochasticProcess(self, t_xn, t_nt):
        # integrates xn
        dx1 = self.m_model.process(t_x, t_t)
        x1  = self.m_errorGenerator[0].addError(t_x + dx1 * self.m_dt / 2.0, t_t)
        dx2 = self.m_model.process(x1, t_t+self.m_dt/2.0)
        x2  = self.m_errorGenerator[1].addError(t_x + dx2 * self.m_dt / 2.0, t_t)
        dx3 = self.m_model.process(x2, t_t+self.m_dt/2.0)
        x3  = self.m_errorGenerator[2].addError(t_x + dx3 * self.m_dt, t_t)
        dx4 = self.m_model.process(x3, t_t+self.m_dt)
        dx  = ( dx1 + 2.0 * dx2 + 2.0 * dx3 + dx4 ) / 6.0
        return self.m_errorGenerator[3].addError(t_x + dx * self.m_dt, t_t)

#__________________________________________________

class StochasticRK4Integrator(AbstractRK4Integrator, AbstractStochasticProcess):

    #_________________________

    def __init__(self, t_dt = 0.01, t_model = DeterministicLorenz63Model(), t_eg = IndependantGaussianRNG()):
        AbstractRK4Integrator.__init__(self, t_dt, t_model)
        AbstractStochasticProcess.__init__(self, t_eg)

#__________________________________________________
 
class MultiStochasticRK4Integrator(AbstractRK4Integrator, AbstractMultiStochasticProcess):

    #_________________________

    def __init__(self, t_dt = 0.01, t_model = DeterministicLorenz63Model(), t_eg = []):
        AbstractRK4Integrator.__init__(self)
        AbstractMultiStochasticProcess.__init__(self, t_eg)

#__________________________________________________
 
class DeterministicRK4Integrator(AbstractRK4Integrator, AbstractDeterministicProcess):

    #_________________________

    def __init__(self, t_dt = 0.01, t_model = DeterministicLorenz63Model()):
        AbstractRK4Integrator.__init__(self, t_dt, t_model)
        AbstractDeterministicProcess.__init__(self)

#__________________________________________________

