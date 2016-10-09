#! /usr/bin/env python

#__________________________________________________
# pyLorenz/utils/integration/
# kpintegrator.py
#__________________________________________________
# author        : colonel
# last modified : 2016/10/9
#__________________________________________________
#
# classes to handle the integration step according to KP scheme
#

import numpy as np

from abstractintegrator              import AbstractIntegrator
from ..random.independantgaussianrng import IndependantGaussianRNG
from ...model.lorenz63               import DeterministicLorenz63Model
from ..process.abstractprocess       import AbstractStochasticProcess
from ..process.abstractprocess       import AbstractMultiStochasticProcess
from ..process.abstractprocess       import AbstractDeterministicProcess

#__________________________________________________

class AbstractKPIntegrator(AbstractIntegrator):

    #_________________________

    def __init__(self, t_dt = 0.01, t_model = DeterministicLorenz63Model()):
        AbstractIntegrator.__init__(self, t_dt, t_model)

    #_________________________

    def deterministicProcess(self, t_x, t_t):
        # integrates x according to the model
        dx1 = self.m_model.process(t_x, t_t)
        x1  = t_x + dx1 * self.m_dt                   # + potential errors
        dx2 = self.m_model.process(x1, t_t+self.m_dt)
        return t_x + ( dx1 + dx2 ) * self.m_dt / 2.0  # + potential errors
    #_________________________

    def multiStochasticProcess(self, t_x, t_t):
        # the same as deterministicProcess but including the potential errors
        dx1 = self.m_model.process(t_x, t_t)
        x1  = self.m_errorGenerator[0].addError(t_x + dx1 * self.m_dt, t_t)
        dx2 = self.m_model.process(x1, t_t+self.m_dt)
        return self.m_errorGenerator[1].addError(t_x + ( dx1 + dx2 ) * self.m_dt / 2.0, t_t)

#__________________________________________________

class StochasticKPIntegrator(AbstractKPIntegrator, AbstractStochasticProcess):

    #_________________________

    def __init__(self, t_dt = 0.01, t_model = DeterministicLorenz63Model(), t_eg = IndependantGaussianRNG()):
        AbstractKPIntegrator.__init__(self, t_dt, t_model)
        AbstractStochasticProcess.__init__(self, t_eg)

#__________________________________________________
 
class MultiStochasticKPIntegrator(AbstractKPIntegrator, AbstractMultiStochasticProcess):

    #_________________________

    def __init__(self, t_dt = 0.01, t_model = DeterministicLorenz63Model(), t_eg = []):
        AbstractKPIntegrator.__init__(self, t_dt, t_model)
        AbstractMultiStochasticProcess.__init__(self, t_eg)

#__________________________________________________
 
class DeterministicKPIntegrator(AbstractKPIntegrator, AbstractDeterministicProcess):

    #_________________________

    def __init__(self, t_dt = 0.01, t_model = DeterministicLorenz63Model()):
        AbstractKPIntegrator.__init__(self, t_dt, t_model)
        AbstractDeterministicProcess.__init__(self)

#__________________________________________________

