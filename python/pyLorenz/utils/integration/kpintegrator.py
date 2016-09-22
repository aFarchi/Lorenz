#! /usr/bin/env python

#__________________________________________________
# pyLorenz/utils/integration/
# kpintegrator.py
#__________________________________________________
# author        : colonel
# last modified : 2016/9/21
#__________________________________________________
#
# classes to handle the integration step according to KP scheme
#

import numpy as np

from abstractintegrator              import AbstractStochasticIntegrator
from abstractintegrator              import AbstractMultiStochasticIntegrator
from abstractintegrator              import AbstractDeterministicIntegrator 
from ..random.independantgaussianrng import IndependantGaussianRNG
from ...model.lorenz63               import DeterministicLorenz63Model

#__________________________________________________

class AbstractKPIntegrator(object):

    #_________________________

    def __init__(self):
        self.m_deterministicIntegratorClass = DeterministicKPIntegrator

    #_________________________

    def deterministicProcess(self, t_xn, t_nt):
        # integrates xn
        dx = self.m_model.process(t_xn, t_nt*self.m_dt)
        x  = self.potentiallyAddError(t_xn + dx * self.m_dt)
        dx = ( self.m_model.process(x, (t_nt+1.0)*self.m_dt) + dx ) / 2.0
        return self.potentiallyAddError(t_xn + dx * self.m_dt)

#__________________________________________________

class StochasticKPIntegrator(AbstractKPIntegrator, AbstractStochasticIntegrator):

    #_________________________

    def __init__(self, t_eg = IndependantGaussianRNG(), t_dt = 0.01, t_model = DeterministicLorenz63Model()):
        AbstractKPIntegrator.__init__(self)
        AbstractStochasticIntegrator.__init__(self, t_eg, t_dt, t_model)

#__________________________________________________
 
class MultiStochasticKPIntegrator(AbstractKPIntegrator, AbstractMultiStochasticIntegrator):

    #_________________________

    def __init__(self, t_eg = IndependantGaussianRNG(), t_dt = 0.01, t_model = DeterministicLorenz63Model()):
        AbstractKPIntegrator.__init__(self)
        AbstractMultiStochasticIntegrator.__init__(self, t_eg, t_dt, t_model)

#__________________________________________________
 
class DeterministicKPIntegrator(AbstractKPIntegrator, AbstractDeterministicIntegrator):

    #_________________________

    def __init__(self, t_dt = 0.01, t_model = DeterministicLorenz63Model()):
        AbstractKPIntegrator.__init__(self)
        AbstractDeterministicIntegrator.__init__(self, t_dt, t_model)

#__________________________________________________

