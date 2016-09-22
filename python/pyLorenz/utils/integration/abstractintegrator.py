#! /usr/bin/env python

#__________________________________________________
# pyLorenz/utils/integration/
# abstractintegrator.py
#__________________________________________________
# author        : colonel
# last modified : 2016/9/21
#__________________________________________________
#
# classes to handle integration processes
#

import numpy as np

from ..process.abstractprocess       import AbstractStochasticProcess
from ..process.abstractprocess       import AbstractMultiStochasticProcess
from ..process.abstractprocess       import AbstractDeterministicProcess
from ..random.independantgaussianrng import IndependantGaussianRNG
from ...model.lorenz63               import DeterministicLorenz63Model

#__________________________________________________

class AbstractIntegrator(object):

    #_________________________

    def __init__(self, t_dt = 0.01, t_model = DeterministicLorenz63Model()):
        self.setIntegratorParameters(t_dt, t_model)

    #_________________________

    def setIntegratorParameters(self, t_dt = 0.01, t_model = DeterministicLorenz63Model()):
        # set integration parameters
        self.m_dt    = t_dt
        self.m_model = t_model

#__________________________________________________

class AbstractStochasticIntegrator(AbstractStochasticProcess, AbstractIntegrator):

    #_________________________

    def __init__(self, t_eg = IndependantGaussianRNG(), t_dt = 0.01, t_model = DeterministicLorenz63Model()):
        AbstractStochasticProcess.__init__(self, t_eg)
        AbstractIntegrator.__init__(self, t_dt, t_model)

    #_________________________

    def deterministicIntegrator(self):
        return self.m_deterministicIntegratorClass(self.m_dt, self.m_model.deterministicModel())

#__________________________________________________

class AbstractMultiStochasticIntegrator(AbstractMultiStochasticProcess, AbstractStochasticIntegrator):

    #_________________________

    def __init__(self, t_eg = IndependantGaussianRNG(), t_dt = 0.01, t_model = DeterministicLorenz63Model()):
        AbstractMultiStochasticProcess.__init__(self, t_eg)
        AbstractStochasticIntegrator.__init__(self, t_eg, t_dt, t_model)

#__________________________________________________

class AbstractDeterministicIntegrator(AbstractDeterministicProcess, AbstractIntegrator):

    #_________________________

    def __init__(self, t_dt = 0.01, t_model = DeterministicLorenz63Model()):
        AbstractDeterministicProcess.__init__(self)
        AbstractIntegrator.__init__(self, t_dt, t_model)

    #_________________________

    def deterministicIntegrator(self):
        return self

#__________________________________________________

