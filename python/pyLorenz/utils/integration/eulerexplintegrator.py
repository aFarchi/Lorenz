#! /usr/bin/env python

#__________________________________________________
# pyLorenz/utils/integration/
# eulerexplintegrator.py
#__________________________________________________
# author        : colonel
# last modified : 2016/10/9
#__________________________________________________
#
# classes to handle the integration step according to euler's explicit scheme
#

import numpy as np

from abstractintegrator              import AbstractIntegrator
from ..random.independantgaussianrng import IndependantGaussianRNG
from ...model.lorenz63               import DeterministicLorenz63Model
from ..process.abstractprocess       import AbstractStochasticProcess
from ..process.abstractprocess       import AbstractMultiStochasticProcess
from ..process.abstractprocess       import AbstractDeterministicProcess

#__________________________________________________

class AbstractEulerExplIntegrator(AbstractIntegrator):

    #_________________________

    def __init__(self, t_dt = 0.01, t_model = DeterministicLorenz63Model()):
        AbstractIntegrator.__init__(self, t_dt, t_model)
    
    #_________________________

    def deterministicProcess(self, t_x, t_t):
        # integrates x according to the model
        # no subprocess here
        # hence no multistochastic integrator
        return t_x + self.m_dt * self.m_model.process(t_x, t_t) # + potential errors

#__________________________________________________

class StochasticEulerExplIntegrator(AbstractEulerExplIntegrator, AbstractStochasticProcess):

    #_________________________

    def __init__(self, t_dt = 0.01, t_model = DeterministicLorenz63Model(), t_eg = IndependantGaussianRNG()):
        AbstractEulerExplIntegrator.__init__(self, t_dt, t_model)
        AbstractStochasticProcess.__init__(self, t_eg)

#__________________________________________________

class DeterministicEulerExplIntegrator(AbstractEulerExplIntegrator, AbstractDeterministicProcess):

    #_________________________

    def __init__(self, t_dt = 0.01, t_model = DeterministicLorenz63Model()):
        AbstractEulerExplIntegrator.__init__(self, t_dt, t_model)
        AbstractDeterministicProcess.__init__(self)

#__________________________________________________

