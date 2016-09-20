#! /usr/bin/env python

#__________________________________________________
# pyLorenz/utils/integration/
# eulerexplintegrator.py
#__________________________________________________
# author        : colonel
# last modified : 2016/9/20
#__________________________________________________
#
# classes to handle the integration step according to euler's explicit scheme
#

import numpy as np

from ..process.abstractprocess import AbstractStochasticProcess
from ..process.abstractprocess import AbstractDeterministicProcess

#__________________________________________________

class AbstractEulerExplIntegrator:

    #_________________________

    def setParameters(self, t_dt):
        # set integration parameters
        self.m_dt = t_dt

    #_________________________

    def deterministicProcess(self, t_model, t_xn):
        # integrates xn
        dx = t_model.process(t_xn)
        return t_xn + dx * self.m_dt

#__________________________________________________

class StochasticEulerExplIntegrator(AbstractEulerExplIntegrator, AbstractStochasticProcess):
    pass

#__________________________________________________

class DeterministicEulerExplIntegrator(AbstractEulerExplIntegrator, AbstractDeterministicProcess):
    pass

#__________________________________________________

