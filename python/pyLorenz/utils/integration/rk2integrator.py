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

from ..process.abstractprocess import AbstractStochasticProcess 
from ..process.abstractprocess import AbstractMultiStochasticProcess
from ..process.abstractprocess import AbstractDeterministicProcess

#__________________________________________________

class AbstractRK2Integrator:

    #_________________________

    def setParameters(self, t_dt):
        # set integration parameters
        self.m_dt = t_dt

    #_________________________

    def deterministicProcess(self, t_model, t_xn):
        # integrates xn
        dx = t_model.process(t_xn)
        x  = self.potentiallyAddError(t_xn + dx * self.m_dt / 2.0)
        dx = t_model.process(x)
        return self.potentiallyAddError(t_xn + dx * self.m_dt)

#__________________________________________________

class StochasticRK2Integrator(AbstractRK2Integrator, AbstractStochasticProcess):
    pass

#__________________________________________________
 
class MultiStochasticRK2Integrator(AbstractRK2Integrator, AbstractMultiStochasticProcess):
    pass

#__________________________________________________
 
class DeterministicRK2Integrator(AbstractRK2Integrator, AbstractDeterministicProcess):
    pass

#__________________________________________________

