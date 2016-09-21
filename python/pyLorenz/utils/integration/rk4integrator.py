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

from ..process.abstractprocess import AbstractStochasticProcess 
from ..process.abstractprocess import AbstractMultiStochasticProcess
from ..process.abstractprocess import AbstractDeterministicProcess

#__________________________________________________

class AbstractRK4Integrator:

    #_________________________

    def setParameters(self, t_dt):
        # set integration parameters
        self.m_dt = t_dt

    #_________________________

    def deterministicProcess(self, t_model, t_xn):
        # integrates xn
        dx1 = t_model.process(t_xn)
        x1  = self.potentiallyAddError(t_xn + dx1 * self.m_dt / 2.0)
        dx2 = t_model.process(x1)
        x2  = self.potentiallyAddError(t_xn + dx2 * self.m_dt / 2.0)
        dx3 = t_model.process(x2)
        x3  = self.potentiallyAddError(t_xn + dx3 * self.m_dt)
        dx4 = t_model.process(x3)
        dx  = ( dx1 + 2.0 * dx2 + 2.0 * dx3 + dx4 ) / 6.0
        return self.potentiallyAddError(t_xn + dx * self.m_dt)

#__________________________________________________

class StochasticRK4Integrator(AbstractRK4Integrator, AbstractStochasticProcess):
    pass

#__________________________________________________
 
class MultiStochasticRK4Integrator(AbstractRK4Integrator, AbstractMultiStochasticProcess):
    pass

#__________________________________________________
 
class DeterministicRK4Integrator(AbstractRK4Integrator, AbstractDeterministicProcess):
    pass

#__________________________________________________

