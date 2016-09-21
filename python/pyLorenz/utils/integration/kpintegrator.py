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

from ..process.abstractprocess import AbstractStochasticProcess 
from ..process.abstractprocess import AbstractMultiStochasticProcess
from ..process.abstractprocess import AbstractDeterministicProcess

#__________________________________________________

class AbstractKPIntegrator:

    #_________________________

    def setParameters(self, t_dt):
        # set integration parameters
        self.m_dt = t_dt

    #_________________________

    def deterministicProcess(self, t_model, t_xn):
        # integrates xn
        dx = t_model.process(t_xn)
        x  = self.potentiallyAddError(t_xn + dx * self.m_dt)
        dx = ( t_model.process(x) + dx ) / 2.0
        return self.potentiallyAddError(t_xn + dx * self.m_dt)

#__________________________________________________

class StochasticKPIntegrator(AbstractKPIntegrator, AbstractStochasticProcess):
    pass

#__________________________________________________
 
class MultiStochasticKPIntegrator(AbstractKPIntegrator, AbstractMultiStochasticProcess):
    pass

#__________________________________________________
 
class DeterministicKPIntegrator(AbstractKPIntegrator, AbstractDeterministicProcess):
    pass

#__________________________________________________

