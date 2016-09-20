#! /usr/bin/env python

#__________________________________________________
# pyLorenz/utils/integration/
# rk4.py
#__________________________________________________
# author        : colonel
# last modified : 2016/9/20
#__________________________________________________
#
# class to handle the integration step according to RK4 scheme
#

import numpy as np
from ..random.abstractStochasticProcess import AbstractStochasticProcess

#__________________________________________________

class RK4Scheme(AbstractStochasticProcess):

    #_________________________

    def setParameters(self, t_dt):
        # set model parameters
        self.m_dt = t_dt

    #_________________________

    def deterministicProcessForward(self, t_model, t_xn):
        # integrates xn
        dx1 = t_model.computeDerivate(t_xn)
        x1  = t_xn + dx1 * self.m_dt / 2.0
        dx2 = t_model.computeDerivate(x1)
        x2  = t_xn + dx2 * self.m_dt / 2.0
        dx3 = t_model.computeDerivate(x2)
        x3  = t_xn + dx3 * self.m_dt
        dx4 = t_model.computeDerivate(x3)
        dx  = ( dx1 + 2.0 * dx2 + 2.0 * dx3 + dx4 ) / 6.0
        return t_xn + dx * self.m_dt

#__________________________________________________

