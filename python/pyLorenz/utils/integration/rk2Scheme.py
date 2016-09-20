#! /usr/bin/env python

#__________________________________________________
# pyLorenz/utils/integration/
# rk2.py
#__________________________________________________
# author        : colonel
# last modified : 2016/9/20
#__________________________________________________
#
# class to handle the integration step according to RK2 scheme
#

import numpy as np
from ..random.abstractStochasticProcess import AbstractStochasticProcess

#__________________________________________________

class RK2Scheme(AbstractStochasticProcess):

    #_________________________

    def setParameters(self, t_dt):
        # set model parameters
        self.m_dt = t_dt

    #_________________________

    def deterministicProcessForward(self, t_model, t_xn):
        # integrates xn
        dx = t_model.computeDerivate(t_xn)
        x  = t_xn + dx * self.m_dt / 2.0
        dx = t_model.computeDerivate(x)
        return t_xn + dx * self.m_dt

#__________________________________________________

