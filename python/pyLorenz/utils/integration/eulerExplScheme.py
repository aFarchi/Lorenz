#! /usr/bin/env python

#__________________________________________________
# pyLorenz/utils/integration/
# eulerExplScheme.py
#__________________________________________________
# author        : colonel
# last modified : 2016/9/20
#__________________________________________________
#
# class to handle the integration step according to euler's explicit scheme
#

import numpy as np
from ..random.abstractStochasticProcess import AbstractStochasticProcess

#__________________________________________________

class EulerExplScheme(AbstractStochasticProcess):

    #_________________________

    def setParameters(self, t_dt):
        # set model parameters
        self.m_dt = t_dt

    #_________________________

    def deterministicProcessForward(self, t_model, t_xn):
        # integrates xn
        dx = t_model.computeDerivate(t_xn)
        return t_xn + dx * self.m_dt

#__________________________________________________

