#! /usr/bin/env python

#__________________________________________________
# pyLorenz/utils/integration/
# eulerexplintegrationstep.py
#__________________________________________________
# author        : colonel
# last modified : 2016/10/14
#__________________________________________________
#
# class to handle an integration step according to Euler's explicit scheme
#

import numpy as np

from utils.integration.abstractintegrationstep import AbstractIntegrationStep

#__________________________________________________

class EulerExplIntegrationStep(AbstractIntegrationStep):

    #_________________________

    def __init__(self, t_dt, t_model, t_errorGenerator = None):
        AbstractIntegrationStep.__init__(self, t_dt, t_model, t_errorGenerator)
        self.setEulerExplIntegrationStepParameters()
    
    #_________________________

    def setEulerExplIntegrationStepParameters(self):
        # number of substeps per algorithm step
        self.m_nSStep = 1

    #_________________________

    def deterministicIntegrate(self, t_x, t_t, t_dx):
        # euler explicit scheme
        # dx is a working array whith the same shape as x
        self.m_model(t_x[0], t_t, t_dx[0])
        t_x[1] = t_x[0] + self.m_dt * t_dx[0]

    #_________________________

    def stochasticIntegrate(self, t_x, t_t, t_dx):
        # euler explicit scheme + random errors
        # dx is a working array whith the same shape as x
        self.m_model(t_x[0], t_t, t_dx[0])
        t_x[1] = t_x[0] + self.m_dt * t_dx[0] + self.m_errorGenerator.drawSamples(t_t, t_x[0].shape, np.sqrt(self.m_dt))

#__________________________________________________

