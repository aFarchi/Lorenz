#! /usr/bin/env python

#__________________________________________________
# pyLorenz/utils/integration/
# rk2integrationstep.py
#__________________________________________________
# author        : colonel
# last modified : 2016/10/14
#__________________________________________________
#
# class to handle an integration step according to RK2 scheme
#

import numpy as np

from utils.integration.abstractintegrationstep import AbstractIntegrationStep

#__________________________________________________

class RK2InterationStep(AbstractIntegrationStep):

    #_________________________

    def __init__(self, t_dt, t_model, t_errorGenerator = None):
        AbstractIntegrationStep.__init__(self, t_dt, t_model, t_errorGenerator)
        self.setRK2IntegrationStepParameters()

    #_________________________

    def setRK2IntegrationStepParameters(self):
        # number of substeps per algorithm step
        self.m_nSStep = 2

    #_________________________

    def deterministicIntegrate(self, t_x, t_t, t_dx):
        # rk2 scheme
        # dx is a working array whith the same shape as x
        self.m_model(t_x[0], t_t, t_dx[0])
        t_x[1] = t_x[0] + 0.5 * self.m_dt * t_dx[0]
        self.m_model(t_x[1], t_t+0.5*self.m_dt, t_dx[1])
        t_x[2] = t_x[0] + self.m_dt * t_dx[1]

    #_________________________

    def stochasticIntegrate(self, t_x, t_t, t_dx):
        # rk2 scheme + errors
        # dx is a working array whith the same shape as x
        self.m_model(t_x[0], t_t, t_dx[0])
        t_x[1] = t_x[0] + 0.5 * self.m_dt * t_dx[0] + self.m_errorGenerator.drawSamples(t_t, t_x[0].shape, np.sqrt(0.5*self.m_dt))
        self.m_model(t_x[1], t_t+0.5*self.m_dt, t_dx[1])
        t_x[2] = t_x[0] + self.m_dt * t_dx[1] + self.m_errorGenerator.drawSamples(t_t+0.5*self.m_dt, t_x[0].shape, np.sqrt(self.m_dt))

#__________________________________________________

