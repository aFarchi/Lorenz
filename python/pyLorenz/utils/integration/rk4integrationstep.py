#! /usr/bin/env python

#__________________________________________________
# pyLorenz/utils/integration/
# rk4integrationstep.py
#__________________________________________________
# author        : colonel
# last modified : 2016/10/14
#__________________________________________________
#
# class to handle an integration step according to RK4 scheme
#

import numpy as np

from abstractintegrationstep import AbstractIntegrationStep

#__________________________________________________

class RK4InterationStep(AbstractIntegrationStep):

    #_________________________

    def __init__(self, t_dt, t_model, t_errorGenerator = None):
        AbstractIntegrationStep.__init__(self, t_dt, t_model, t_errorGenerator)
        self.setRK4IntegrationStepParameters()

    #_________________________

    def setRK4IntegrationStepParameters(self):
        # number of substeps per algorithm step
        self.m_nSStep = 4
        # dx weights
        self.m_dxw    = np.array([1.0, 2.0, 2.0, 1.0]) / 6.0

    #_________________________

    def deterministicIntegrate(self, t_x, t_t, t_dx):
        # rk4 scheme
        # dx is a working array whith the same shape as x
        self.m_model(t_x[0], t_t, t_dx[0])
        t_x[1] = t_x[0] + 0.5 * self.m_dt * t_dx[0]
        self.m_model(t_x[1], t_t+0.5*self.m_dt, t_dx[1])
        t_x[2] = t_x[0] + 0.5 * self.m_dt * t_dx[1]
        self.m_model(t_x[2], t_t+0.5*self.m_dt, t_dx[2])
        t_x[3] = t_x[0] + self.m_dt * t_dx[2]
        self.m_model(t_x[3], t_t+self.m_dt, t_dx[3])
        t_x[4] = t_x[0] + self.m_dt * np.average(t_dx[0:4], axis = 0, weights = self.m_dxw)

    #_________________________

    def stochasticIntegrate(self, t_x, t_t, t_dx):
        # rk4 scheme + errors
        # dx is a working array whith the same shape as x
        self.m_model(t_x[0], t_t, t_dx[0])
        t_x[1] = t_x[0] + 0.5 * self.m_dt * t_dx[0] + self.m_errorGenerator.drawSamples(t_t, t_x[0].shape, np.sqrt(0.5*self.m_dt))
        self.m_model(t_x[1], t_t+0.5*self.m_dt, t_dx[1])
        t_x[2] = t_x[0] + 0.5 * self.m_dt * t_dx[1] + self.m_errorGenerator.drawSamples(t_t+0.5*self.m_dt, t_x[0].shape, np.sqrt(0.5*self.m_dt))
        self.m_model(t_x[2], t_t+0.5*self.m_dt, t_dx[2])
        t_x[3] = t_x[0] + self.m_dt * t_dx[2] + self.m_errorGenerator.drawSamples(t_t+0.5*self.m_dt, t_x[0].shape, np.sqrt(self.m_dt))
        self.m_model(t_x[3], t_t+self.m_dt, t_dx[3])
        t_x[4] = ( t_x[0] + self.m_dt * np.average(t_dx[0:4], axis = 0, weights = self.m_dxw) +
                self.m_errorGenerator.drawSyntheticSamples([t_t, t_t+0.5*self.m_dt, t_t+0.5*self.m_dt, t_t+self.m_dt], self.m_dxw, t_x[0].shape, np.sqrt(self.m_dt)) )

#__________________________________________________

