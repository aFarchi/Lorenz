#! /usr/bin/env python

#__________________________________________________
# pyLorenz/utils/integration/
# kpintegrationstep.py
#__________________________________________________
# author        : colonel
# last modified : 2016/10/14
#__________________________________________________
#
# classes to handle an integration step according to KP scheme
#

import numpy as np
 
from utils.integration.abstractintegrationstep import AbstractIntegrationStep 

#__________________________________________________

class KPIntegrationStep(AbstractIntegrationStep):

    #_________________________

    def __init__(self, t_dt, t_model, t_errorGenerator = None):
        AbstractIntegrationStep.__init__(self, t_dt, t_model, t_errorGenerator)
        self.setKPIntegrationStepParameters()

    #_________________________

    def setKPIntegrationStepParameters(self):
        # number of substeps per algorithm step
        self.m_nSStep = 2

    #_________________________

    def deterministicIntegrate(self, t_x, t_t, t_dx):
        # kp scheme
        # dx is a working array whith the same shape as x
        self.m_model(t_x[0], t_t, t_dx[0])
        t_x[1] = t_x[0] + self.m_dt * t_dx[0]
        self.m_model(t_x[1], t_t+self.m_dt, t_dx[1])
        t_x[2] = t_x[0] + self.m_dt * t_dx[0:2].mean(axis = 0)

    #_________________________

    def stochasticIntegrate(self, t_x, t_t, t_dx):
        # kp scheme + random errors
        # dx is a working array whith the same shape as x
        self.m_model(t_x[0], t_t, t_dx[0])
        t_x[1] = t_x[0] + self.m_dt * t_dx[0] + self.m_errorGenerator.drawSamples(t_t, t_x[0].shape, np.sqrt(self.m_dt))
        self.m_model(t_x[1], t_t+self.m_dt, t_dx[1])
        t_x[2] = t_x[0] + self.m_dt * t_dx[0:2].mean() + self.m_errorGenerator.drawSyntheticSamples([t_t, t_t+self.m_dt], [0.5, 0.5], t_x[0].shape, np.sqrt(self.m_dt))

#__________________________________________________

