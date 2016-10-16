#! /usr/bin/env python

#__________________________________________________
# pyLorenz/utils/integration/
# abstractintegrationstep.py
#__________________________________________________
# author        : colonel
# last modified : 2016/10/14
#__________________________________________________
#
# base class to handle an integration step
#

#__________________________________________________

class AbstractIntegrationStep(object):

    #_________________________

    def __init__(self, t_dt, t_model, t_errorGenerator = None):
        self.setAbstractIntegrationStepParameters(t_dt, t_model, t_errorGenerator)
    
    #_________________________

    def setAbstractIntegrationStepParameters(self, t_dt, t_model, t_errorGenerator):
        # space dimension
        self.m_spaceDimension = t_model.m_spaceDimension
        # time step interval
        self.m_dt             = t_dt
        # model to integrate
        self.m_model          = t_model
        # error generator (for stochastic step)
        self.m_errorGenerator = t_errorGenerator

    #_________________________

    def deterministicIntegrate(self, t_x, t_nt, t_dx):
        # deterministic integration
        raise NotImplementedError

    #_________________________

    def stochasticIntegrate(self, t_x, t_nt, t_dx):
        # stochastic integration
        raise NotImplementedError

#__________________________________________________

