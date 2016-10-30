#! /usr/bin/env python

#__________________________________________________
# pyLorenz/filters/pf
# abstractenkf.py
#__________________________________________________
# author        : colonel
# last modified : 2016/10/12
#__________________________________________________
#
# abstract class to handle an EnKF
#

import numpy as np

from ..abstractensemblefilter import AbstractEnsembleFilter

#__________________________________________________

class AbstractEnKF(AbstractEnsembleFilter):

    #_________________________

    def __init__(self, t_initialiser, t_integrator, t_observationOperator, t_observationTimes, t_output, t_label, t_Ns, t_outputFields, t_inflation):
        AbstractEnsembleFilter.__init__(self, t_initialiser, t_integrator, t_observationOperator, t_observationTimes, t_output, t_label, t_Ns, t_outputFields)
        self.setAbstractEnKFParameters(t_inflation)
        self.setAbstractEnKFTemporaryArrays()

    #_________________________

    def setAbstractEnKFParameters(self, t_inflation):
        # inflation
        self.m_inflation = t_inflation

    #_________________________

    def setAbstractEnKFTemporaryArrays(self):
        # allocate temporary arrays 
        self.m_Hxf = np.zeros((self.m_Ns, self.m_observationOperator.m_spaceDimension))

    #_________________________

    def computeAnalysePerformance(self, t_xt):
        AbstractEnsembleFilter.computeAnalysePerformance(self, t_xt)
        # also apply inflation around the ensemble mean
        self.m_x[self.m_integrationIndex] = self.m_estimation + self.m_inflation * ( self.m_x[self.m_integrationIndex] - self.m_estimation )

    #_________________________

    def estimate(self):
        # mean of x
        self.m_estimation = self.m_x[self.m_integrationIndex].mean(axis = -2)

#__________________________________________________

