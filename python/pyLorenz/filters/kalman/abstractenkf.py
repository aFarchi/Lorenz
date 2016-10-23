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

    def __init__(self, t_label, t_integrator, t_observationOperator, t_Ns, t_covarianceInflation):
        AbstractEnsembleFilter.__init__(self, t_label, t_integrator, t_observationOperator, t_Ns)
        self.setAbstractEnKFParameters(t_covarianceInflation)
        self.setAbstractEnKFTemporaryArrays()

    #_________________________

    def setAbstractEnKFParameters(self, t_covarianceInflation):
        # covariance inflation
        self.m_covarianceInflation = t_covarianceInflation

    #_________________________

    def setAbstractEnKFTemporaryArrays(self):
        # allocate temporary arrays 
        self.m_Hxf = np.zeros((self.m_Ns, self.m_observationOperator.m_spaceDimension))

    #_________________________

    def computeAnalysePerformance(self, t_xt, t_iEnd):
        AbstractEnsembleFilter.computeAnalysePerformance(self, t_xt, t_iEnd)

        # also apply inflation around the ensemble mean
        self.m_x[t_iEnd] = self.m_estimation + self.m_covarianceInflation * ( self.m_x[t_iEnd] - self.m_estimation )

    #_________________________

    def estimate(self, t_index):
        # mean of x
        return self.m_x[t_index].mean(axis = -2)

#__________________________________________________

