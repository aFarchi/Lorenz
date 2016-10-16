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

    def __init__(self, t_integrator, t_observationOperator, t_Ns, t_covarianceInflation):
        AbstractEnsembleFilter.__init__(self, t_integrator, t_observationOperator, t_Ns)
        self.setAbstractEnKFParameters(t_covarianceInflation)

    #_________________________

    def setAbstractEnKFParameters(self, t_covarianceInflation):
        # covariance inflation
        self.m_covarianceInflation = t_covarianceInflation

    #_________________________

    def computeAnalysePerformance(self, t_xt, t_iEnd, t_index):
        # record estimation and compute performance between tStart (excluded) and tEnd (included)
        # also apply inflation 

        AbstractEnsembleFilter.computeAnalysePerformance(self, t_xt, t_iEnd, t_index)
        # apply inflation around the ensemble mean
        self.m_x[t_iEnd] = self.m_estimate[t_index] + self.m_covarianceInflation * ( self.m_x[t_iEnd] - self.m_estimate[t_index] )

    #_________________________

    def estimate(self, t_index):
        # mean of x
        return self.m_x[t_index].mean(axis = -2)

#__________________________________________________

