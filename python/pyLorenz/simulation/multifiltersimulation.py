#! /usr/bin/env python

#__________________________________________________
# pyLorenz/simulation/
# multifiltersimulation.py
#__________________________________________________
# author        : colonel
# last modified : 2016/10/15
#__________________________________________________
#
# classes to handle the simulation of a model
# with multiples filtering process
#

import numpy as np

from basicsimulation import BasicSimulation

#__________________________________________________

class MultiFilterSimulation(BasicSimulation):

    #_________________________

    def __init__(self, t_integrator, t_initialiser, t_outputPrinter, t_observationTimes, t_observationOperator):
        BasicSimulation.__init__(self, t_integrator, t_initialiser, t_outputPrinter, t_observationTimes, t_observationOperator)
        self.m_filters      = []
        self.m_filtersLabel = []

    #_________________________

    def addFilter(self, t_filter, t_label):
        # add filter to the list of filters
        self.m_filters.append(t_filter)
        self.m_filtersLabel.append(t_label)

    #_________________________

    def initialise(self):
        BasicSimulation.initialise(self)

        # size of the integration arrays
        (sx, sdx)    = self.m_integrator.iArrayMaxSize(self.m_observationTimes)
        # initialise the filter
        for f in self.m_filters:
            f.initialise(self.m_initialiser, self.m_observationTimes.size, sx, sdx)

    #_________________________

    def cycle(self, t_tStart, t_tEnd, t_index):
        # perform simulation from tStart to tEnd
        self.m_outputPrinter.printCycle(t_index, self.m_observationTimes.size)

        # aux index variable
        iEnd = self.m_integrator.indexTEnd(t_tStart, t_tEnd)

        # integrate truth and record it
        self.m_integrator.integrate(self.m_xt, t_tStart, t_tEnd, self.m_dxt)
        self.m_xt_record[t_index] = self.m_xt[iEnd]

        # observe the truth at time tEnd and record it
        observation               = self.m_observationOperator.observe(self.m_xt[iEnd], t_tEnd)
        self.m_xo_record[t_index] = observation

        for f in self.m_filters:
            # forecast until ntEnd pententially making use of the observation
            # (e.g. for sampling according to a proposal)
            f.forecast(t_tStart, t_tEnd, observation)
            f.computeForecastPerformance(self.m_xt[iEnd], iEnd, t_index)

            # analyse observation
            f.analyse(iEnd, t_tEnd, observation)
            f.computeAnalysePerformance(self.m_xt[iEnd], iEnd, t_index)

        # permute self.m_xt to prepare next cycle
        self.m_xt[0] = self.m_xt[iEnd]
        for f in self.m_filters:
            f.permute(iEnd)

    #_________________________

    def recordToFile(self, t_outputDir):
        BasicSimulation.recordToFile(self, t_outputDir)
        for (f, p) in zip(self.m_filters, self.m_filtersLabel):
            f.recordToFile(t_outputDir, p)

#__________________________________________________

