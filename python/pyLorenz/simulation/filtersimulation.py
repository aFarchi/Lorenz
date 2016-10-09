#! /usr/bin/env python

#__________________________________________________
# pyLorenz/simulation/
# filtersimulation.py
#__________________________________________________
# author        : colonel
# last modified : 2016/9/21
#__________________________________________________
#
# classes to handle a basic simulation of any model
# with a filtering process
#

import numpy as np

from basicsimulation                                       import BasicSimulation
from ..utils.integration.rk4integrator                     import DeterministicRK4Integrator
from ..utils.initialisation.gaussianindependantinitialiser import GaussianIndependantInitialiser
from ..utils.output.basicoutputprinter                     import BasicOutputPrinter
from ..filters.kalman.stochasticenkf                       import StochasticEnKF
from ..observations.iobservations                          import StochasticIObservations

default = object()

#__________________________________________________

class FilterSimulation(BasicSimulation):

    #_________________________

    def __init__(self, t_Nt = 1000, t_integrator = DeterministicRK4Integrator(), 
            t_initialiser = GaussianIndependantInitialiser(), t_outputPrinter = BasicOutputPrinter(), t_ntObs = default,
            t_filter = StochasticEnKF(), t_obsOp = StochasticIObservations()):
        BasicSimulation.__init__(self, t_Nt, t_integrator, t_initialiser, t_outputPrinter)
        self.setFilterSimulationParameters(t_ntObs, t_filter, t_obsOp)

    #_________________________

    def setFilterSimulationParameters(self, t_ntObs = default, t_filter = StochasticEnKF(), t_obsOp = StochasticIObservations()):
        # set observation times
        if t_ntObs is default:
            self.m_ntObs = np.arange(self.m_Nt)
        else:
            self.m_ntObs = t_ntObs
        if not self.m_ntObs[-1] == self.m_Nt - 1: # always observe last time step
            l = self.m_ntObs.tolist()
            l.append(self.m_Nt-1)
            self.m_ntObs = np.array(l)
        # set filter
        self.m_filter = t_filter
        # set observation operator
        self.m_observationOperator = t_obsOp

    #_________________________

    def initialise(self):
        BasicSimulation.initialise(self)
        # arrays for tracking
        self.m_xo_record = np.zeros((self.m_Nt, self.m_observationOperator.m_spaceDimension))
        # initialise the filter
        self.m_filter.initialise(self.m_initialiser, self.m_Nt)

    #_________________________

    def analyseCycle(self, t_ntStart, t_ntEnd):
        if t_ntEnd < t_ntStart:
            return
        # perform an algorithm step
        # t_ntStart is the current time step
        # t_ntEnd is the next time step where a measurement is available
        if t_ntEnd > t_ntStart:
            self.m_outputPrinter.printStep(t_ntStart, self)

        # apply time step to the truth and record it
        for nt in np.arange(t_ntEnd-t_ntStart) + t_ntStart:
            self.m_xt                 = self.m_integrator.process(self.m_xt, nt)
            self.m_xt_record[nt+1, :] = self.m_xt[:]

        # observe the truth at time step t_ntEnd and record it
        observation                  = self.m_observationOperator.process(self.m_xt, t_ntEnd)
        self.m_xo_record[t_ntEnd, :] = observation[:]

        # forecast until time step t_ntEnd
        # pententially making use of the observation
        # (e.g. for sampling according to a proposal)
        if t_ntEnd > t_ntStart:
            self.m_filter.forecast(t_ntStart, t_ntEnd, observation)

        # Analyse observation
        self.m_filter.analyse(t_ntEnd, observation)

    #_________________________

    def run(self):
        # run function
        self.m_outputPrinter.printStart(self)

        self.initialise()
        self.analyseCycle(0, self.m_ntObs[0])
        for i in np.arange(self.m_ntObs.size-1):
            self.analyseCycle(self.m_ntObs[i], self.m_ntObs[i+1])

        self.m_outputPrinter.printEnd(self)

    #_________________________

    def observedSteps(self):
        #-------------------
        # TODO: improve this
        #-------------------
        return 1.0 * self.m_ntObs

    #_________________________

    def recordToFile(self, t_outputDir = './', t_filterPrefix = 'kalman'):
        BasicSimulation.recordToFile(self, t_outputDir)
        self.m_xo_record.tofile(t_outputDir+'xo_record.bin')
        self.observedSteps().tofile(t_outputDir+'nt_obs.bin')
        self.m_filter.recordToFile(t_outputDir, t_filterPrefix)

#__________________________________________________

