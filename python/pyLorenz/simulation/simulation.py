#! /usr/bin/env python

#__________________________________________________
# pyLorenz/simulation/
# simulation.py
#__________________________________________________
# author        : colonel
# last modified : 2016/10/23
#__________________________________________________
#
# classes to handle the simulation of a model
#

import numpy as np
from itertools import chain
import numpy.random as rnd

#__________________________________________________

class Simulation(object):

    #_________________________

    def __init__(self, t_initialiser, t_integrator, t_observationOperator, t_observationTimes, t_output, t_randomSeed):
        self.setSimulationParameters(t_initialiser, t_integrator, t_observationOperator, t_observationTimes, t_output, t_randomSeed)

    #_________________________

    def setSimulationParameters(self, t_initialiser, t_integrator, t_observationOperator, t_observationTimes, t_output, t_randomSeed):
        # initialiser
        self.m_initialiser         = t_initialiser
        # integration
        self.m_integrator          = t_integrator
        # observation
        self.m_observationOperator = t_observationOperator
        # observation times
        self.m_observationTimes    = t_observationTimes
        # simulation output
        self.m_output              = t_output
        # random seed
        self.m_randomSeed          = t_randomSeed

        # filters
        self.m_filters             = []

        # space dimension
        self.m_spaceDimension      = t_integrator.m_spaceDimension

        # size for arrays
        (self.m_sx, self.m_sdx)    = t_integrator.iArrayMaxSize(t_observationTimes)

    #_________________________

    def addFilter(self, t_filter):
        # add filter to the list of filters
        self.m_filters.append(t_filter)

    #_________________________

    def initialise(self):
        rnd.seed(self.m_randomSeed)
        # truth
        self.m_xt    = np.zeros((self.m_sx, self.m_spaceDimension))
        self.m_dxt   = np.zeros((self.m_sdx, self.m_spaceDimension))
        self.m_initialiser.initialiseTruth(self.m_xt[0])

        # observation
        self.m_observation = np.zeros(self.m_observationOperator.m_spaceDimension)

        # initialise the filter
        for f in self.m_filters:
            f.initialise(self.m_initialiser, self.m_sx, self.m_sdx)

    #_________________________

    def forecast(self, t_tStart, t_tEnd, t_iEnd):
        # integrate and observe truth
        self.m_integrator.integrate(self.m_xt, t_tStart, t_tEnd, self.m_dxt)
        self.m_observationOperator.observe(self.m_xt[t_iEnd], t_tEnd, self.m_observation)

        for f in self.m_filters:
            # forecast until ntEnd pententially making use of the observation
            f.forecast(t_tStart, t_tEnd, self.m_observation)
            f.computeForecastPerformance(self.m_xt[t_iEnd], t_iEnd)

    #_________________________

    def analyse(self, t_tEnd, t_iEnd):
        # analyse observation
        for f in self.m_filters:
            f.analyse(t_iEnd, t_tEnd, self.m_observation)
            f.computeAnalysePerformance(self.m_xt[t_iEnd], t_iEnd)

    #_________________________

    def permute(self, t_iEnd):
        # permute arrays to prepare next cycle
        self.m_xt[0] = self.m_xt[t_iEnd]
        for f in self.m_filters:
            f.permute(t_iEnd)

    #_________________________

    def writeForecast(self, t_iEnd):
        # write truth and observation
        self.m_output.writeTruth(self.m_xt[t_iEnd], self.m_observation)

        # write filters forecast
        for f in self.m_filters:
            f.writeForecast(self.m_output, t_iEnd)

    #_________________________

    def writeAnalyse(self, t_iEnd):
        # write filters analyse
        for f in self.m_filters:
            f.writeAnalyse(self.m_output, t_iEnd)

    #_________________________

    def run(self):
        self.m_output.start(self.m_spaceDimension, self.m_observationOperator.m_spaceDimension, self.m_filters)

        # initialisation
        self.m_output.initialise()
        self.initialise()

        # cycles
        for (tStart, tEnd, index) in zip(chain([0.0], self.m_observationTimes[:-1]), self.m_observationTimes, range(self.m_observationTimes.size)):
            self.m_output.cycle(index, self.m_observationTimes.size)
            iEnd = self.m_integrator.indexTEnd(tStart, tEnd)

            self.forecast(tStart, tEnd, iEnd)
            self.writeForecast(iEnd)
            self.analyse(tEnd, iEnd)
            self.writeAnalyse(iEnd)
            self.permute(iEnd)

            self.m_output.write(self.m_filters)

        # finalise
        self.m_output.finalise(self.m_filters, self.m_observationTimes)

#__________________________________________________

