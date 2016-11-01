#! /usr/bin/env python

#__________________________________________________
# pyLorenz/simulation/
# simulation.py
#__________________________________________________
# author        : colonel
# last modified : 2016/10/30
#__________________________________________________
#
# classes to handle the simulation of a model
#

import random 

#__________________________________________________

class Simulation(object):

    #_________________________

    def __init__(self, t_truth, t_filter, t_output, t_observationTimes):
        self.setSimulationParameters(t_truth, t_filter, t_output, t_observationTimes)

    #_________________________

    def setSimulationParameters(self, t_truth, t_filter, t_output, t_observationTimes):
        # truth
        self.m_truth            = t_truth
        # filter
        self.m_filter           = t_filter
        # output
        self.m_output           = t_output
        # observation times
        self.m_observationTimes = t_observationTimes

    #_________________________

    def initialise(self):
        # initialisation
        self.m_output.initialise()
        self.m_truth.initialise()
        self.m_filter.initialise()

    #_________________________

    def forecast(self, t_tStart, t_tEnd):
        # integrate and observe truth
        self.m_truth.forecast(t_tStart, t_tEnd)
        self.m_filter.forecast(t_tStart, t_tEnd, self.m_truth.observation())
        self.m_filter.computeForecastPerformance(self.m_truth.truth())
        # record truth and observations
        self.m_truth.record()
        # record filter forecast
        self.m_filter.recordForecast()

    #_________________________

    def analyse(self, t_tEnd):
        # analyse observation
        self.m_filter.analyse(t_tEnd, self.m_truth.observation())
        self.m_filter.computeAnalysePerformance(self.m_truth.truth())
        # record filter analyse
        self.m_filter.recordAnalyse()

    #_________________________

    def permute(self):
        # permute arrays to prepare next cycle
        self.m_truth.permute()
        self.m_filter.permute()

    #_________________________

    def run(self):

        # run simulation
        self.m_output.start()
        # initialisation
        self.initialise()

        # cycles
        for nCycle in range(self.m_observationTimes.numberOfCycles()):
            (tStart, tEnd) = self.m_observationTimes.cycleTimes(nCycle)

            self.m_output.cycle(nCycle, self.m_observationTimes.numberOfCycles())

            self.forecast(tStart, tEnd)
            self.analyse(tEnd)
            self.permute()

            self.m_output.write()

        # finalise
        self.m_output.finalise(self.m_observationTimes)

#__________________________________________________

