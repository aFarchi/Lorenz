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

from itertools import chain

#__________________________________________________

class Simulation(object):

    #_________________________

    def __init__(self, t_truth, t_filters, t_output, t_observationTimes):
        self.setSimulationParameters(t_truth, t_filters, t_output, t_observationTimes)

    #_________________________

    def setSimulationParameters(self, t_truth, t_filters, t_output, t_observationTimes):
        # truth
        self.m_truth            = t_truth
        # filter
        self.m_filters          = t_filters
        # output
        self.m_output           = t_output
        # observation times
        self.m_observationTimes = t_observationTimes
        # crashed filters
        self.m_crashed          = []

    #_________________________

    def initialise(self):
        # initialisation
        self.m_output.initialise()
        self.m_truth.initialise()
        for filter in self.m_filters:
            filter.initialise()

    #_________________________

    def forecast(self, t_tStart, t_tEnd):
        # integrate and observe truth
        self.m_truth.forecast(t_tStart, t_tEnd)
        for filter in self.m_filters:
            try:
                filter.forecast(t_tStart, t_tEnd, self.m_truth.observation())
            except Exception:
                self.m_filters.remove(filter)
                self.m_crashed.append(filter)
            filter.computeForecastPerformance(self.m_truth.truth())
        # record truth and observations
        self.m_truth.record()
        # record filter forecast
        for filter in self.m_filters:
            filter.recordForecast()

    #_________________________

    def analyse(self, t_tEnd):
        for filter in self.m_filters:
            # analyse observation
            try:
                filter.analyse(t_tEnd, self.m_truth.observation())
            except Exception:
                self.m_filters.remove(filter)
                self.m_crashed.append(filter)
            filter.computeAnalysePerformance(self.m_truth.truth())
            # record filter analyse
            filter.recordAnalyse()

    #_________________________

    def permute(self):
        # permute arrays to prepare next cycle
        self.m_truth.permute()
        for filter in self.m_filters:
            filter.permute()

    #_________________________

    def finalise(self):
        # finalise simulation
        self.m_output.finalise(self.m_crashed)
        self.m_truth.finalise()
        for filter in chain(self.m_filters, self.m_crashed):
            filter.finalise()

    #_________________________

    def run(self):

        # run simulation
        self.m_output.start()
        # initialisation
        self.initialise()

        # cycles
        for (nCycle, tStart, tEnd) in self.m_observationTimes:

            if not self.m_filters:
                break

            self.m_output.cycle(nCycle, self.m_observationTimes.numberOfCycles())

            self.forecast(tStart, tEnd)
            self.analyse(tEnd)
            self.permute()

            self.m_output.write()

        # finalise
        self.finalise()

#__________________________________________________

