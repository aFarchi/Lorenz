#! /usr/bin/env python

#__________________________________________________
# pyLorenz/observations/
# regularobservationtimes.py
#__________________________________________________
# author        : colonel
# last modified : 2016/10/30
#__________________________________________________
#
# class to handle regular observation times
#

import numpy as np

#__________________________________________________

class RegularObservationTimes(object):

    #_________________________

    def __init__(self, t_dt, t_Nt):
        self.setRegularObservationTimesParameters(t_dt, t_Nt)

    #_________________________

    def setRegularObservationTimesParameters(self, t_dt, t_Nt):
        # observation times
        self.m_observationTimes = np.insert(t_dt*np.arange(t_Nt), 0, 0.0)
        # longest assimilation cycle
        self.m_longestCycle     = t_dt

    #_________________________

    def numberOfCycles(self):
        # number of assimilation cycles to perform
        return self.m_observationTimes.size - 1

    #_________________________

    def cycleTimes(self, t_index):
        # return tStart, tEnd for the index-th cycle
        return (self.m_observationTimes[t_index], self.m_observationTimes[t_index+1])

    #_________________________

    def longestCycle(self):
        # return longest assimilation cycle
        return self.m_longestCycle

    #_________________________

    def observationTimes(self):
        # return all observation times
        return self.m_observationTimes[1:]

#__________________________________________________

