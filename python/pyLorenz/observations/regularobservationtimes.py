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
        self.m_observationTimes = t_dt * np.arange(t_Nt)
        # longest assimilation cycle
        self.m_longestCycle     = t_dt

    #_________________________

    def numberOfCycles(self):
        # number of assimilation cycles to perform
        return self.m_observationTimes.size 

    #_________________________

    def __iter__(self):
        # iteration: nCycle, tStart and tEnd
        yield 0, 0.0, 0.0

        for i in range(self.m_observationTimes.size-1):
            yield (i+1, self.m_observationTimes[i], self.m_observationTimes[i+1])

    #_________________________

    def longestCycle(self):
        # return longest assimilation cycle
        return self.m_longestCycle

#__________________________________________________

