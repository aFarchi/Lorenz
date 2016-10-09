#! /usr/bin/env python

#__________________________________________________
# pyLorenz/utils/trigger/
# counterorthresholdtrigger.py
#__________________________________________________
# author        : colonel
# last modified : 2016/9/29
#__________________________________________________
#
#
#

#__________________________________________________

class CounterOrThresholdTrigger(object):

    #_________________________

    def __init__(self, t_threshold, t_counterLimit):
        self.setThresholdTriggerParameters(t_threshold, t_counterLimit)

    #_________________________

    def setThresholdTriggerParameters(self, t_threshold, t_counterLimit):
        self.m_threshold    = t_threshold
        self.m_counter      = 0
        self.m_counterLimit = t_counterLimit

    #_________________________

    def trigger(self, t_value, t_nt):
        self.m_counter += 1
        if self.m_counter > self.m_counterLimit or t_value < self.m_threshold:
            self.m_counter = 0
            return True
        else:
            return False

#__________________________________________________

