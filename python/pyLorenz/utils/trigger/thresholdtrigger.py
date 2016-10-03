#! /usr/bin/env python

#__________________________________________________
# pyLorenz/utils/trigger/
# thresholdTrigger.py
#__________________________________________________
# author        : colonel
# last modified : 2016/9/29
#__________________________________________________
#
#
#

#__________________________________________________

class ThresholdTrigger(object):

    #_________________________

    def __init__(self, t_threshold):
        self.setThresholdTriggerParameters(t_threshold)

    #_________________________

    def setThresholdTriggerParameters(self, t_threshold):
        self.m_threshold = t_threshold

    #_________________________

    def trigger(self, t_value, t_nt):
        return ( t_value < self.m_threshold )

#__________________________________________________

