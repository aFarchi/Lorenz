#! /usr/bin/env python

#__________________________________________________
# pyLorenz/utils/trigger/
# thresholdtrigger.py
#__________________________________________________
# author        : colonel
# last modified : 2016/11/18
#__________________________________________________
#
# trigger event if a threshold is exceeded
#

#__________________________________________________

class ThresholdTrigger(object):

    #_________________________

    def __init__(self, t_threshold):
        self.set_threshold_trigger_parameters(t_threshold)

    #_________________________

    def set_threshold_trigger_parameters(self, t_threshold):
        self.m_threshold = t_threshold

    #_________________________

    def __call__(self, t_value, *t_args, **t_kwargs):
        return ( t_value < self.m_threshold )

#__________________________________________________

