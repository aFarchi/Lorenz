#! /usr/bin/env python

#__________________________________________________
# pyLorenz/utils/trigger/
# counterorthresholdtrigger.py
#__________________________________________________
# author        : colonel
# last modified : 2016/11/18
#__________________________________________________
#
# trigger event if a threshold is exceeded or if a counter is exceeded
#

#__________________________________________________

class CounterOrThresholdTrigger(object):

    #_________________________

    def __init__(self, t_threshold, t_counter_limit):
        self.set_counter_or_threshold_parameters(t_threshold, t_counter_limit)

    #_________________________

    def set_counter_or_threshold_parameters(self, t_threshold, t_counter_limit):
        self.m_threshold     = t_threshold
        self.m_counter       = 0
        self.m_counter_limit = t_counter_limit

    #_________________________

    def __call__(self, t_value, *t_args, **t_kwargs):
        self.m_counter += 1
        if self.m_counter > self.m_counter_limit or t_value < self.m_threshold:
            self.m_counter = 0
            return True
        else:
            return False

#__________________________________________________

