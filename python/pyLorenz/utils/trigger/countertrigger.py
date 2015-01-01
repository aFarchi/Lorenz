#! /usr/bin/env python

#__________________________________________________
# pyLorenz/utils/trigger/
# countertrigger.py
#__________________________________________________
# author        : colonel
# last modified : 2016/11/18
#__________________________________________________
#
# trigger event if a counter is exceeded
#

#__________________________________________________

class CounterTrigger(object):

    #_________________________

    def __init__(self, t_counter_limit):
        self.set_counter_trigger_parameters(t_counter_limit)

    #_________________________

    def set_counter_trigger_parameters(self, t_counter_limit):
        self.m_counter       = 0
        self.m_counter_limit = t_counter_limit

    #_________________________

    def __call__(self, *t_args, **t_kwargs):
        self.m_counter += 1
        if self.m_counter > self.m_counter_limit:
            self.m_counter = 0
            return True
        else:
            return False

#__________________________________________________

