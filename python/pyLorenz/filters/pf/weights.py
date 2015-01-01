#! /usr/bin/env python

#__________________________________________________
# pyLorenz/filters/pf/
# weights.py
#__________________________________________________
# author        : colonel
# last modified : 2016/10/15
#__________________________________________________
#
# class to handle the weights for a particle filter
#

import numpy as np

#__________________________________________________

class Weights(object):

    #_________________________

    def __init__(self, t_Ns):
        self.m_Ns = t_Ns
        self.initialise()

    #_________________________

    def initialise(self):
        # normal weights
        self.m_w     = ( 1.0 / self.m_Ns ) * np.ones(self.m_Ns)
        # log weights
        self.m_log_w = - np.log(self.m_Ns) * np.ones(self.m_Ns)
        # statistics
        self.compute_statistics()

    #_________________________

    def compute_statistics(self):
        # Neff
        self.m_Neff  = 1.0 / ( ( self.m_w**2 ).sum() * self.m_Ns )
        # max w
        self.m_max_w = self.m_w.max()

    #_________________________

    def re_weight(self, t_log_w):
        self.m_log_w += t_log_w

    #_________________________

    def normalise(self):
        log_w_max     = self.m_log_w.max()
        self.m_log_w -= ( log_w_max + np.log(np.exp(self.m_log_w-log_w_max).sum()) )
        self.m_w      = np.exp(self.m_log_w)
        self.compute_statistics()

    #_________________________

    def record(self, t_forecast_or_analyse, t_output, t_label):
        t_output.record(t_label, t_forecast_or_analyse+'_neff', self.m_Neff)
        t_output.record(t_label, t_forecast_or_analyse+'_maxw', self.m_max_w)

#__________________________________________________

