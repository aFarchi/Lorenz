#! /usr/bin/env python

#__________________________________________________
# pyLorenz/model/
# nomodel.py
#__________________________________________________
# author        : colonel
# last modified : 2016/11/10
#__________________________________________________
#
# no model
#

import numpy as np

#__________________________________________________

class NoModel(object):

    #_________________________

    def __init__(self, t_spaceDimension):
        self.setNoModelParameters(t_spaceDimension)

    #_________________________

    def setNoModelParameters(self, t_spaceDimension):
        # state space dimension
        self.m_spaceDimension = t_spaceDimension

    #_________________________

    def __call__(self, t_x, t_t, t_dx):
        # dx = 0
        t_dx[..., :] = 0.0

#__________________________________________________

