#! /usr/bin/env python

#__________________________________________________
# pyLorenz/observations/
# iobservations.py
#__________________________________________________
# author        : colonel
# last modified : 2016/9/20
#__________________________________________________
#
# class to handle an observation operator that is the identity
#

import numpy as np

from ..utils.process.abstractprocess import AbstractStochasticProcess

#__________________________________________________

class StochasticIObservations(AbstractStochasticProcess):

    #_________________________

    def deterministicProcess(self, t_x):
        # just observe everything with the identity operator
        return t_x

    #_________________________

    def observationPDF(self, t_obs, t_x, t_inflation = 1.0):
        # observation pdf at obs - H(x)
        # error variance is inflated by factor t_inflation
        shape = t_x.shape
        if len(shape) == 1:
            return self.m_errorGenerator.pdf(t_obs-t_x, t_inflation)
        else:
            return self.m_errorGenerator.pdf(np.tile(t_obs, (shape[0], 1))-t_x, t_inflation)

#__________________________________________________

