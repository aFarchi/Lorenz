#! /usr/bin/env python

#__________________________________________________
# pyLorenz/utils/initialisation/
# gaussianindependantinitialiser.py
#__________________________________________________
# author        : colonel
# last modified : 2016/10/9
#__________________________________________________
#
# class to handle an initialiser
#

import numpy as np

from ..random.independantgaussianrng import IndependantGaussianRNG

#__________________________________________________

class GaussianIndependantInitialiser(object):

    #_________________________

    def __init__(self, t_truth = np.zeros(0), t_eg = IndependantGaussianRNG()):
        self.setGaussianIndependantInitialiserParameters(t_truth, t_eg)

    #_________________________

    def setGaussianIndependantInitialiserParameters(self, t_truth = np.zeros(0), t_eg = IndependantGaussianRNG()):
        # truth
        self.m_truth          = t_truth
        # error generator
        self.m_errorGenerator = t_eg

    #_________________________

    def initialiseTruth(self):
        return np.copy(self.m_truth)

    #_________________________

    def initialiseSamples(self, t_Ns):
        return self.m_truth + self.m_errorGenerator.drawSamples(t_Ns, 0)

#__________________________________________________

