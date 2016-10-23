#! /usr/bin/env python

#__________________________________________________
# pyLorenz/utils/initialisation/
# randominitialiser.py
#__________________________________________________
# author        : colonel
# last modified : 2016/10/13
#__________________________________________________
#
# class to handle a random initialiser around the truth
#

#__________________________________________________

class RandomInitialiser(object):

    #_________________________

    def __init__(self, t_truth, t_errorGenerator):
        self.setRandomInitialiserParameters(t_truth, t_errorGenerator)

    #_________________________

    def setRandomInitialiserParameters(self, t_truth, t_errorGenerator):
        # truth
        self.m_truth          = t_truth
        # error generator
        self.m_errorGenerator = t_errorGenerator

    #_________________________

    def initialiseTruth(self, t_xt):
        # return the truth
        t_xt[:] = self.m_truth

    #_________________________

    def initialiseSamples(self, t_x):
        # return the truth + some error
        t_x[:]  = self.m_truth + self.m_errorGenerator.drawSamples(0, t_x.shape)

#__________________________________________________

