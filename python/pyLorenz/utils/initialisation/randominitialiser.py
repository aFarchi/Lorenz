#! /usr/bin/env python

#__________________________________________________
# pyLorenz/utils/initialisation/
# randominitialiser.py
#__________________________________________________
# author        : colonel
# last modified : 2016/11/5
#__________________________________________________
#
# class to handle a random initialiser around the truth
#

#__________________________________________________

class RandomInitialiser(object):

    #_________________________

    def __init__(self, t_mean, t_errorGenerator):
        self.setRandomInitialiserParameters(t_mean, t_errorGenerator)

    #_________________________

    def setRandomInitialiserParameters(self, t_mean, t_errorGenerator):
        # mean
        self.m_mean          = t_mean
        # error generator
        self.m_errorGenerator = t_errorGenerator

    #_________________________

    def initialise(self, t_x):
        # return mean + error
        t_x[:] = self.m_mean + self.m_errorGenerator.drawSamples(0, t_x.shape)

#__________________________________________________

