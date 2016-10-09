#! /usr/bin/env python

#__________________________________________________
# pyLorenz/utils/process/
# abstractprocess.py
#__________________________________________________
# author        : colonel
# last modified : 2016/9/21
#__________________________________________________
#
# define abstract classes for a process with or without noise
#

import numpy as np

from ..random.independantgaussianrng import IndependantGaussianRNG

#__________________________________________________

class AbstractStochasticProcess(object):

    #_________________________

    def __init__(self, t_eg = IndependantGaussianRNG()):
        self.setErrorGenerator(t_eg)

    #_________________________

    def setErrorGenerator(self, t_eg = IndependantGaussianRNG()):
        # error rng
        # it will be called at the end of method stochasticProcess()
        self.m_errorGenerator = t_eg

    #_________________________

    def stochasticProcess(self, t_x, t_t):
        return self.m_errorGenerator.addError(self.deterministicProcess(t_x, t_t), t_t)

    #_________________________

    def process(self, t_x, t_t):
        # call stochastic process and add error
        return self.stochasticProcess(t_x, t_t)

    #_________________________

    def errorCovarianceMatrix(self, t_t):
        # returns covariance matrix of the error rng
        return self.m_errorGenerator.covarianceMatrix(t_t)

    #_________________________

    def errorCovarianceMatrix_diag(self, t_t):
        # returns diagonal of the covariance matrix of the error rng
        return self.m_errorGenerator.covarianceMatrix_diag(t_t)

#__________________________________________________

class AbstractMultiStochasticProcess(AbstractStochasticProcess):

    #_________________________

    def __init__(self, t_eg = []):
        AbstractStochasticProcess.__init__(self, t_eg)

    #_________________________

    def addErrorGenerator(self, t_eg = IndependantGaussianRNG()):
        # append error rng to the list
        # they will be called after each sub-process in method multiStochasticProcess()
        self.m_errorGenerator.append(t_eg)

    #_________________________

    def process(self, t_x, t_t):
        # call multiStochasticProcess
        # note that it should be implemented
        return self.multiStochasticProcess(t_x, t_t)

#__________________________________________________

class AbstractDeterministicProcess(object):

    #_________________________

    def __init__(self):
        pass

    #_________________________

    def process(self, t_x, t_t):
        # call deterministic process
        return self.deterministicProcess(t_x, t_t)

#__________________________________________________

