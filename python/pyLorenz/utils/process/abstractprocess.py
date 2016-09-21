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

#__________________________________________________

class AbstractStochasticProcess:

    #_________________________

    def setErrorGenerator(self, t_eg):
        # error rng
        self.m_errorGenerator = t_eg

    #_________________________

    def potentiallyAddError(self, t_x):
        # for compatibility with multi stochastic processes
        # will be called by the deterministicProcess() method
        # it should not add error since error will be added in the end
        return t_x

    #_________________________

    def process(self, *args, **kwargs):
        # call deterministic process and add error
        return self.m_errorGenerator.addError(self.deterministicProcess(*args, **kwargs))

#__________________________________________________

class AbstractMultiStochasticProcess:

    #_________________________

    def setErrorGenerator(self, t_eg):
        # error rng
        self.m_errorGenerator = t_eg

    #_________________________

    def potentiallyAddError(self, t_x):
        # add error to vector x
        return self.m_errorGenerator.addError(t_x)

    #_________________________

    def process(self, *args, **kwargs):
        # call deterministic process
        # no need to add error here since error is added with the potentiallyAddError() method
        return self.deterministicProcess(*args, **kwargs)

#__________________________________________________

class AbstractDeterministicProcess:

    #_________________________

    def potentiallyAddError(self, t_x):
        # for compatibility with multi stochastic processes
        # will be called by the deterministicProcess() method
        # it should not add error since we are deterministic here
        return t_x

    #_________________________

    def process(self, *args, **kwargs):
        # call deterministic process
        return self.deterministicProcess(*args, **kwargs)

#__________________________________________________

