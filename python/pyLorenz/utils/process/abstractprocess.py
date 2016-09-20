#! /usr/bin/env python

#__________________________________________________
# pyLorenz/utils/process/
# abstractprocess.py
#__________________________________________________
# author        : colonel
# last modified : 2016/9/20
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

    def process(self, *args, **kwargs):
        # call deterministic process and add error
        return self.m_errorGenerator.addError(self.deterministicProcess(*args, **kwargs))

#__________________________________________________

class AbstractDeterministicProcess:

    #_________________________

    def process(self, *args, **kwargs):
        # call deterministic process
        return self.deterministicProcess(*args, **kwargs)

#__________________________________________________

