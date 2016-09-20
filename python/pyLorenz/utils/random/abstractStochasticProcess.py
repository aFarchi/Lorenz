#! /usr/bin/env python

#__________________________________________________
# pyLorenz/utils/random/
# abstractStochasticProcess.py
#__________________________________________________
# author        : colonel
# last modified : 2016/9/20
#__________________________________________________
#
# define an abstract class for a process with some noise
#

import numpy as np

#__________________________________________________

class AbstractStochasticProcess:

    #_________________________

    def setErrorGenerator(self, t_eg):
        # error rng
        self.m_errorGenerator = t_eg

    #_________________________

    def stochasticProcessForward(self, *args, **kwargs):
        # call deterministic process and add error
        kwargs['t_stochastic'] = True
        return self.m_errorGenerator.addError(self.deterministicProcessForward(*args, **kwargs))

#__________________________________________________

