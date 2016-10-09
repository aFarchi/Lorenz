#! /usr/bin/env python

#__________________________________________________
# pyLorenz/filters/pf
# abstractfilter.py
#__________________________________________________
# author        : colonel
# last modified : 2016/10/9
#__________________________________________________
#
# abstract class to handle a filtering process
#

import numpy as np

from ..utils.integration.rk4integrator import DeterministicRK4Integrator
from ..observations.iobservations      import StochasticIObservations

#__________________________________________________

class AbstractFilter(object):

    #_________________________

    def __init__(self, t_integrator = DeterministicRK4Integrator(), t_obsOp = StochasticIObservations()):
        # set integrator
        self.m_integrator          = t_integrator
        # set observation operator
        self.m_observationOperator = t_obsOp

#__________________________________________________

