#!/usr/bin/env python

#__________________________________________________
# pyLorenz/utils/minimisation/
# scipyminimiser.py
#__________________________________________________
# author        : colonel
# last modified : 2016/11/17
#__________________________________________________
#
# class to perform minimisation using scipy's minimisation
#

import numpy as np

from scipy.optimize import minimize
from scipy.optimize import minimize_scalar

#__________________________________________________

class ScipyAbstractMinimiser(object):

    #_________________________

    def __init__(self, t_scipy_minimisation, t_method, t_tol = None, t_options = None):
        self.setScipyAbstractMinimiserParameters(t_scipy_minimisation, t_method, t_tol, t_options)

    #_________________________

    def setScipyAbstractMinimiserParameters(self, t_scipy_minimisation, t_method, t_tol, t_options):
        # scipy's function
        self.m_minimise = t_scipy_minimisation
        # scipy's method
        self.m_method   = t_method
        # tolerance
        self.m_tol      = t_tol
        # options
        self.m_options  = t_options

    #_________________________

    def argmin(self, t_fun, t_x0 = None, **t_kwargs):
        # call scipy's minimize function
        # kwargs are just forwarded
        if t_x0 is None:
            res = self.m_minimise(t_fun, method = self.m_method, tol = self.m_tol, options = self.m_options, **t_kwargs)
        else:
            res = self.m_minimise(t_fun, t_x0, method = self.m_method, tol = self.m_tol, options = self.m_options, **t_kwargs)
        # return argmin
        return res.x

#__________________________________________________

class ScipyVectorMinimisation(ScipyAbstractMinimiser):

    #_________________________

    def __init__(self, t_method, t_tol = None, t_options = None):
        ScipyAbstractMinimiser.__init__(self, minimize, t_method, t_tol, t_options)

    #_________________________

    def argmin(self, t_fun, t_x0, **t_kwargs):
        # remove default None value for x0
        return ScipyAbstractMinimiser.argmin(self, t_fun, t_x0, **t_kwargs)

#__________________________________________________

class ScipyScalarMinimisation(ScipyAbstractMinimiser):

    #_________________________

    def __init__(self, t_method, t_tol = None, t_options = None):
        ScipyAbstractMinimiser.__init__(self, minimize_scalar, t_method, t_tol, t_options)

#__________________________________________________

