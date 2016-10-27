#! /usr/bin/env python

#__________________________________________________
# launcher/utils/
# firststageoptimisation.py
#__________________________________________________
# author        : colonel
# last modified : 2016/10/26
#__________________________________________________
#
# functions related to the first stage optimisation
#

import numpy as np

from ConfigParser import NoOptionError
from cast         import *

#__________________________________________________

def optimalInterval(t_x, t_y, t_relevantThr):
    # returns the interval of x that minimise y with tolerance min(y) * relevantThr
    # y must be positive
    # relevantThr must be >= 1
    if t_y.size < 3:
        return t_x

    indices    = t_y.argsort()
    ymin       = t_y[indices[0]]
    ymin2      = t_y[indices[1]]

    relevantX  = []
    relevantX2 = []

    for i in range(t_x.size):

        if t_y[i] < ymin * t_relevantThr:
            relevantX.append(t_x[i])
        elif t_y[i] < ymin2 * t_relevantThr:
            relevantX2.append(t_x[i])

    if len(relevantX) < 2:
        relevantX.extend(relevantX2)

    return np.array(relevantX)

#__________________________________________________

def EnKFVaryingParameters():
    # return list of EnKF parameters for which multiple values can be tested
    return np.array(['inflation', 'Ns', 'integration_jitter'])

#__________________________________________________

def PFVaryingParameters():
    # return list of PF parameters for which multiple values can be tested
    return np.array(['resampling_thr', 'Ns', 'integration_jitter'])

#__________________________________________________

def varyingParameters(t_filter):
    # return list of parameters for which multiple values can be tested
    if 'en' in t_filter and 'kf' in t_filter:
        return EnKFVaryingParameters()

    elif 'pf' in t_filter:
        return PFVaryingParameters()
    
#__________________________________________________

def isFirstStageOptimisationNecessary(t_msConfig, t_filter, t_parameters):
    # check if a first stage optimisation is necessary for the given filter
    try:
        # variable
        variable = t_msConfig.get(t_filter, 'first_stage_variable')
        # check if optimisation over variable is allowed
        # if not, then disable optimisation options
        if not variable in t_parameters:
            t_msConfig.remove_option(t_filter, 'first_stage_variable')
            t_msConfig.remove_option(t_filter, 'second_stage_size')
            return False
        # values to investigate
        variable = makeNumpyArray(eval(t_msConfig.get(t_filter, variable)))
        # only perform optimisation if there are more than two values to investigate
        if variable.size > 2:
            return True
        elif variable.size == 2:
            return False
        else:
            # if there are less than two values, then disable optimisation options
            t_msConfig.remove_option(t_filter, 'first_stage_variable')
            t_msConfig.remove_option(t_filter, 'second_stage_size')
            return False

    except NoOptionError:
        # if first_stage_variable is not defined, then no optimisation is performed
        t_msConfig.remove_option(t_filter, 'second_stage_size')
        return False

#__________________________________________________

def permutationFirstStageOptimisation(t_msConfig, t_filter, t_parameters):
    # permute parameters such that optimisation is performed on the last parameter
    # note that parameters must be a numpy array of strings

    try:
        # variable
        variable = t_msConfig.get(t_filter, 'first_stage_variable')

        # index of the variable in the parameter list
        index    = ( t_parameters == variable ).argmax()

        # arrays for each parameter
        fixedParameters = []
        for parameter in t_parameters:
            fixedParameters.append(makeNumpyArray(eval(t_msConfig.get(t_filter, parameter))))

        # extract the variable from the parameters
        variable = fixedParameters.pop(index)
        # now fixedParameters contains the parameters over which no optimisation is performed
        # and the optimisation is performed over the parameter variable

        # permutation from optimisation order to 'natural' order 
        # (i.e. order in which the parameters were given)
        def permutation(*args):
            l = list(args)
            v = l.pop()
            l.insert(index, v)
            return tuple(l)

        return (variable.size>2, variable, fixedParameters, permutation)


    except NoOptionError:
        # if first_stage_variable is not defined, then no optimisation is performed
        # last parameter is arbitrary chosen as 'variable' such that the permutation is the identity
        fixedParameters = []
        for parameter in t_parameters:
            fixedParameters.append(makeNumpyArray(eval(t_msConfig.get(t_filter, parameter))))
        variable = fixedParameters.pop(index)

        def permutation(*args):
            return args

        return (False, variable, fixedParameters, permutation)

#__________________________________________________

