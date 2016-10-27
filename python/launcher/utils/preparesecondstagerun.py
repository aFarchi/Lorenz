#! /usr/bin/env python

#__________________________________________________
# launcher/utils/
# preparesecondstagerun.py
#__________________________________________________
# author        : colonel
# last modified : 2016/10/27
#__________________________________________________
#
# functions to prepare second stage run
#

import numpy as np

from bash                   import *
from cast                   import *
from firststageoptimisation import *
from simulationconfig       import *
from simulationoutput       import *
from multisimulationconfig  import *
from itertools              import product

#__________________________________________________

def prepareENKFSecondStageRun(t_msConfig, t_filter, t_observation_dt, t_observation_var, t_integration_var, t_SSConfigs):
    # build configs for second stage run of an EnKF

    #SSConfigs      = []

    # parameters allowed to vary
    parameterNames = EnKFVaryingParameters()

    # extract first stage values for the parameters
    # and the permutation from optimisation order to 'natural' order
    (optimised, variable, fixed, permutation) = permutationFirstStageOptimisation(t_msConfig, t_filter, parameterNames)

    for (fixed1, fixed2) in product(*fixed):

        # if optimisation was performed
        if optimised:

            # extract first stage average RMSE
            rmse = np.zeros(variable.size)
            for i in range(variable.size):
                (inflation, Ns, integration_jitter) = permutation(fixed1, fixed2, variable[i])
                outputSDir    = outputSubDir(t_msConfig.get('output', 'directory')+'firstStage/', t_observation_dt, t_observation_var, t_integration_var)
                filterLabel   = EnKFLabel(t_filter, Ns, inflation, integration_jitter)
                rmseFileName  = analyseRMSEFileName(outputSDir, filterLabel)
                Nt_burnout    = int(eval(t_msConfig.get('first-stage', 'Nt_burnout')))
                rmse[i]       = np.fromfile(rmseFileName)[Nt_burnout:].mean()

            # select 'optimal' interval for the variable
            #===================
            #print 'rmse=', rmse
            #===================
            rmse_limit_factor = eval(t_msConfig.get('first-stage', 'rmse_limit_factor'))
            optimalVar        = optimalInterval(variable, rmse, rmse_limit_factor)

        # if no optimisation was performed
        # everything behaves as if optimal interval were the first stage interval
        else:
            optimalVar        = variable

        try:
            # determine second stage values for the variable
            second_stage_size = int(eval(t_msConfig.get(t_filter, 'second_stage_size')))
            if second_stage_size == 1:
                ssVariable    = np.array([0.5*(optimalVar.min()+optimalVar.max())])
            else:
                ssVariable    = np.linspace(optimalVar.min(), optimalVar.max(), second_stage_size)
        except NoOptionError:
            # if second_stage_size is not defined, then just select the first stage interval
            ssVariable        = optimalVar

        # build configuration for the second stage (ie. full) run
        for var in ssVariable:

            (inflation, Ns, integration_jitter) = permutation(fixed1, fixed2, var)

            config     = buildEnKFSimulationConfig('second', t_msConfig, t_filter, t_observation_dt, t_observation_var, t_integration_var, Ns, integration_jitter, inflation)
            outputSDir = outputSubDir(t_msConfig.get('output', 'directory')+'secondStage/', t_observation_dt, t_observation_var, t_integration_var)
            configFN   = outputSDir+EnKFLabel(t_filter, Ns, inflation, integration_jitter)+'.cfg'
            configFile = open(configFN, 'w')
            config.write(configFile)
            configFile.close()

            #SSConfigs.append(configFN)
            t_SSConfigs[(t_observation_dt, t_observation_var, t_integration_var)][(t_filter, inflation, Ns, integration_jitter)] = configFN

    #return SSConfigs

#__________________________________________________

def preparePFSecondStageRun(t_msConfig, t_filter, t_observation_dt, t_observation_var, t_integration_var, t_SSConfigs):
    # build configs for second stage run of a PF

    #SSConfigs      = []

    # parameters allowed to vary
    parameterNames = PFVaryingParameters()

    # extract first stage values for the parameters
    # and the permutation from optimisation order to 'natural' order
    (optimised, variable, fixed, permutation) = permutationFirstStageOptimisation(t_msConfig, t_filter, parameterNames)

    for (fixed1, fixed2) in product(*fixed):

        # if optimisation was performed
        if optimised:

            # extract first stage average RMSE
            rmse = np.zeros(variable.size)
            for i in range(variable.size):
                (resampling_thr, Ns, integration_jitter) = permutation(fixed1, fixed2, variable[i])
                outputSDir    = outputSubDir(t_msConfig.get('output', 'directory')+'firstStage/', t_observation_dt, t_observation_var, t_integration_var)
                filterLabel   = PFLabel(t_filter, Ns, resampling_thr, integration_jitter)
                rmseFileName  = analyseRMSEFileName(outputSDir, filterLabel)
                Nt_burnout    = int(eval(t_msConfig.get('first-stage', 'Nt_burnout')))
                rmse[i]       = np.fromfile(rmseFileName)[Nt_burnout:].mean()

            # select 'optimal' interval for the variable
            #===================
            #print 'rmse=', rmse
            #===================
            rmse_limit_factor = eval(t_msConfig.get('first-stage', 'rmse_limit_factor'))
            optimalVar        = optimalInterval(variable, rmse, rmse_limit_factor)

        # if no optimisation was performed
        # everything behaves as if optimal interval were the first stage interval
        else:
            optimalVar        = variable

        try:
            # determine second stage values for the variable
            second_stage_size = int(eval(t_msConfig.get(t_filter, 'second_stage_size')))
            ssVariable        = np.linspace(optimalVar.min(), optimalVar.max(), second_stage_size+2)[1:-1]
        except NoOptionError:
            # if second_stage_size is not defined, then just select the first stage interval
            ssVariable        = optimalVar

        # build configuration for the second stage (ie. full) run
        for var in ssVariable:

            (resampling_thr, Ns, integration_jitter) = permutation(fixed1, fixed2, var)

            config     = buildPFSimulationConfig('second', t_msConfig, t_filter, t_observation_dt, t_observation_var, t_integration_var, Ns, integration_jitter, resampling_thr)
            outputSDir = outputSubDir(t_msConfig.get('output', 'directory')+'secondStage/', t_observation_dt, t_observation_var, t_integration_var)
            configFN   = outputSDir+PFLabel(t_filter, Ns, resampling_thr, integration_jitter)+'.cfg'
            configFile = open(configFN, 'w')
            config.write(configFile)
            configFile.close()

            #SSConfigs.append(configFN)
            t_SSConfigs[(t_observation_dt, t_observation_var, t_integration_var)][(t_filter, resampling_thr, Ns, integration_jitter)] = configFN

    #return SSConfigs

#__________________________________________________

def prepareFilterSecondStageRun(t_msConfig, t_filter, t_observation_dt, t_observation_var, t_integration_var, t_SSConfigs):
    # build configs for second stage run of a filter

    if 'en' in t_filter and 'kf' in t_filter:
        #return prepareENKFSecondStageRun(t_msConfig, t_filter, t_observation_dt, t_observation_var, t_integration_var)
        prepareENKFSecondStageRun(t_msConfig, t_filter, t_observation_dt, t_observation_var, t_integration_var, t_SSConfigs)

    elif 'pf' in t_filter:
        #return preparePFSecondStageRun(t_msConfig, t_filter, t_observation_dt, t_observation_var, t_integration_var)
        preparePFSecondStageRun(t_msConfig, t_filter, t_observation_dt, t_observation_var, t_integration_var, t_SSConfigs)

#__________________________________________________

def prepareSecondStageRun(t_msConfig):
    # build configs for second stage run

    #SSConfigs       = []
    SSConfigs       = {}

    observation_dt  = makeNumpyArray(eval(t_msConfig.get('observation', 'dt')))
    observation_var = makeNumpyArray(eval(t_msConfig.get('observation', 'variance')))
    integration_var = makeNumpyArray(eval(t_msConfig.get('integration', 'variance')))

    afilters        = availableFiltersFromConfig(t_msConfig)

    for (obs_dt, obs_var, int_var) in product(observation_dt, observation_var, integration_var):

        createDir(outputSubDir(t_msConfig.get('output', 'directory')+'secondStage/', obs_dt, obs_var, int_var))
        SSConfigs[(obs_dt, obs_var, int_var)] = {}

        for f in afilters:
            #SSConfigs.extend(prepareFilterSecondStageRun(t_msConfig, f, obs_dt, obs_var, int_var))
            prepareFilterSecondStageRun(t_msConfig, f, obs_dt, obs_var, int_var, SSConfigs)

    SSConfigFile = open(t_msConfig.get('output', 'directory')+'second-stage-configs.dat', 'w')
    #for ssc in SSConfigs:
        #SSConfigFile.write(ssc+'\n')
    for truthParameters in SSConfigs:
        for filterParameters in SSConfigs[truthParameters]:
            SSConfigFile.write(SSConfigs[truthParameters][filterParameters]+'\n')
    SSConfigFile.close()
    return SSConfigs

#__________________________________________________

