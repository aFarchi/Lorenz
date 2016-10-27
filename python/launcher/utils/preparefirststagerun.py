#! /usr/bin/env python

#__________________________________________________
# launcher/utils/
# preparefirststagerun.py
#__________________________________________________
# author        : colonel
# last modified : 2016/10/26
#__________________________________________________
#
# functions to prepare first stage run
#

from bash                   import *
from cast                   import *
from firststageoptimisation import *
from simulationconfig       import *
from simulationoutput       import *
from multisimulationconfig  import *
from itertools              import product

#__________________________________________________

def prepareENKFFirstStageRun(t_msConfig, t_filter, t_observation_dt, t_observation_var, t_integration_var, t_FSConfigs):
    # build configs for first stage run of an EnKF

    # parameters allowed to vary
    parameterNames = EnKFVaryingParameters()

    # check if a first stage optimisation is necessary for filter
    if isFirstStageOptimisationNecessary(t_msConfig, t_filter, parameterNames):

        # extract first stage values for the parameters
        parameters = []
        for parameterName in parameterNames:
            parameters.append(makeNumpyArray(eval(t_msConfig.get(t_filter, parameterName))))

        # build configuration for the first stage run
        for (inflation, Ns, integration_jitter) in product(*parameters):

            config     = buildEnKFSimulationConfig('first', t_msConfig, t_filter, t_observation_dt, t_observation_var, t_integration_var, Ns, integration_jitter, inflation)
            outputSDir = outputSubDir(t_msConfig.get('output', 'directory')+'firstStage/', t_observation_dt, t_observation_var, t_integration_var)
            configFN   = outputSDir+EnKFLabel(t_filter, Ns, inflation, integration_jitter)+'.cfg'
            configFile = open(configFN, 'w')
            config.write(configFile)
            configFile.close()

            #t_FSConfigList.append(configFN)
            t_FSConfigs[(t_observation_dt, t_observation_var, t_integration_var)][(t_filter, inflation, Ns, integration_jitter)] = configFN

    #return FSConfigs

#__________________________________________________

def preparePFFirstStageRun(t_msConfig, t_filter, t_observation_dt, t_observation_var, t_integration_var, t_FSConfigs):
    # build configs for first stage run of a PF

    FSConfigs      = []

    # parameters allowed to vary
    parameterNames = PFVaryingParameters()

    # check if a first stage optimisation is necessary for filter
    if isFirstStageOptimisationNecessary(t_msConfig, t_filter, parameterNames):

        # extract first stage values for the parameters
        parameters = []
        for parameterName in parameterNames:
            parameters.append(makeNumpyArray(eval(t_msConfig.get(t_filter, parameterName))))

        # build configuration for the first stage run
        for (resampling_thr, Ns, integration_jitter) in product(*parameters):

            config     = buildPFSimulationConfig('first', t_msConfig, t_filter, t_observation_dt, t_observation_var, t_integration_var, Ns, integration_jitter, resampling_thr)
            outputSDir = outputSubDir(t_msConfig.get('output', 'directory')+'firstStage/', t_observation_dt, t_observation_var, t_integration_var)
            configFN   = outputSDir+PFLabel(t_filter, Ns, resampling_thr, integration_jitter)+'.cfg'
            configFile = open(configFN, 'w')
            config.write(configFile)
            configFile.close()

            #FSConfigs.append(configFN)
            t_FSConfigs[(t_observation_dt, t_observation_var, t_integration_var)][(t_filter, resampling_thr, Ns, integration_jitter)] = configFN

    #return FSConfigs

#__________________________________________________

def prepareFilterFirstStageRun(t_msConfig, t_filter, t_observation_dt, t_observation_var, t_integration_var, t_FSConfigs):
    # build configs for first stage run of a filter

    if 'en' in t_filter and 'kf' in t_filter:
        #return prepareENKFFirstStageRun(t_msConfig, t_filter, t_observation_dt, t_observation_var, t_integration_var)
        prepareENKFFirstStageRun(t_msConfig, t_filter, t_observation_dt, t_observation_var, t_integration_var, t_FSConfigs)

    elif 'pf' in t_filter:
        #return preparePFFirstStageRun(t_msConfig, t_filter, t_observation_dt, t_observation_var, t_integration_var)
        preparePFFirstStageRun(t_msConfig, t_filter, t_observation_dt, t_observation_var, t_integration_var, t_FSConfigs)

#__________________________________________________

def prepareFirstStageRun(t_msConfig):
    # build configs for first stage run

    createDir(t_msConfig.get('output', 'directory'))

    #FSConfigs       = []
    FSConfigs       = {}

    observation_dt  = makeNumpyArray(eval(t_msConfig.get('observation', 'dt')))
    observation_var = makeNumpyArray(eval(t_msConfig.get('observation', 'variance')))
    integration_var = makeNumpyArray(eval(t_msConfig.get('integration', 'variance')))

    afilters        = availableFiltersFromConfig(t_msConfig)

    for (obs_dt, obs_var, int_var) in product(observation_dt, observation_var, integration_var):

        createDir(outputSubDir(t_msConfig.get('output', 'directory')+'firstStage/', obs_dt, obs_var, int_var))
        FSConfigs[(obs_dt, obs_var, int_var)] = {}

        for f in afilters:
            #FSConfigs.extend(prepareFilterFirstStageRun(t_msConfig, f, obs_dt, obs_var, int_var))
            prepareFilterFirstStageRun(t_msConfig, f, obs_dt, obs_var, int_var, FSConfigs)

    FSConfigFile = open(t_msConfig.get('output', 'directory')+'first-stage-configs.dat', 'w')
    #for fsc in FSConfigs:
        #FSConfigFile.write(fsc+'\n')
    for truthParameters in FSConfigs:
        for filterParameters in FSConfigs[truthParameters]:
            FSConfigFile.write(FSConfigs[truthParameters][filterParameters]+'\n')
    FSConfigFile.close()

    return FSConfigs

#__________________________________________________

