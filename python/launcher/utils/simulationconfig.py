#! /usr/bin/env python

#__________________________________________________
# launcher/utils/
# simulationconfig.py
#__________________________________________________
# author        : colonel
# last modified : 2016/10/26
#__________________________________________________
#
# functions to build simulation config from multi simulation config
#

from ConfigParser import SafeConfigParser

#__________________________________________________

def buildSimulationConfig(t_firstOrSecondStage, t_msConfig, t_filter, t_observation_dt, t_observation_var, t_integration_var):
    # build simulation config from multi simulation config (msConfig) for a filter with given parameters

    config = SafeConfigParser()

    # get general sections from multi simulation config
    for section in ['dimensions', 'initialisation', 'model', 'integration', 'observation', 'output']:
        config.add_section(section)
        for option in t_msConfig.options(section):
            config.set(section, option, t_msConfig.get(section, option))

    # fill parameters
    config.set('integration', 'variance', str(t_integration_var))
    config.set('observation', 'variance', str(t_observation_var))
    config.set('observation', 'dt', str(t_observation_dt))

    # for first stage modify a few options
    if t_firstOrSecondStage == 'first':
        config.set('observation', 'Nt', t_msConfig.get('first-stage', 'Nt'))
        config.set('output', 'class', t_msConfig.get('first-stage', 'output_class'))
    
    # append firstStage/ or secondStage/ to outputDir
    config.set('output', 'directory', t_msConfig.get('output', 'directory')+t_firstOrSecondStage+'Stage/')

    # filter section
    config.add_section('assimilation')
    config.set('assimilation', 'filter', t_filter)
    for option in t_msConfig.options(t_filter):
        config.set('assimilation', option, t_msConfig.get(t_filter, option))
    # remove multi simulation specifics options
    config.remove_option('assimilation', 'enable')
    config.remove_option('assimilation', 'first_stage_variable')
    config.remove_option('assimilation', 'second_stage_size')

    return config

#__________________________________________________

def buildEnKFSimulationConfig(t_firstOrSecondStage, t_msConfig, t_filter, t_observation_dt, t_observation_var, t_integration_var, t_Ns, t_integration_jitter, t_inflation):
    # build sub config for an EnKF with given parameters

    config = buildSimulationConfig(t_firstOrSecondStage, t_msConfig, t_filter, t_observation_dt, t_observation_var, t_integration_var)

    # EnKF specific parameters
    config.set('assimilation', 'Ns', str(t_Ns))
    config.set('assimilation', 'integration_jitter', str(t_integration_jitter))
    config.set('assimilation', 'inflation', str(t_inflation))

    return config

#__________________________________________________

def buildPFSimulationConfig(t_firstOrSecondStage, t_msConfig, t_filter, t_observation_dt, t_observation_var, t_integration_var, t_Ns, t_integration_jitter, t_resampling_thr):
    # build sub config for an EnKF with given parameters

    config = buildSimulationConfig(t_firstOrSecondStage, t_msConfig, t_filter, t_observation_dt, t_observation_var, t_integration_var)

    # PF specifics
    config.set('assimilation', 'Ns', str(t_Ns))
    config.set('assimilation', 'integration_jitter', str(t_integration_jitter))
    config.set('assimilation', 'resampling_thr', str(t_resampling_thr))

    return config

#__________________________________________________

