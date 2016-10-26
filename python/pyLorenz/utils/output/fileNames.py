#! /usr/bin/env python

#__________________________________________________
# pyLorenz/utils/output/
# fileNames.py
#__________________________________________________
# author        : colonel
# last modified : 2016/10/23
#__________________________________________________
#
# fonctions used to determine file names from config
#

#__________________________________________________

def outputSubDir(t_config):
    # output sub directory
    outputDir = t_config.get('output', 'directory')
    obs_dt    = str(eval(t_config.get('observation', 'dt'))).replace('.', 'p')
    obs_var   = str(eval(t_config.get('observation', 'variance'))).replace('.', 'p')
    int_var   = str(eval(t_config.get('integration', 'variance'))).replace('.', 'p')
    return ( outputDir + obs_dt + '/' + obs_var + '/' + int_var + '/' )

#__________________________________________________

def EnKFLabel(t_config):
    # generic EnKF label
    flavor    = t_config.get('assimilation', 'filter')
    Ns        = str(t_config.getint('assimilation', 'Ns'))
    inflation = str(eval(t_config.get('assimilation', 'inflation'))).replace('.', 'p')
    jitter    = str(eval(t_config.get('assimilation', 'integration_jitter'))).replace('.', 'p')
    return ( flavor + '_' + Ns + '_' + inflation + '_' + jitter )

#__________________________________________________

def PFLabel(t_config):
    # generic particle filter label
    flavor  = t_config.get('assimilation', 'filter')
    Ns      = str(t_config.getint('assimilation', 'Ns'))
    res_thr = str(eval(t_config.get('assimilation', 'resampling_thr'))).replace('.', 'p')
    jitter  = str(eval(t_config.get('assimilation', 'integration_jitter'))).replace('.', 'p')
    return ( flavor + '_' + Ns + '_' + res_thr + '_' + jitter )
    
#__________________________________________________

def filterLabel(t_config):
    # generic filter label
    flavor = t_config.get('assimilation', 'filter')
    if 'en' in flavor and 'kf' in flavor:
        return EnKFLabel(t_config, 'assimilation')
    elif 'pf' in flavor:
        return PFLabel(t_config, 'assimilation')

#__________________________________________________

