#! /usr/bin/env python

#__________________________________________________
# launcher/utils/
# simulationoutput.py
#__________________________________________________
# author        : colonel
# last modified : 2016/10/26
#__________________________________________________
#
# functions related to the output of a simulation
# equivalents are to be found in pyLorenz/utils/output/fileNames.py
# FilterFlavorList() returns the list of filters implemented in pyLorenz/importall.py
#

#__________________________________________________

def EnKFLabel(t_filter, t_Ns, t_inflation, t_integration_jitter):
    # generic EnKF label
    Ns        = str(t_Ns)
    inflation = str(t_inflation).replace('.', 'p')
    jitter    = str(t_integration_jitter).replace('.', 'p')
    return ( t_filter + '_' + Ns + '_' + inflation + '_' + jitter )

#__________________________________________________

def PFLabel(t_filter, t_Ns, t_resampling_thr, t_integration_jitter):
    # generic particle filter label
    Ns      = str(t_Ns)
    res_thr = str(t_resampling_thr).replace('.', 'p')
    jitter  = str(t_integration_jitter).replace('.', 'p')
    return ( t_filter + '_' + Ns + '_' + res_thr + '_' + jitter )

#__________________________________________________

def outputSubDir(t_outputDir, t_observation_dt, t_observation_var, t_integration_var):
    # output sub dir
    obs_dt  = str(t_observation_dt).replace('.', 'p')
    obs_var = str(t_observation_var).replace('.', 'p')
    int_var = str(t_integration_var).replace('.', 'p')
    return ( t_outputDir + obs_dt + '/' + obs_var + '/' + int_var + '/' )

#__________________________________________________

def analyseRMSEFileName(t_outputSubDir, t_filterLabel):
    return t_outputSubDir + t_filterLabel + '_analyseRMSE.bin'

#__________________________________________________

def ENKFFlavorList():
    # list of all EnKF
    return ['senkf', 'entkf', 'entkfn-dual-capped', 'entkfn-dual']

#__________________________________________________

def PFFlavorList():
    # list of all PF
    return ['sirpf', 'asirpf', 'oisirpf']

#__________________________________________________

def FilterFlavorList():
    # list of all filters
    flist = ENKFFlavorList()
    flist.extend(PFFlavorList())
    return flist

#__________________________________________________

